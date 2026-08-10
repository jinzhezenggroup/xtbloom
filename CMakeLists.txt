cmake_minimum_required(VERSION 3.24)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/xtbloom_version.cmake)
xtbloom_resolve_version(XTBLOOM_VERSION_RELEASE XTBLOOM_VERSION_FULL)

project(
  xtbloom
  VERSION ${XTBLOOM_VERSION_RELEASE}
  DESCRIPTION "Batched CPU and GPU inference for GFN-xTB"
  LANGUAGES C CXX
)

# The product version follows Git tags. ABI generation and SONAME changes are
# independent compatibility decisions and must remain manually reviewed.
set(XTBLOOM_ABI_SOVERSION 0)
set(XTBLOOM_GENERATED_INCLUDE_DIR
  ${CMAKE_CURRENT_BINARY_DIR}/generated/include
)
file(MAKE_DIRECTORY ${XTBLOOM_GENERATED_INCLUDE_DIR}/xtbloom)
configure_file(
  cmake/xtbloom_version.h.in
  ${XTBLOOM_GENERATED_INCLUDE_DIR}/xtbloom/version.h
  @ONLY
)
# Several provider-free ABI tests compile the public header without linking the
# library target. Make the one generated public header visible consistently.
include_directories(BEFORE ${XTBLOOM_GENERATED_INCLUDE_DIR})

include(CheckLanguage)
include(CheckLibraryExists)
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/xtbloom_cuda_toolkit.cmake)
find_package(Threads REQUIRED)

set(XTBLOOM_ENABLE_CUDA "AUTO" CACHE STRING "Build the CUDA backend: AUTO, ON, or OFF")
set_property(CACHE XTBLOOM_ENABLE_CUDA PROPERTY STRINGS AUTO ON OFF)
option(XTBLOOM_BUILD_TESTS "Build xtbloom tests" ${PROJECT_IS_TOP_LEVEL})
option(XTBLOOM_BUILD_WEB_DEMO
  "Build the GitHub Pages WebAssembly demo assets (requires the Emscripten toolchain)"
  OFF)
option(XTBLOOM_PYTHON_WHEEL
  "Install an unversioned single-copy shared library for a Python wheel" OFF)
option(XTBLOOM_BUNDLE_WHEEL_OPENBLAS
  "Build the private OpenBLAS shim whose dependencies auditwheel vendors" OFF)
set(XTBLOOM_CPU_LINALG_LIBRARY "" CACHE FILEPATH
  "Optional monolithic shared LP64 LAPACKE+CBLAS runtime with local thread control for the CPU eigensolver (e.g. libscipy_openblas.so or libmkl_rt.so). When unset, compatible runtimes are auto-discovered with find_package(BLAS/LAPACK)."
)

string(TOUPPER "${XTBLOOM_ENABLE_CUDA}" XTBLOOM_ENABLE_CUDA_NORMALIZED)
if(NOT XTBLOOM_ENABLE_CUDA_NORMALIZED MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "XTBLOOM_ENABLE_CUDA must be AUTO, ON, or OFF")
endif()

set(XTBLOOM_HAS_CUDA OFF)
if(NOT XTBLOOM_ENABLE_CUDA_NORMALIZED STREQUAL "OFF")
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    # FindCUDAToolkit is optional provider discovery: its required variable
    # set includes libcudart, while xtbloom can build without provider shared
    # libraries. The enabled compiler remains the authority for toolkit
    # version and the cudart header set.
    find_package(CUDAToolkit QUIET)
    xtbloom_resolve_cuda_toolkit_metadata(
      XTBLOOM_CUDA_TOOLKIT_VERSION
      XTBLOOM_CUDA_TOOLKIT_MAJOR
      XTBLOOM_CUDA_TOOLKIT_MINOR
      XTBLOOM_CUDA_TOOLKIT_INCLUDE_DIRS)
    xtbloom_validate_cuda_runtime_headers(${XTBLOOM_CUDA_TOOLKIT_INCLUDE_DIRS})
    set(XTBLOOM_HAS_CUDA ON)
  elseif(XTBLOOM_ENABLE_CUDA_NORMALIZED STREQUAL "ON")
    message(FATAL_ERROR "XTBLOOM_ENABLE_CUDA=ON, but no CUDA compiler was found")
  endif()
endif()

# The eigensolver requires one dlopen-able runtime that provides LAPACKE and
# CBLAS together. FindBLAS/FindLAPACK may also report split reference libraries
# (for example liblapack.so plus libblas.so), so validate every candidate as a
# self-contained runtime before baking its path into the library or enabling
# production-backend tests.
function(xtbloom_cpu_linalg_has_symbol_set candidate symbol_prefix result)
  set(symbol_set_complete ON)
  foreach(symbol IN ITEMS
      LAPACKE_dpotrf_work
      LAPACKE_dpocon_work
      LAPACKE_dsyevd_work
      cblas_dtrsm
      cblas_dgemm
  )
    unset(XTBLOOM_CPU_LINALG_SYMBOL_FOUND CACHE)
    check_library_exists(
      "${candidate}" "${symbol_prefix}${symbol}" ""
      XTBLOOM_CPU_LINALG_SYMBOL_FOUND
    )
    if(NOT XTBLOOM_CPU_LINALG_SYMBOL_FOUND)
      set(symbol_set_complete OFF)
      unset(XTBLOOM_CPU_LINALG_SYMBOL_FOUND CACHE)
      break()
    endif()
    unset(XTBLOOM_CPU_LINALG_SYMBOL_FOUND CACHE)
  endforeach()
  set(${result} ${symbol_set_complete} PARENT_SCOPE)
endfunction()

function(xtbloom_cpu_linalg_has_local_thread_control candidate result)
  # xtbloom parallelizes over systems. Accept a provider only when each worker
  # can suppress nested BLAS threads without mutating another worker's state.
  set(has_local_thread_control OFF)
  foreach(symbol IN ITEMS
      MKL_Set_Num_Threads_Local
      openblas_set_num_threads_local
      scipy_openblas_set_num_threads_local
  )
    unset(XTBLOOM_CPU_LINALG_THREAD_CONTROL_FOUND CACHE)
    check_library_exists(
      "${candidate}" "${symbol}" ""
      XTBLOOM_CPU_LINALG_THREAD_CONTROL_FOUND
    )
    if(XTBLOOM_CPU_LINALG_THREAD_CONTROL_FOUND)
      set(has_local_thread_control ON)
      unset(XTBLOOM_CPU_LINALG_THREAD_CONTROL_FOUND CACHE)
      break()
    endif()
    unset(XTBLOOM_CPU_LINALG_THREAD_CONTROL_FOUND CACHE)
  endforeach()
  set(${result} ${has_local_thread_control} PARENT_SCOPE)
endfunction()

function(xtbloom_validate_cpu_linalg_runtime candidate result symbol_prefix_result)
  if(NOT EXISTS "${candidate}" OR IS_DIRECTORY "${candidate}")
    set(${result} OFF PARENT_SCOPE)
    set(${symbol_prefix_result} "" PARENT_SCOPE)
    return()
  endif()

  # check_library_exists can successfully link a static archive, but the
  # production backend must be able to dlopen the exact configured path.
  get_filename_component(candidate_realpath "${candidate}" REALPATH)
  string(TOLOWER "${candidate}" candidate_lower)
  string(TOLOWER "${candidate_realpath}" candidate_realpath_lower)
  if(candidate_lower MATCHES "\\.a$" OR candidate_realpath_lower MATCHES "\\.a$")
    set(${result} OFF PARENT_SCOPE)
    set(${symbol_prefix_result} "" PARENT_SCOPE)
    return()
  endif()
  # OpenBLAS publishes its integer ABI in the get_config string embedded in
  # the shared object. Reject INTERFACE64 builds before tests link to them and
  # call the provider directly with xtbloom's 32-bit LapackInt declarations.
  file(STRINGS "${candidate_realpath}" ilp64_markers REGEX "USE64BITINT" LIMIT_COUNT 1)
  if(ilp64_markers)
    set(${result} OFF PARENT_SCOPE)
    set(${symbol_prefix_result} "" PARENT_SCOPE)
    return()
  endif()

  xtbloom_cpu_linalg_has_local_thread_control(
    "${candidate}" has_local_thread_control
  )
  if(NOT has_local_thread_control)
    set(${result} OFF PARENT_SCOPE)
    set(${symbol_prefix_result} "" PARENT_SCOPE)
    return()
  endif()

  xtbloom_cpu_linalg_has_symbol_set("${candidate}" "" has_standard_symbols)
  if(has_standard_symbols)
    set(${result} ON PARENT_SCOPE)
    set(${symbol_prefix_result} "" PARENT_SCOPE)
    return()
  endif()

  # scipy-openblas32 deliberately prefixes its public ABI with ``scipy_`` to
  # avoid interposing on another BLAS already loaded by NumPy or the host.
  xtbloom_cpu_linalg_has_symbol_set("${candidate}" "scipy_" has_scipy_symbols)
  set(${result} ${has_scipy_symbols} PARENT_SCOPE)
  if(has_scipy_symbols)
    set(${symbol_prefix_result} "scipy_" PARENT_SCOPE)
  else()
    set(${symbol_prefix_result} "" PARENT_SCOPE)
  endif()
endfunction()

# Linux wheels carry a reviewed private OpenBLAS cohort rather than requiring
# end users to install scipy-openblas32. The provider package exists only in
# the isolated PEP 517 build environment; this configure-time step verifies
# every byte without importing its side-effectful Python module. A private shim
# gives auditwheel a DT_NEEDED edge from which it vendors and collision-renames
# the complete provider dependency closure.
set(XTBLOOM_WHEEL_OPENBLAS OFF)
if(XTBLOOM_BUNDLE_WHEEL_OPENBLAS AND NOT XTBLOOM_PYTHON_WHEEL)
  message(FATAL_ERROR
    "XTBLOOM_BUNDLE_WHEEL_OPENBLAS requires XTBLOOM_PYTHON_WHEEL=ON")
endif()
if(XTBLOOM_BUNDLE_WHEEL_OPENBLAS AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" xtbloom_wheel_processor)
  if(xtbloom_wheel_processor MATCHES "^(x86_64|amd64)$")
    set(XTBLOOM_WHEEL_OPENBLAS_ARCH x86_64)
  elseif(xtbloom_wheel_processor MATCHES "^(aarch64|arm64)$")
    set(XTBLOOM_WHEEL_OPENBLAS_ARCH aarch64)
  else()
    message(FATAL_ERROR
      "Python wheels support private OpenBLAS only on Linux x86_64/aarch64, not "
      "'${CMAKE_SYSTEM_PROCESSOR}'")
  endif()
  if(XTBLOOM_CPU_LINALG_LIBRARY)
    message(FATAL_ERROR
      "XTBLOOM_CPU_LINALG_LIBRARY cannot override the reviewed private provider "
      "in XTBLOOM_PYTHON_WHEEL builds")
  endif()

  # dlmopen is part of the wheel's host-isolation contract, not an optional
  # optimization. Prove both namespace creation and dlinfo inspection before
  # accepting the build input.
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_DL_LIBS})
  check_cxx_source_compiles([=[
    #ifndef _GNU_SOURCE
    #define _GNU_SOURCE 1
    #endif
    #include <dlfcn.h>
    #include <link.h>
    int main() {
      void* handle = dlmopen(LM_ID_NEWLM, nullptr, RTLD_NOW | RTLD_LOCAL);
      Lmid_t namespace_id = LM_ID_BASE;
      return handle == nullptr ? 0 : dlinfo(handle, RTLD_DI_LMID, &namespace_id);
    }
  ]=] XTBLOOM_HAS_DLMOPEN_NAMESPACE)
  cmake_pop_check_state()
  if(NOT XTBLOOM_HAS_DLMOPEN_NAMESPACE)
    message(FATAL_ERROR
      "XTBLOOM_PYTHON_WHEEL requires glibc dlmopen/dlinfo namespace isolation")
  endif()

  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(XTBLOOM_OPENBLAS_PROVENANCE
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/3rdparty/scipy_openblas32_manifest.json")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/python/ci/resolve-openblas-wheel.py"
      --manifest "${XTBLOOM_OPENBLAS_PROVENANCE}"
      --architecture "${XTBLOOM_WHEEL_OPENBLAS_ARCH}"
    RESULT_VARIABLE xtbloom_openblas_resolve_status
    OUTPUT_VARIABLE xtbloom_openblas_resolve_output
    ERROR_VARIABLE xtbloom_openblas_resolve_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT xtbloom_openblas_resolve_status EQUAL 0)
    message(FATAL_ERROR
      "failed to resolve the reviewed scipy-openblas32 build input:\n"
      "${xtbloom_openblas_resolve_error}")
  endif()
  string(JSON XTBLOOM_WHEEL_OPENBLAS_LIBRARY GET
    "${xtbloom_openblas_resolve_output}" provider_path)
  get_filename_component(XTBLOOM_WHEEL_OPENBLAS_DIRECTORY
    "${XTBLOOM_WHEEL_OPENBLAS_LIBRARY}" DIRECTORY)
  set(XTBLOOM_WHEEL_OPENBLAS ON)
  message(STATUS
    "xtbloom wheel OpenBLAS build input: ${XTBLOOM_WHEEL_OPENBLAS_LIBRARY}")
endif()

# Bake a verified absolute path into the library so a non-system LP64 runtime
# can be found without LD_LIBRARY_PATH. This remains a lazy dlopen: libxtbloom
# never gains a hard DT_NEEDED dependency that would make the CUDA backend
# unloadable when no BLAS is installed.
set(XTBLOOM_CPU_LINALG_RUNTIME "")
set(XTBLOOM_CPU_LINALG_SYMBOL_PREFIX "")
if(XTBLOOM_CPU_LINALG_LIBRARY)
  set(XTBLOOM_CONFIGURED_CPU_LINALG_LIBRARY "${XTBLOOM_CPU_LINALG_LIBRARY}")
else()
  set(XTBLOOM_CONFIGURED_CPU_LINALG_LIBRARY "")
endif()

if(XTBLOOM_CONFIGURED_CPU_LINALG_LIBRARY)
  xtbloom_validate_cpu_linalg_runtime(
    "${XTBLOOM_CONFIGURED_CPU_LINALG_LIBRARY}"
    configured_runtime_is_compatible
    configured_runtime_symbol_prefix
  )
  if(NOT configured_runtime_is_compatible)
    message(FATAL_ERROR
      "XTBLOOM_CPU_LINALG_LIBRARY=${XTBLOOM_CONFIGURED_CPU_LINALG_LIBRARY} does not provide a "
      "complete LP64 LAPACKE+CBLAS runtime with local thread control"
    )
  endif()
  set(XTBLOOM_CPU_LINALG_RUNTIME "${XTBLOOM_CONFIGURED_CPU_LINALG_LIBRARY}")
  set(XTBLOOM_CPU_LINALG_SYMBOL_PREFIX "${configured_runtime_symbol_prefix}")
elseif(NOT XTBLOOM_WHEEL_OPENBLAS)
  if(DEFINED BLA_SIZEOF_INTEGER)
    set(XTBLOOM_SAVED_BLA_SIZEOF_INTEGER "${BLA_SIZEOF_INTEGER}")
    set(XTBLOOM_HAD_BLA_SIZEOF_INTEGER ON)
  else()
    set(XTBLOOM_HAD_BLA_SIZEOF_INTEGER OFF)
  endif()
  # xtbloom's public and internal eigensolver ABI uses 32-bit LAPACK integers;
  # prevent FindBLAS/FindLAPACK from selecting an ILP64 provider automatically.
  set(BLA_SIZEOF_INTEGER 4)
  find_package(BLAS QUIET)
  if(BLAS_FOUND)
    find_package(LAPACK QUIET)
  endif()
  if(XTBLOOM_HAD_BLA_SIZEOF_INTEGER)
    set(BLA_SIZEOF_INTEGER "${XTBLOOM_SAVED_BLA_SIZEOF_INTEGER}")
  else()
    unset(BLA_SIZEOF_INTEGER)
  endif()
  set(XTBLOOM_CPU_LINALG_CANDIDATES "")
  foreach(candidate IN LISTS LAPACK_LIBRARIES BLAS_LIBRARIES)
    if(NOT IS_ABSOLUTE "${candidate}")
      continue()
    endif()
    string(TOLOWER "${candidate}" candidate_lower)
    if(candidate_lower MATCHES "intel|mkl_rt|openblas|scipy_openblas")
      list(APPEND XTBLOOM_CPU_LINALG_CANDIDATES "${candidate}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES XTBLOOM_CPU_LINALG_CANDIDATES)
  foreach(candidate IN LISTS XTBLOOM_CPU_LINALG_CANDIDATES)
    xtbloom_validate_cpu_linalg_runtime(
      "${candidate}" candidate_is_compatible candidate_symbol_prefix
    )
    if(candidate_is_compatible)
      set(XTBLOOM_CPU_LINALG_RUNTIME "${candidate}")
      set(XTBLOOM_CPU_LINALG_SYMBOL_PREFIX "${candidate_symbol_prefix}")
      break()
    endif()
    message(STATUS
      "xtbloom ignored incomplete CPU linear-algebra runtime: ${candidate}"
    )
  endforeach()
endif()
if(XTBLOOM_CPU_LINALG_RUNTIME)
  message(STATUS "xtbloom CPU eigensolver BLAS/LAPACK runtime: ${XTBLOOM_CPU_LINALG_RUNTIME}")
endif()

# Host-isolated MKL provider shim.
#
# libmkl_rt is an interface-layer dispatcher: loading it and calling
# MKL_Set_Interface_Layer mutates process-global MKL state that an embedding
# application may already own, and MKL_INTERFACE_LAYER can make xtbloom depend on
# the host's MKL lifecycle. When the selected runtime is MKL, build a tiny
# private shim with fixed DT_NEEDED dependencies on libmkl_intel_lp64,
# libmkl_sequential, and libmkl_core. Loading those components directly (never
# libmkl_rt) yields an LP64 + sequential provider. The factory loads the sibling
# shim in a new glibc link-map namespace with RTLD_LOCAL: RTLD_LOCAL by itself
# still permits a globally loaded host libmkl_rt to interpose on dependencies.
# The shim is lazy and never adds a hard DT_NEEDED dependency to libxtbloom.
set(XTBLOOM_CPU_LINALG_SHIM OFF)
set(XTBLOOM_CPU_LINALG_RUNTIME_IS_MKL OFF)
if(XTBLOOM_CPU_LINALG_RUNTIME)
  get_filename_component(xtbloom_linalg_runtime_name
    "${XTBLOOM_CPU_LINALG_RUNTIME}" NAME
  )
  if(xtbloom_linalg_runtime_name MATCHES "mkl_rt")
    set(XTBLOOM_CPU_LINALG_RUNTIME_IS_MKL ON)

    # Namespace isolation depends on GNU-compatible dlmopen/dlinfo support,
    # not merely on the Linux platform name. Prove the complete compile/link
    # interface before accepting an MKL provider.
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_DL_LIBS})
    check_cxx_source_compiles([=[
      #ifndef _GNU_SOURCE
      #define _GNU_SOURCE 1
      #endif
      #include <dlfcn.h>
      #include <link.h>
      int main() {
        void* handle = dlmopen(LM_ID_NEWLM, nullptr, RTLD_NOW | RTLD_LOCAL);
        Lmid_t namespace_id = LM_ID_BASE;
        return handle == nullptr ? 0 : dlinfo(handle, RTLD_DI_LMID, &namespace_id);
      }
    ]=] XTBLOOM_HAS_DLMOPEN_NAMESPACE)
    cmake_pop_check_state()

    get_filename_component(xtbloom_linalg_runtime_realpath
      "${XTBLOOM_CPU_LINALG_RUNTIME}" REALPATH
    )
    get_filename_component(xtbloom_mkl_component_dir
      "${xtbloom_linalg_runtime_realpath}" DIRECTORY
    )
    get_filename_component(xtbloom_linalg_runtime_realname
      "${xtbloom_linalg_runtime_realpath}" NAME
    )
    get_filename_component(xtbloom_mkl_runtime_suffix
      "${xtbloom_linalg_runtime_realname}" EXT
    )

    # All three components must be the exact SONAME cohort adjacent to the
    # selected runtime. Mixing files across MKLROOT or unrelated installations
    # can bind an ABI-incoherent provider even when every filename exists.
    set(XTBLOOM_HAS_MKL_COMPONENTS ON)
    foreach(component IN ITEMS intel_lp64 sequential core)
      set(component_path
        "${xtbloom_mkl_component_dir}/libmkl_${component}${xtbloom_mkl_runtime_suffix}"
      )
      if(NOT EXISTS "${component_path}" OR IS_DIRECTORY "${component_path}")
        set(XTBLOOM_HAS_MKL_COMPONENTS OFF)
      endif()
      set("XTBLOOM_MKL_${component}_LIB" "${component_path}")
    endforeach()

    if(XTBLOOM_HAS_MKL_COMPONENTS AND XTBLOOM_HAS_DLMOPEN_NAMESPACE)
      add_library(xtbloom_mkl_lp64_shim SHARED
        src/runtime/mkl_lp64_shim.cpp
      )
      target_compile_features(xtbloom_mkl_lp64_shim PRIVATE cxx_std_17)
      set_target_properties(xtbloom_mkl_lp64_shim PROPERTIES
        EXPORT_NAME mkl_lp64_shim
        OUTPUT_NAME xtbloom_mkl_lp64_shim
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
      )
      get_filename_component(xtbloom_mkl_component_dir
        "${XTBLOOM_MKL_intel_lp64_LIB}" DIRECTORY
      )
      target_link_libraries(xtbloom_mkl_lp64_shim PRIVATE
        "${XTBLOOM_MKL_intel_lp64_LIB}"
        "${XTBLOOM_MKL_sequential_LIB}"
        "${XTBLOOM_MKL_core_LIB}"
      )
      # The shim itself references none of the MKL symbols directly; its ELF
      # NEEDED records are what make dlopen resolve the whole LP64/sequential
      # component scope together. Force the records even on --as-needed hosts.
      # Emit DT_RPATH deliberately: DT_RUNPATH is searched after LD_LIBRARY_PATH
      # and would allow a same-SONAME component from another MKL installation to
      # replace one member of the configure-time cohort.
      target_link_options(xtbloom_mkl_lp64_shim PRIVATE
        "LINKER:--no-as-needed"
        "LINKER:--disable-new-dtags"
      )
      set_target_properties(xtbloom_mkl_lp64_shim PROPERTIES
        BUILD_RPATH "${xtbloom_mkl_component_dir}"
        INSTALL_RPATH "${xtbloom_mkl_component_dir}"
        # The runtime resolves only a sibling shim. Keep build-tree tests and
        # installed shared libraries on the same relocation-safe contract.
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
      )
      install(TARGETS xtbloom_mkl_lp64_shim EXPORT xtbloomTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      )
      set(XTBLOOM_CPU_LINALG_SHIM ON)
      message(STATUS
        "xtbloom host-isolated MKL LP64 provider shim: enabled"
      )
    else()
      set(mkl_isolation_error
        "XTBLOOM CPU MKL isolation requires dlmopen/dlinfo namespace support "
        "and the exact adjacent libmkl_intel_lp64, libmkl_sequential, and "
        "libmkl_core SONAME cohort"
      )
      if(XTBLOOM_CONFIGURED_CPU_LINALG_LIBRARY)
        message(FATAL_ERROR "${mkl_isolation_error}")
      endif()
      message(STATUS "${mkl_isolation_error}; disabling the auto-discovered runtime")
      set(XTBLOOM_CPU_LINALG_RUNTIME "")
      set(XTBLOOM_CPU_LINALG_SYMBOL_PREFIX "")
      set(XTBLOOM_CPU_LINALG_RUNTIME_IS_MKL OFF)
    endif()
  endif()
endif()

if(XTBLOOM_WHEEL_OPENBLAS)
  add_library(xtbloom_openblas_lp64_shim SHARED
    src/runtime/openblas_lp64_shim.c
  )
  target_link_libraries(xtbloom_openblas_lp64_shim PRIVATE
    "${XTBLOOM_WHEEL_OPENBLAS_LIBRARY}"
  )
  target_link_options(xtbloom_openblas_lp64_shim PRIVATE
    "LINKER:--no-as-needed"
  )
  set_target_properties(xtbloom_openblas_lp64_shim PROPERTIES
    OUTPUT_NAME xtbloom_openblas_lp64_shim
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    BUILD_RPATH "${XTBLOOM_WHEEL_OPENBLAS_DIRECTORY}"
    # repair-wheel.sh supplies the reviewed directory through LD_LIBRARY_PATH;
    # never record a temporary build-environment path in the unrepaired wheel.
    INSTALL_RPATH ""
  )
  install(TARGETS xtbloom_openblas_lp64_shim
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

add_library(xtbloom_gfn2_cpu OBJECT
  src/backends/common/gfn2_plan_schema.cpp
  src/model/gfn2/aes2.cpp
  src/model/gfn2/basis.cpp
  src/model/gfn2/coordination.cpp
  src/model/gfn2/d4.cpp
  src/model/gfn2/eigensolver.cpp
  src/model/gfn2/es2.cpp
  src/model/gfn2/es3.cpp
  src/model/gfn2/external_point_charges.cpp
  src/model/gfn2/force.cpp
  src/model/gfn2/h0.cpp
  src/model/gfn2/integrals.cpp
  src/model/gfn2/mulliken.cpp
  src/model/gfn2/periodic_embedding.cpp
  src/model/gfn2/repulsion.cpp
  src/model/gfn2/scc_driver.cpp
  src/model/gfn2/scc_mixer.cpp
  src/model/gfn2/spin.cpp
  src/model/gfn2/wavefunction.cpp
)
target_compile_features(xtbloom_gfn2_cpu PUBLIC cxx_std_17)
target_include_directories(xtbloom_gfn2_cpu PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
set_target_properties(xtbloom_gfn2_cpu PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  POSITION_INDEPENDENT_CODE ON
  VISIBILITY_INLINES_HIDDEN YES
)
if(CMAKE_DL_LIBS)
  target_link_libraries(xtbloom_gfn2_cpu PUBLIC ${CMAKE_DL_LIBS})
endif()
if(XTBLOOM_CPU_LINALG_RUNTIME)
  target_compile_definitions(xtbloom_gfn2_cpu PRIVATE
    "XTBLOOM_CONFIGURED_CPU_LINALG_RUNTIME=\"${XTBLOOM_CPU_LINALG_RUNTIME}\""
  )
endif()
if(XTBLOOM_CPU_LINALG_RUNTIME_IS_MKL)
  target_compile_definitions(xtbloom_gfn2_cpu PRIVATE
    XTBLOOM_CONFIGURED_CPU_LINALG_MKL=1
  )
endif()
if(XTBLOOM_CPU_LINALG_SHIM)
  target_compile_definitions(xtbloom_gfn2_cpu PRIVATE
    XTBLOOM_CONFIGURED_CPU_LINALG_SHIM=1
  )
  add_dependencies(xtbloom_gfn2_cpu xtbloom_mkl_lp64_shim)
endif()
if(XTBLOOM_WHEEL_OPENBLAS)
  target_compile_definitions(xtbloom_gfn2_cpu PRIVATE
    XTBLOOM_CONFIGURED_WHEEL_OPENBLAS=1
  )
  add_dependencies(xtbloom_gfn2_cpu xtbloom_openblas_lp64_shim)
endif()

add_library(xtbloom
  src/api.cpp
  src/runtime/backend.cpp
  src/runtime/gfn2_cpu_execution.cpp
  src/runtime/gfn2_plan.cpp
  src/runtime/result_owner.cpp
  src/runtime/validation.cpp
  $<TARGET_OBJECTS:xtbloom_gfn2_cpu>
)
add_library(xtbloom::xtbloom ALIAS xtbloom)

target_compile_features(xtbloom PUBLIC cxx_std_17)
target_include_directories(xtbloom
  PUBLIC
    $<BUILD_INTERFACE:${XTBLOOM_GENERATED_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_definitions(xtbloom PRIVATE XTBLOOM_BUILDING_LIBRARY)
target_link_libraries(xtbloom PRIVATE Threads::Threads)
if(CMAKE_DL_LIBS)
  target_link_libraries(xtbloom PRIVATE ${CMAKE_DL_LIBS})
endif()
set_target_properties(xtbloom PROPERTIES
  CUDA_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
)
if(NOT XTBLOOM_PYTHON_WHEEL)
  # Native SDK installs retain the normal SONAME chain.  Wheel archives often
  # dereference those symlinks and would otherwise store the large CUDA shared
  # object three times, so wheel builds deliberately install one unversioned
  # library instead.
  set_target_properties(xtbloom PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${XTBLOOM_ABI_SOVERSION}
  )
endif()
if(UNIX AND NOT APPLE)
  target_link_options(xtbloom PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/xtbloom.map"
  )
  set_property(TARGET xtbloom APPEND PROPERTY
    LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/xtbloom.map
  )
endif()

if(XTBLOOM_HAS_CUDA)
  # implib's hand-written trampolines are GNU ELF assembly for Linux.  Keep the
  # accepted architecture aliases explicit so an unsupported wheel cannot
  # configure successfully and then fail at load time.
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "the CUDA dlopen backend currently requires Linux ELF")
  endif()
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _xtbloom_cuda_processor)
  if(_xtbloom_cuda_processor MATCHES "^(x86_64|amd64)$")
    set(XTBLOOM_CUDA_IMPLIB_TARGET x86_64)
  elseif(_xtbloom_cuda_processor MATCHES "^(aarch64|arm64)$")
    set(XTBLOOM_CUDA_IMPLIB_TARGET aarch64)
  else()
    message(FATAL_ERROR
      "the CUDA dlopen backend supports only Linux x86_64 and aarch64, not "
      "'${CMAKE_SYSTEM_PROCESSOR}'")
  endif()

  # The generated shims contain all CUDA-facing host definitions needed by the
  # backend. Neither the host nor device link may silently reintroduce a static
  # CUDA runtime.
  set(CMAKE_CUDA_RUNTIME_LIBRARY None)
  enable_language(C ASM)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  set(XTBLOOM_CUDA_IMPLIB_GEN
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/3rdparty/implib/implib-gen.py")
  set(XTBLOOM_CUDA_STUBLIB_GEN
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/implib_stubgen.py")
  set(XTBLOOM_CUDA_IMPLIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/cuda_implib")
  file(MAKE_DIRECTORY "${XTBLOOM_CUDA_IMPLIB_DIR}")

  # Curate the object-level names actually referenced by xtbloom.  CUDA headers
  # normalize several source API names to versioned ELF aliases; listing those
  # aliases here keeps -z defs useful and prevents a full-library catch-all from
  # masking a new dependency or an ABI-unsafe fallback category.
  set(XTBLOOM_CUDART_SYMBOLS
    __cudaInitModule
    __cudaPopCallConfiguration
    __cudaPushCallConfiguration
    __cudaRegisterFatBinary
    __cudaRegisterFatBinaryEnd
    __cudaRegisterFunction
    __cudaRegisterVar
    __cudaUnregisterFatBinary
    cudaDeviceGetAttribute
    cudaDeviceSynchronize
    cudaEventCreate
    cudaEventCreateWithFlags
    cudaEventDestroy
    cudaEventElapsedTime
    cudaEventRecord
    cudaEventSynchronize
    cudaFree
    cudaFreeHost
    cudaGetDevice
    cudaGetDeviceCount
    cudaGetErrorName
    cudaGetErrorString
    cudaGetLastError
    cudaGraphAddNode
    cudaGraphChildGraphNodeGetGraph
    cudaGraphConditionalHandleCreate
    cudaGraphCreate
    cudaGraphDestroy
    cudaGraphExecDestroy
    cudaGraphGetNodes
    cudaGraphInstantiate
    cudaGraphLaunch
    cudaGraphMemcpyNodeGetParams
    cudaGraphNodeGetType
    cudaGraphUpload
    cudaHostGetFlags
    cudaHostRegister
    cudaHostUnregister
    cudaLaunchHostFunc
    cudaLaunchKernel
    cudaMalloc
    cudaMallocHost
    cudaMallocManaged
    cudaMemcpy
    cudaMemcpyAsync
    cudaMemcpyFromSymbol
    cudaMemcpyToSymbol
    cudaMemset
    cudaMemsetAsync
    cudaPeekAtLastError
    cudaPointerGetAttributes
    cudaRuntimeGetVersion
    cudaSetDevice
    cudaStreamBeginCapture
    cudaStreamBeginCaptureToGraph
    cudaStreamCreate
    cudaStreamCreateWithFlags
    cudaStreamDestroy
    cudaStreamEndCapture
    cudaStreamGetCaptureInfo_v2
    cudaStreamGetDevice
    cudaStreamIsCapturing
    cudaStreamSynchronize
    cudaStreamUpdateCaptureDependencies
    cudaStreamWaitEvent
  )
  set(XTBLOOM_CUBLAS_SYMBOLS
    cublasCreate_v2
    cublasDestroy_v2
    cublasDtrsmBatched
    cublasGetMathMode
    cublasGetPointerMode_v2
    cublasGetStream_v2
    cublasGetVersion_v2
    cublasSetMathMode
    cublasSetPointerMode_v2
    cublasSetStream_v2
    cublasSetWorkspace_v2
  )
  set(XTBLOOM_CUSOLVER_SYMBOLS
    cusolverDnCreate
    cusolverDnCreateParams
    cusolverDnCreateSyevjInfo
    cusolverDnDestroy
    cusolverDnDestroyParams
    cusolverDnDestroySyevjInfo
    cusolverDnDormtr
    cusolverDnDormtr_bufferSize
    cusolverDnDpotrfBatched
    cusolverDnDsytrd
    cusolverDnDsytrd_bufferSize
    cusolverDnDsyevjBatched
    cusolverDnDsyevjBatched_bufferSize
    cusolverDnGetStream
    cusolverDnSetStream
    cusolverDnXsyevjSetMaxSweeps
    cusolverDnXsyevjSetSortEig
    cusolverDnXsyevjSetTolerance
    cusolverDnXsyevBatched
    cusolverDnXsyevBatched_bufferSize
    cusolverGetProperty
  )
  set(XTBLOOM_CUDA_DRIVER_SYMBOLS
    cuGetErrorString
    cuMemGetAddressRange_v2
  )

  function(xtbloom_write_implib_symbol_file output_path)
    file(WRITE "${output_path}" "")
    foreach(_xtbloom_symbol IN LISTS ARGN)
      file(APPEND "${output_path}" "${_xtbloom_symbol}\n")
    endforeach()
  endfunction()

  function(xtbloom_read_elf_soname library_path output_variable)
    execute_process(
      COMMAND "${XTBLOOM_READELF}" -dW "${library_path}"
      RESULT_VARIABLE _xtbloom_readelf_status
      OUTPUT_VARIABLE _xtbloom_readelf_output
      ERROR_VARIABLE _xtbloom_readelf_error)
    if(NOT _xtbloom_readelf_status EQUAL 0)
      message(FATAL_ERROR
        "failed to read CUDA library SONAME from '${library_path}': "
        "${_xtbloom_readelf_error}")
    endif()
    string(REGEX MATCH "\\(SONAME\\)[^[]*\\[([^]]+)\\]"
      _xtbloom_soname_match "${_xtbloom_readelf_output}")
    if(NOT _xtbloom_soname_match)
      message(FATAL_ERROR "CUDA library '${library_path}' has no ELF SONAME")
    endif()
    set(${output_variable} "${CMAKE_MATCH_1}" PARENT_SCOPE)
  endfunction()

  # The CUDA provider shared libraries are dlopen'd lazily at runtime and are
  # never linked. When the build environment ships them (the usual toolkit
  # layout), the vendored implib scanner derives each shim and SONAME from the
  # real ELF so the artifacts adapt to the exact toolchain version. When they
  # are absent, xtbloom generates byte-identical shims from the curated symbol
  # lists with pinned SONAMEs (overridable cache variables), so configuring
  # and building the CUDA backend requires no provider shared libraries;
  # nvcc's compiler-support files and the cudart headers still come from the
  # CUDA compiler installation.
  # The runtime cohort preflight in cuda_dlopen.c enforces the deployed names
  # and symbol set either way.
  xtbloom_cuda_default_provider_sonames(
    "${XTBLOOM_CUDA_TOOLKIT_MAJOR}"
    _xtbloom_cuda_default_cudart_soname
    _xtbloom_cuda_default_cublas_soname
    _xtbloom_cuda_default_cusolver_soname
    _xtbloom_cuda_default_driver_soname)
  xtbloom_cache_cuda_provider_soname(
    XTBLOOM_CUDA_CUDART_SONAME "${_xtbloom_cuda_default_cudart_soname}"
    "dlopen'd CUDA runtime SONAME used when the build toolkit ships no CUDA libraries")
  xtbloom_cache_cuda_provider_soname(
    XTBLOOM_CUDA_CUBLAS_SONAME "${_xtbloom_cuda_default_cublas_soname}"
    "dlopen'd cuBLAS SONAME used when the build toolkit ships no CUDA libraries")
  xtbloom_cache_cuda_provider_soname(
    XTBLOOM_CUDA_CUSOLVER_SONAME "${_xtbloom_cuda_default_cusolver_soname}"
    "dlopen'd cuSOLVER SONAME used when the build toolkit ships no CUDA libraries")
  xtbloom_cache_cuda_provider_soname(
    XTBLOOM_CUDA_DRIVER_SONAME "${_xtbloom_cuda_default_driver_soname}"
    "dlopen'd CUDA driver SONAME used when the build toolkit ships no CUDA libraries")
  set(_xtbloom_cuda_providers_present ON)
  foreach(_xtbloom_cuda_target IN ITEMS CUDA::cudart CUDA::cublas CUDA::cusolver)
    if(NOT TARGET ${_xtbloom_cuda_target})
      set(_xtbloom_cuda_providers_present OFF)
    endif()
  endforeach()
  if(NOT CUDA_cuda_driver_LIBRARY)
    set(_xtbloom_cuda_providers_present OFF)
  endif()
  if(_xtbloom_cuda_providers_present)
    message(STATUS
      "xtbloom CUDA provider shims: scanning the complete toolkit library cohort")
  else()
    foreach(_xtbloom_cuda_required_soname_var IN ITEMS
        XTBLOOM_CUDA_CUDART_SONAME
        XTBLOOM_CUDA_CUBLAS_SONAME
        XTBLOOM_CUDA_CUSOLVER_SONAME)
      if("${${_xtbloom_cuda_required_soname_var}}" STREQUAL "")
        message(FATAL_ERROR
          "CUDA ${XTBLOOM_CUDA_TOOLKIT_VERSION} has no curated fallback provider "
          "SONAME mapping; set XTBLOOM_CUDA_CUDART_SONAME, "
          "XTBLOOM_CUDA_CUBLAS_SONAME, and XTBLOOM_CUDA_CUSOLVER_SONAME explicitly")
      endif()
    endforeach()
    message(STATUS
      "xtbloom CUDA provider shims: curated symbols with toolkit-major SONAMEs")
  endif()

  set(_xtbloom_cuda_implib_bases
    libcudart.so libcublas.so libcusolver.so libcuda.so)
  set(_xtbloom_cuda_implib_soname_vars
    XTBLOOM_CUDA_CUDART_SONAME
    XTBLOOM_CUDA_CUBLAS_SONAME
    XTBLOOM_CUDA_CUSOLVER_SONAME
    XTBLOOM_CUDA_DRIVER_SONAME)
  set(_xtbloom_cuda_symbol_sets
    XTBLOOM_CUDART_SYMBOLS
    XTBLOOM_CUBLAS_SYMBOLS
    XTBLOOM_CUSOLVER_SYMBOLS
    XTBLOOM_CUDA_DRIVER_SYMBOLS)

  set(_xtbloom_cuda_implib_inputs "")
  if(_xtbloom_cuda_providers_present)
    find_program(XTBLOOM_READELF NAMES readelf REQUIRED)
    foreach(_xtbloom_cuda_target IN ITEMS CUDA::cudart CUDA::cublas CUDA::cusolver)
      get_target_property(_xtbloom_cuda_location
        "${_xtbloom_cuda_target}" IMPORTED_LOCATION)
      if(NOT _xtbloom_cuda_location)
        message(FATAL_ERROR
          "cannot locate ${_xtbloom_cuda_target} for dlopen shim generation")
      endif()
      list(APPEND _xtbloom_cuda_implib_inputs "${_xtbloom_cuda_location}")
    endforeach()
    list(APPEND _xtbloom_cuda_implib_inputs "${CUDA_cuda_driver_LIBRARY}")
  endif()

  set(XTBLOOM_CUDA_DYNLOADER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/cuda_dlopen.c")
  set(_xtbloom_cuda_sonames "")
  set(_xtbloom_cuda_index 0)
  foreach(_xtbloom_cuda_base _xtbloom_cuda_soname_var _xtbloom_cuda_symbol_set
      IN ZIP_LISTS
        _xtbloom_cuda_implib_bases
        _xtbloom_cuda_implib_soname_vars
        _xtbloom_cuda_symbol_sets)
    set(_xtbloom_cuda_symbol_file
      "${XTBLOOM_CUDA_IMPLIB_DIR}/${_xtbloom_cuda_base}.symbols")
    xtbloom_write_implib_symbol_file(
      "${_xtbloom_cuda_symbol_file}" ${${_xtbloom_cuda_symbol_set}})
    if(_xtbloom_cuda_providers_present)
      list(GET _xtbloom_cuda_implib_inputs ${_xtbloom_cuda_index} _xtbloom_cuda_input)
      get_filename_component(_xtbloom_cuda_input_base "${_xtbloom_cuda_input}" NAME)
      if(NOT _xtbloom_cuda_input_base STREQUAL _xtbloom_cuda_base)
        message(FATAL_ERROR
          "expected ${_xtbloom_cuda_base} from FindCUDAToolkit, got "
          "'${_xtbloom_cuda_input_base}'")
      endif()
      if(_xtbloom_cuda_base STREQUAL "libcuda.so")
        # The driver library may live outside the toolkit tree; its pinned
        # default applies and the runtime preflight still verifies the name.
        set(_xtbloom_cuda_soname "${XTBLOOM_CUDA_DRIVER_SONAME}")
      else()
        xtbloom_read_elf_soname("${_xtbloom_cuda_input}" _xtbloom_cuda_soname)
      endif()
      execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${XTBLOOM_CUDA_IMPLIB_GEN}"
                "${_xtbloom_cuda_input}"
                --target "${XTBLOOM_CUDA_IMPLIB_TARGET}"
                --no-dlopen
                --lazy-load
                --library-load-name "${_xtbloom_cuda_soname}"
                --dlsym-callback xtbloom_cuda_dlsym
                --symbol-list "${_xtbloom_cuda_symbol_file}"
                --outdir "${XTBLOOM_CUDA_IMPLIB_DIR}"
        RESULT_VARIABLE _xtbloom_cuda_implib_result
        OUTPUT_VARIABLE _xtbloom_cuda_implib_output
        ERROR_VARIABLE _xtbloom_cuda_implib_error
        COMMAND_ERROR_IS_FATAL ANY)
    else()
      set(_xtbloom_cuda_soname "${${_xtbloom_cuda_soname_var}}")
      execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${XTBLOOM_CUDA_STUBLIB_GEN}"
                --base-name "${_xtbloom_cuda_base}"
                --symbol-list "${_xtbloom_cuda_symbol_file}"
                --load-name "${_xtbloom_cuda_soname}"
                --target "${XTBLOOM_CUDA_IMPLIB_TARGET}"
                --implib-root "${CMAKE_CURRENT_SOURCE_DIR}/cmake/3rdparty/implib"
                --outdir "${XTBLOOM_CUDA_IMPLIB_DIR}"
        RESULT_VARIABLE _xtbloom_cuda_stublib_result
        OUTPUT_VARIABLE _xtbloom_cuda_stublib_output
        ERROR_VARIABLE _xtbloom_cuda_stublib_error
        COMMAND_ERROR_IS_FATAL ANY)
    endif()
    list(APPEND _xtbloom_cuda_sonames "${_xtbloom_cuda_soname}")
    list(APPEND XTBLOOM_CUDA_DYNLOADER_SOURCES
      "${XTBLOOM_CUDA_IMPLIB_DIR}/${_xtbloom_cuda_base}.tramp.S"
      "${XTBLOOM_CUDA_IMPLIB_DIR}/${_xtbloom_cuda_base}.init.c")
    math(EXPR _xtbloom_cuda_index "${_xtbloom_cuda_index} + 1")
  endforeach()
  list(GET _xtbloom_cuda_sonames 0 XTBLOOM_CUDART_SONAME)
  list(GET _xtbloom_cuda_sonames 1 XTBLOOM_CUBLAS_SONAME)
  list(GET _xtbloom_cuda_sonames 2 XTBLOOM_CUSOLVER_SONAME)
  list(GET _xtbloom_cuda_sonames 3 XTBLOOM_CUDA_DRIVER_SONAME)

  # Runtime bootstrap consumes the same configure-time list used to generate
  # the trampolines, so cohort preflight cannot drift from the link closure.
  set(XTBLOOM_CUDA_DLOPEN_SYMBOLS_HEADER
    "${XTBLOOM_CUDA_IMPLIB_DIR}/cuda_dlopen_symbols.h")
  file(WRITE "${XTBLOOM_CUDA_DLOPEN_SYMBOLS_HEADER}"
    "/* Generated by CMake; do not edit. */\n#pragma once\n")
  foreach(_xtbloom_cuda_name IN ITEMS CUDART CUBLAS CUSOLVER CUDA_DRIVER)
    string(TOLOWER "${_xtbloom_cuda_name}" _xtbloom_cuda_name_lower)
    file(APPEND "${XTBLOOM_CUDA_DLOPEN_SYMBOLS_HEADER}"
      "static const char* const xtbloom_${_xtbloom_cuda_name_lower}_required_symbols[] = {\n")
    foreach(_xtbloom_symbol IN LISTS XTBLOOM_${_xtbloom_cuda_name}_SYMBOLS)
      file(APPEND "${XTBLOOM_CUDA_DLOPEN_SYMBOLS_HEADER}"
        "    \"${_xtbloom_symbol}\",\n")
    endforeach()
    file(APPEND "${XTBLOOM_CUDA_DLOPEN_SYMBOLS_HEADER}"
      "    (const char*)0,\n};\n")
  endforeach()

  add_library(xtbloom_cuda_backend OBJECT
    ${XTBLOOM_CUDA_DYNLOADER_SOURCES}
    src/backends/cuda/cuda_runtime.cu
    src/runtime/cuda_descriptor_validation.cu
    src/runtime/gfn2_cuda_topology_staging.cu
    src/runtime/gfn2_cuda_execution.cu
    src/runtime/result_owner_cuda.cu
    src/backends/cuda/gfn2_aes2.cu
    src/backends/cuda/gfn2_classical_force.cu
    src/backends/cuda/gfn2_d4.cu
    src/backends/cuda/gfn2_density.cu
    src/backends/cuda/gfn2_electronic_gradient.cu
    src/backends/cuda/gfn2_energy_force_execution.cu
    src/backends/cuda/gfn2_eigensolver.cu
    src/backends/cuda/gfn2_es2.cu
    src/backends/cuda/gfn2_es3.cu
    src/backends/cuda/gfn2_external_point_charges.cu
    src/backends/cuda/gfn2_force_composition.cu
    src/backends/cuda/gfn2_geometry.cu
    src/backends/cuda/gfn2_preprocessing.cu
    src/backends/cuda/gfn2_public_result_bridge.cu
    src/backends/cuda/gfn2_h0_force.cu
    src/backends/cuda/gfn2_hamiltonian.cu
    src/backends/cuda/gfn2_hamiltonian_force.cu
    src/backends/cuda/gfn2_inference_publication.cu
    src/backends/cuda/gfn2_integrals.cu
    src/backends/cuda/gfn2_mulliken.cu
    src/backends/cuda/gfn2_occupations.cu
    src/backends/cuda/gfn2_pairlist.cu
    src/backends/cuda/gfn2_parameters.cu
    src/backends/cuda/gfn2_plan_schema.cu
    src/backends/cuda/gfn2_periodic_embedding.cu
    src/backends/cuda/gfn2_post_scc_potential.cu
    src/backends/cuda/gfn2_repulsion.cu
    src/backends/cuda/gfn2_scc.cu
    src/backends/cuda/gfn2_scc_bridge.cu
    src/backends/cuda/gfn2_scc_classical_energy.cu
    src/backends/cuda/gfn2_scc_energy.cu
    src/backends/cuda/gfn2_scc_free_energy.cu
    src/backends/cuda/gfn2_scc_iteration.cu
    src/backends/cuda/gfn2_scc_loop.cu
    src/backends/cuda/gfn2_scc_iteration_arena.cu
    src/backends/cuda/gfn2_scc_iteration_control.cu
    src/backends/cuda/gfn2_scc_iteration_initialize.cu
    src/backends/cuda/gfn2_scc_iteration_reports.cu
    src/backends/cuda/gfn2_scc_mixer.cu
    src/backends/cuda/gfn2_scc_potential.cu
    src/backends/cuda/gfn2_scc_publication.cu
    src/backends/cuda/gfn2_scc_setup_eigensolver.cu
    src/backends/cuda/gfn2_scc_setup_inputs.cu
    src/backends/cuda/gfn2_scc_setup_topology.cu
    src/backends/cuda/gfn2_spin.cu
    src/backends/cuda/gfn2_terminal_classical_energy.cu
    src/backends/cuda/gfn2_total_energy.cu
  )
  target_include_directories(xtbloom_cuda_backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${XTBLOOM_CUDA_TOOLKIT_INCLUDE_DIRS}
  )
  target_include_directories(xtbloom_cuda_backend PRIVATE
    ${XTBLOOM_CUDA_IMPLIB_DIR}
  )
  # The startup bootstrap uses pthread_once and a diagnostic mutex. Preserve
  # that dependency for every executable that consumes the backend objects,
  # not just for the final xtbloom library target.
  target_link_libraries(xtbloom_cuda_backend PUBLIC Threads::Threads)
  # The CUDA runtime/math/driver shared libraries are deliberately NOT linked.
  # The priority-101 bootstrap pre-resolves the complete shim cohort before
  # ordinary NVCC registration constructors run.
  target_compile_definitions(xtbloom_cuda_backend PRIVATE
    XTBLOOM_HAS_CUDA=1
    XTBLOOM_CUDART_SONAME="${XTBLOOM_CUDART_SONAME}"
    XTBLOOM_CUBLAS_SONAME="${XTBLOOM_CUBLAS_SONAME}"
    XTBLOOM_CUSOLVER_SONAME="${XTBLOOM_CUSOLVER_SONAME}"
    XTBLOOM_CUDA_DRIVER_SONAME="${XTBLOOM_CUDA_DRIVER_SONAME}"
  )
  if(XTBLOOM_BUILD_TESTS)
    # White-box CUDA kernels exercise otherwise unreachable transactional
    # failure and overflow boundaries without shipping test entry points in
    # release builds.
    target_compile_definitions(xtbloom_cuda_backend PRIVATE XTBLOOM_CUDA_TEST_HOOKS=1)
  endif()
  set_target_properties(xtbloom_cuda_backend PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_VISIBILITY_PRESET hidden
    CUDA_RUNTIME_LIBRARY None
    C_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
    VISIBILITY_INLINES_HIDDEN YES
  )
  # The committed sparse pair-list CN consumer (gfn2_pairlist.cu) recomputes
  # coordination pair values from live positions while the dense reference
  # (gfn2_geometry.cu) reads the geometry cache.  The sparse candidate is
  # parity-gated bit-for-bit against dense, which only holds when both files
  # compile the pair evaluation with the same FMA contraction; otherwise every
  # >40-atom CUDA force call is rejected by the gate with 1-ulp differences.
  # Pin both translation units to -fmad=false so the two paths agree exactly.
  set_source_files_properties(
    src/backends/cuda/gfn2_pairlist.cu
    src/backends/cuda/gfn2_geometry.cu
    PROPERTIES COMPILE_OPTIONS "-fmad=false"
  )

  target_sources(xtbloom PRIVATE
    $<TARGET_OBJECTS:xtbloom_cuda_backend>
  )
  # The NVIDIA runtime/math/driver libraries are intentionally not linked or
  # exported transitively; startup pre-resolution fills hidden trampoline
  # tables and the installed library does not demand them at load time.
  target_compile_definitions(xtbloom PRIVATE XTBLOOM_HAS_CUDA=1)
  set_target_properties(xtbloom PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY None
  )
  # xtbloom uses relocatable device code for cross-translation-unit constants,
  # but it does not launch child kernels through CUDA Dynamic Parallelism.
  # Exclude nvcc's otherwise-default static device-runtime input explicitly.
  target_link_options(xtbloom PRIVATE
    "$<DEVICE_LINK:--cudadevrt=none>"
    "LINKER:-z,defs"
  )
endif()

if(XTBLOOM_BUILD_TESTS)
  enable_testing()
  find_package(Threads REQUIRED)
  find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)

  add_test(
    NAME xtbloom.benchmarks.self_test
    COMMAND ${Python3_EXECUTABLE} -m unittest -v
      benchmarks.test_run benchmarks.test_natoms_scaling
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  add_test(
    NAME xtbloom.cmake.cuda_toolkit_metadata
    COMMAND ${CMAKE_COMMAND} -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/cuda_toolkit_metadata_test.cmake
  )
  add_test(
    NAME xtbloom.cmake.version_resolution
    COMMAND ${CMAKE_COMMAND} -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/version_resolution_test.cmake
  )

  if(XTBLOOM_CPU_LINALG_RUNTIME)
    get_filename_component(XTBLOOM_CPU_LINALG_DIRECTORY
      "${XTBLOOM_CPU_LINALG_RUNTIME}" DIRECTORY
    )
    if(WIN32)
      set(XTBLOOM_RUNTIME_LIBRARY_PATH_VARIABLE PATH)
    elseif(APPLE)
      set(XTBLOOM_RUNTIME_LIBRARY_PATH_VARIABLE DYLD_LIBRARY_PATH)
    else()
      set(XTBLOOM_RUNTIME_LIBRARY_PATH_VARIABLE LD_LIBRARY_PATH)
    endif()

    function(xtbloom_configure_linalg_runtime_test target_name test_name)
      # The production eigensolver dlopens the BLAS/LAPACK runtime by path and
      # soname. Give CTest (including sanitizer builds) an explicit loader path
      # instead of relying only on the executable's build RPATH being inherited
      # by dlopen. The MKL env vars are harmless for OpenBLAS runtimes.
      set_target_properties(${target_name} PROPERTIES
        BUILD_RPATH "${XTBLOOM_CPU_LINALG_DIRECTORY}"
      )
      set_tests_properties(${test_name} PROPERTIES
        ENVIRONMENT "MKL_INTERFACE_LAYER=LP64;MKL_THREADING_LAYER=SEQUENTIAL"
        ENVIRONMENT_MODIFICATION
          "${XTBLOOM_RUNTIME_LIBRARY_PATH_VARIABLE}=path_list_prepend:${XTBLOOM_CPU_LINALG_DIRECTORY}"
      )
    endfunction()
  endif()

  add_executable(xtbloom_c_api_test tests/c_api_test.c)
  target_link_libraries(xtbloom_c_api_test PRIVATE xtbloom::xtbloom)
  set_target_properties(xtbloom_c_api_test PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED YES)
  add_test(NAME xtbloom.c_api COMMAND xtbloom_c_api_test)

  # The browser adapter test supplies a fake public C ABI so failure
  # publication and optimizer math remain testable without a WASM toolchain or
  # an external BLAS/LAPACK runtime.
  add_executable(xtbloom_web_adapter_test tests/web_adapter_test.c)
  target_include_directories(xtbloom_web_adapter_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  if(NOT WIN32)
    target_link_libraries(xtbloom_web_adapter_test PRIVATE m)
  endif()
  set_target_properties(xtbloom_web_adapter_test PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED YES
  )
  add_test(NAME xtbloom.web.adapter COMMAND xtbloom_web_adapter_test)

  add_executable(xtbloom_result_owner_test tests/result_owner_test.cpp)
  target_link_libraries(xtbloom_result_owner_test PRIVATE xtbloom::xtbloom)
  target_compile_features(xtbloom_result_owner_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.result_owner COMMAND xtbloom_result_owner_test)

  add_executable(xtbloom_runtime_test tests/runtime_test.cpp)
  target_link_libraries(xtbloom_runtime_test PRIVATE xtbloom::xtbloom)
  target_compile_features(xtbloom_runtime_test PRIVATE cxx_std_17)
  if(XTBLOOM_HAS_CUDA)
    target_compile_definitions(xtbloom_runtime_test PRIVATE XTBLOOM_TEST_HAS_CUDA=1)
  endif()
  add_test(NAME xtbloom.runtime COMMAND xtbloom_runtime_test)
  if(XTBLOOM_CPU_LINALG_RUNTIME)
    xtbloom_configure_linalg_runtime_test(xtbloom_runtime_test xtbloom.runtime)
  endif()

  if(XTBLOOM_HAS_CUDA)
    # This host-only test must compile in the provider-free configuration. It
    # verifies that provider status returns remain fixed-width integer domains
    # even when compilers are asked to shorten ordinary enum types.
    add_executable(xtbloom_nvidia_host_api_test tests/nvidia_host_api_test.cpp)
    target_include_directories(xtbloom_nvidia_host_api_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${XTBLOOM_CUDA_TOOLKIT_INCLUDE_DIRS}
    )
    target_compile_features(xtbloom_nvidia_host_api_test PRIVATE cxx_std_17)
    target_compile_options(xtbloom_nvidia_host_api_test PRIVATE
      $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-fshort-enums>
    )
    add_test(NAME xtbloom.cuda.nvidia_host_api COMMAND xtbloom_nvidia_host_api_test)
  endif()

  if(XTBLOOM_CPU_LINALG_RUNTIME)
    add_executable(xtbloom_cpu_public_inference_test
      tests/cpu_public_inference_test.cpp
    )
    target_link_libraries(xtbloom_cpu_public_inference_test PRIVATE
      xtbloom::xtbloom
      Threads::Threads
    )
    target_compile_features(xtbloom_cpu_public_inference_test PRIVATE cxx_std_17)
    add_test(NAME xtbloom.cpu.public_inference COMMAND xtbloom_cpu_public_inference_test)
    xtbloom_configure_linalg_runtime_test(
      xtbloom_cpu_public_inference_test xtbloom.cpu.public_inference
    )
  endif()

  if(BUILD_SHARED_LIBS AND UNIX AND NOT APPLE)
    add_test(
      NAME xtbloom.abi_symbols
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/abi/check_symbols.py
              --nm ${CMAKE_NM} --library $<TARGET_FILE:xtbloom>
    )
    if(XTBLOOM_HAS_CUDA)
      find_program(XTBLOOM_READELF NAMES readelf REQUIRED)
      add_test(
        NAME xtbloom.abi_cuda_no_proprietary_needed
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/abi/check_cuda_dependencies.py
                --readelf ${XTBLOOM_READELF} --library $<TARGET_FILE:xtbloom>
      )
      add_test(
        NAME xtbloom.abi_cuda_dependency_checker
        COMMAND ${Python3_EXECUTABLE} -m unittest discover
                -s ${CMAKE_CURRENT_SOURCE_DIR}/tests/abi
                -p test_check_cuda_dependencies.py -v
      )
      add_test(
        NAME xtbloom.abi_cuda_direct_dso_load
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/abi/check_cuda_dlopen_runtime.py
                --library $<TARGET_FILE:xtbloom> --mode load
      )
      add_test(
        NAME xtbloom.abi_cuda_no_runtime_concurrent
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/abi/check_cuda_dlopen_runtime.py
                --library $<TARGET_FILE:xtbloom>
                --mode concurrent-unavailable --threads 16 --calls-per-thread 16
      )
      set_tests_properties(xtbloom.abi_cuda_no_runtime_concurrent PROPERTIES
        ENVIRONMENT "XTBLOOM_CUDA_FORCE_FALLBACK=1"
      )
    endif()
  endif()

  add_executable(xtbloom_batch_validation_test
    tests/batch_validation_test.cpp
    src/runtime/validation.cpp
  )
  target_include_directories(xtbloom_batch_validation_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_compile_features(xtbloom_batch_validation_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.batch_validation COMMAND xtbloom_batch_validation_test)

  add_executable(xtbloom_plan_schema_test tests/plan_schema_test.cpp)
  target_link_libraries(xtbloom_plan_schema_test PRIVATE xtbloom_gfn2_cpu)
  target_compile_features(xtbloom_plan_schema_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.gfn2.plan_schema COMMAND xtbloom_plan_schema_test)

  # CUDA runtime tests link CUDA::cudart and run on a real GPU. They are
  # built only when the build environment actually provides the provider
  # libraries; a stub-generation build (nvcc support files plus cudart
  # headers, but no provider shared objects) still
  # compiles and publishes the CUDA backend but cannot exercise it here.
  if(XTBLOOM_HAS_CUDA AND _xtbloom_cuda_providers_present)
    add_executable(xtbloom_cuda_descriptor_validation_test
      tests/cuda_descriptor_validation_test.cu
    )
    target_link_libraries(xtbloom_cuda_descriptor_validation_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
      CUDA::cudart
    )
    target_include_directories(xtbloom_cuda_descriptor_validation_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_descriptor_validation_test PRIVATE
      XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_descriptor_validation_test PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
    )
    add_test(NAME xtbloom.cuda.descriptor_validation
      COMMAND xtbloom_cuda_descriptor_validation_test
    )

    add_executable(xtbloom_cuda_gfn2_topology_staging_test
      tests/cuda_gfn2_topology_staging_test.cu
    )
    target_link_libraries(xtbloom_cuda_gfn2_topology_staging_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
      CUDA::cudart
    )
    target_include_directories(xtbloom_cuda_gfn2_topology_staging_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_gfn2_topology_staging_test PRIVATE
      XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_gfn2_topology_staging_test PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
    )
    add_test(NAME xtbloom.cuda.gfn2_topology_staging
      COMMAND xtbloom_cuda_gfn2_topology_staging_test
    )

    add_executable(xtbloom_cuda_gfn2_public_result_bridge_test
      tests/cuda_gfn2_public_result_bridge_test.cu
    )
    target_link_libraries(xtbloom_cuda_gfn2_public_result_bridge_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
      CUDA::cudart
    )
    target_include_directories(xtbloom_cuda_gfn2_public_result_bridge_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_gfn2_public_result_bridge_test PRIVATE
      XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_gfn2_public_result_bridge_test PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
    )
    add_test(NAME xtbloom.cuda.gfn2_public_result_bridge
      COMMAND xtbloom_cuda_gfn2_public_result_bridge_test
    )

    add_executable(xtbloom_cuda_public_api_test
      tests/cuda_public_api_test.cu
      tests/support/gfn2_scc_test_case.cpp
    )
    target_link_libraries(xtbloom_cuda_public_api_test PRIVATE
      xtbloom::xtbloom
      xtbloom_gfn2_cpu
      CUDA::cudart
      Threads::Threads
    )
    target_include_directories(xtbloom_cuda_public_api_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_public_api_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_public_api_test PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
    )
    add_test(NAME xtbloom.cuda.public_api COMMAND xtbloom_cuda_public_api_test)
    if(XTBLOOM_CPU_LINALG_RUNTIME)
      xtbloom_configure_linalg_runtime_test(
        xtbloom_cuda_public_api_test xtbloom.cuda.public_api
      )
    endif()

    add_executable(xtbloom_cuda_public_unrestricted_test
      tests/cuda_public_unrestricted_test.cu
    )
    target_link_libraries(xtbloom_cuda_public_unrestricted_test PRIVATE
      xtbloom::xtbloom
      CUDA::cudart
    )
    target_include_directories(xtbloom_cuda_public_unrestricted_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_public_unrestricted_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_public_unrestricted_test PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      # Shared-library builds must load the cudart paired with the selected
      # CUDA compiler, even when another toolkit is the system default.
      BUILD_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>"
    )
    add_test(NAME xtbloom.cuda.public_unrestricted
      COMMAND xtbloom_cuda_public_unrestricted_test
    )

    add_executable(xtbloom_cuda_result_owner_test
      tests/cuda_result_owner_test.cu
    )
    target_link_libraries(xtbloom_cuda_result_owner_test PRIVATE
      xtbloom::xtbloom
      CUDA::cudart
    )
    target_include_directories(xtbloom_cuda_result_owner_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_result_owner_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_result_owner_test PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      BUILD_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>"
    )
    add_test(NAME xtbloom.cuda.result_owner
      COMMAND xtbloom_cuda_result_owner_test
    )

    add_executable(xtbloom_cuda_atomics_test tests/cuda_atomics_test.cu)
    target_link_libraries(xtbloom_cuda_atomics_test PRIVATE CUDA::cudart)
    target_include_directories(xtbloom_cuda_atomics_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_atomics_test PRIVATE
      XTBLOOM_FORCE_FP64_ATOMIC_CAS=1
    )
    set_target_properties(xtbloom_cuda_atomics_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
    )
    add_test(NAME xtbloom.cuda.atomics COMMAND xtbloom_cuda_atomics_test)

    add_executable(xtbloom_cuda_parameters_test tests/cuda_parameters_test.cu)
    target_link_libraries(xtbloom_cuda_parameters_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_parameters_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_parameters_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_parameters_test PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
    )
    add_test(NAME xtbloom.cuda.parameters COMMAND xtbloom_cuda_parameters_test)

    add_executable(xtbloom_cuda_repulsion_test tests/cuda_repulsion_test.cu)
    target_link_libraries(xtbloom_cuda_repulsion_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_repulsion_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_repulsion_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_repulsion_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.repulsion COMMAND xtbloom_cuda_repulsion_test)

    add_executable(xtbloom_cuda_external_point_charges_test
      tests/cuda_external_point_charges_test.cu
    )
    target_link_libraries(xtbloom_cuda_external_point_charges_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_external_point_charges_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_external_point_charges_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_external_point_charges_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.external_point_charges
      COMMAND xtbloom_cuda_external_point_charges_test
    )

    add_executable(xtbloom_cuda_classical_force_test
      tests/cuda_classical_force_test.cu
    )
    target_link_libraries(xtbloom_cuda_classical_force_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_classical_force_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_classical_force_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_classical_force_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.classical_force
      COMMAND xtbloom_cuda_classical_force_test
    )

    add_executable(xtbloom_cuda_force_composition_test
      tests/cuda_force_composition_test.cu
    )
    target_link_libraries(xtbloom_cuda_force_composition_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_force_composition_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_force_composition_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_force_composition_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.force_composition
      COMMAND xtbloom_cuda_force_composition_test
    )

    add_executable(xtbloom_cuda_geometry_test tests/cuda_geometry_test.cu)
    target_link_libraries(xtbloom_cuda_geometry_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_geometry_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_geometry_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_geometry_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.geometry COMMAND xtbloom_cuda_geometry_test)

    add_executable(xtbloom_cuda_pairlist_test tests/cuda_pairlist_test.cu)
    target_link_libraries(xtbloom_cuda_pairlist_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_pairlist_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_pairlist_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_pairlist_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.pairlist COMMAND xtbloom_cuda_pairlist_test)

    # Focused #70 benchmark.  It measures the sparse bucket build, the dense
    # fallback build, and the coordination reuse cost at batch 1/8/32/128 so
    # the dispatch crossover is anchored by reproducible profiling.
    add_executable(xtbloom_cuda_pairlist_benchmark tests/cuda_pairlist_test.cu)
    target_link_libraries(xtbloom_cuda_pairlist_benchmark PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_pairlist_benchmark PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_pairlist_benchmark PRIVATE
      XTBLOOM_HAS_CUDA=1
      XTBLOOM_PAIRLIST_BENCHMARK_ONLY=1
    )
    set_target_properties(xtbloom_cuda_pairlist_benchmark PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(xtbloom_cuda_integrals_test tests/cuda_integrals_test.cu)
    target_link_libraries(xtbloom_cuda_integrals_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_integrals_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_integrals_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_integrals_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.integrals COMMAND xtbloom_cuda_integrals_test)

    add_executable(xtbloom_cuda_integral_force_test tests/cuda_integral_force_test.cu)
    target_link_libraries(xtbloom_cuda_integral_force_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_integral_force_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_integral_force_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_integral_force_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.integral_force COMMAND xtbloom_cuda_integral_force_test)

    add_executable(xtbloom_cuda_h0_force_test tests/cuda_h0_force_test.cu)
    target_link_libraries(xtbloom_cuda_h0_force_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_h0_force_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_h0_force_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_h0_force_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.h0_force COMMAND xtbloom_cuda_h0_force_test)

    add_executable(xtbloom_cuda_hamiltonian_force_test tests/cuda_hamiltonian_force_test.cu)
    target_link_libraries(xtbloom_cuda_hamiltonian_force_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_hamiltonian_force_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_hamiltonian_force_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_hamiltonian_force_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.hamiltonian_force COMMAND xtbloom_cuda_hamiltonian_force_test)

    add_executable(xtbloom_cuda_hamiltonian_test tests/cuda_hamiltonian_test.cu)
    target_link_libraries(xtbloom_cuda_hamiltonian_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_hamiltonian_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_hamiltonian_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_hamiltonian_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.hamiltonian COMMAND xtbloom_cuda_hamiltonian_test)

    add_executable(xtbloom_cuda_mulliken_test tests/cuda_mulliken_test.cu)
    target_link_libraries(xtbloom_cuda_mulliken_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_mulliken_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_mulliken_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_mulliken_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.mulliken COMMAND xtbloom_cuda_mulliken_test)

    add_executable(xtbloom_cuda_spin_test tests/cuda_spin_test.cu)
    target_link_libraries(xtbloom_cuda_spin_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_spin_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_spin_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_spin_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.spin COMMAND xtbloom_cuda_spin_test)

    add_executable(xtbloom_cuda_occupations_test tests/cuda_occupations_test.cu)
    target_link_libraries(xtbloom_cuda_occupations_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_occupations_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_occupations_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_occupations_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.occupations COMMAND xtbloom_cuda_occupations_test)

    add_executable(xtbloom_cuda_density_test tests/cuda_density_test.cu)
    target_link_libraries(xtbloom_cuda_density_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_density_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_density_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_density_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.density COMMAND xtbloom_cuda_density_test)

    add_executable(xtbloom_cuda_electronic_gradient_test
      tests/cuda_electronic_gradient_test.cu
    )
    target_link_libraries(xtbloom_cuda_electronic_gradient_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_electronic_gradient_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_electronic_gradient_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_electronic_gradient_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.electronic_gradient COMMAND xtbloom_cuda_electronic_gradient_test)

    add_executable(xtbloom_cuda_energy_force_execution_test
      tests/cuda_energy_force_execution_test.cu
    )
    target_link_libraries(xtbloom_cuda_energy_force_execution_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_energy_force_execution_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_energy_force_execution_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_energy_force_execution_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.energy_force_execution
      COMMAND xtbloom_cuda_energy_force_execution_test
    )

    add_executable(xtbloom_cuda_gfn2_terminal_execution_test
      tests/cuda_gfn2_terminal_execution_test.cu
    )
    target_link_libraries(xtbloom_cuda_gfn2_terminal_execution_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_gfn2_terminal_execution_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_gfn2_terminal_execution_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_gfn2_terminal_execution_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.gfn2_terminal_execution
      COMMAND xtbloom_cuda_gfn2_terminal_execution_test
    )

    add_executable(xtbloom_cuda_gfn2_runtime_owner_test
      tests/cuda_gfn2_runtime_owner_test.cu
      tests/support/gfn2_scc_test_case.cpp
      # This white-box test exercises the internal context owner directly.
      # Shared-library builds hide that implementation symbol by design, so
      # compile the owner into the test instead of weakening the public ABI.
      src/runtime/backend.cpp
      src/runtime/gfn2_cpu_execution.cpp
    )
    target_link_libraries(xtbloom_cuda_gfn2_runtime_owner_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_gfn2_runtime_owner_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_gfn2_runtime_owner_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_gfn2_runtime_owner_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.gfn2_runtime_owner
      COMMAND xtbloom_cuda_gfn2_runtime_owner_test
    )

    add_executable(xtbloom_cuda_gfn2_runtime_parity_test
      tests/cuda_gfn2_runtime_parity_test.cu
      tests/support/gfn2_scc_test_case.cpp
    )
    target_link_libraries(xtbloom_cuda_gfn2_runtime_parity_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_gfn2_runtime_parity_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_gfn2_runtime_parity_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_gfn2_runtime_parity_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.gfn2_runtime_parity
      COMMAND xtbloom_cuda_gfn2_runtime_parity_test
    )
    if(XTBLOOM_CPU_LINALG_RUNTIME)
      xtbloom_configure_linalg_runtime_test(
        xtbloom_cuda_gfn2_runtime_parity_test xtbloom.cuda.gfn2_runtime_parity
      )
    endif()

    add_executable(xtbloom_cuda_gfn2_runtime_graph_test
      tests/cuda_gfn2_runtime_graph_test.cu
      tests/support/gfn2_scc_test_case.cpp
    )
    target_link_libraries(xtbloom_cuda_gfn2_runtime_graph_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_gfn2_runtime_graph_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_gfn2_runtime_graph_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_gfn2_runtime_graph_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.gfn2_runtime_graph
      COMMAND xtbloom_cuda_gfn2_runtime_graph_test
    )

    add_executable(xtbloom_cuda_gfn2_runtime_cache_diagnostics_test
      tests/cuda_gfn2_runtime_cache_diagnostics_test.cu
      tests/support/gfn2_scc_test_case.cpp
    )
    target_link_libraries(xtbloom_cuda_gfn2_runtime_cache_diagnostics_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_gfn2_runtime_cache_diagnostics_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_gfn2_runtime_cache_diagnostics_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_gfn2_runtime_cache_diagnostics_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.gfn2_runtime_cache_diagnostics
      COMMAND xtbloom_cuda_gfn2_runtime_cache_diagnostics_test
    )

    add_executable(xtbloom_cuda_gfn2_preprocessing_test
      tests/cuda_gfn2_preprocessing_test.cu
    )
    target_link_libraries(xtbloom_cuda_gfn2_preprocessing_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_gfn2_preprocessing_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_gfn2_preprocessing_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_gfn2_preprocessing_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.gfn2_preprocessing
      COMMAND xtbloom_cuda_gfn2_preprocessing_test
    )

    add_executable(xtbloom_cuda_eigensolver_test tests/cuda_eigensolver_test.cu)
    target_link_libraries(xtbloom_cuda_eigensolver_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_eigensolver_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_eigensolver_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_eigensolver_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.eigensolver COMMAND xtbloom_cuda_eigensolver_test)

    # Focused #80/#131 compaction benchmark. Builds homogeneous and
    # heterogeneous restricted eigensolver buckets, drives canonical
    # active-count tier sweeps, and reports exact-capacity bucket activity
    # (active vs submitted vs completed) with compacted/uncompacted latency
    # crossover for batch sizes 1/8/32/128. Not a CTest test: it is a
    # measurement runner like xtbloom_cuda_scc_loop_benchmark.
    add_executable(xtbloom_cuda_compaction_benchmark tests/cuda_eigensolver_test.cu)
    target_link_libraries(xtbloom_cuda_compaction_benchmark PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_compaction_benchmark PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_compaction_benchmark PRIVATE
      XTBLOOM_HAS_CUDA=1
      XTBLOOM_COMPACTION_BENCHMARK_ONLY=1
    )
    set_target_properties(xtbloom_cuda_compaction_benchmark PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(xtbloom_cuda_es3_test tests/cuda_es3_test.cu)
    target_link_libraries(xtbloom_cuda_es3_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_es3_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_es3_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_es3_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.es3 COMMAND xtbloom_cuda_es3_test)

    add_executable(xtbloom_cuda_es2_test tests/cuda_es2_test.cu)
    target_link_libraries(xtbloom_cuda_es2_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_es2_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_es2_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_es2_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.es2 COMMAND xtbloom_cuda_es2_test)

    add_executable(xtbloom_cuda_aes2_test tests/cuda_aes2_test.cu)
    target_link_libraries(xtbloom_cuda_aes2_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_aes2_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_aes2_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_aes2_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.aes2 COMMAND xtbloom_cuda_aes2_test)

    add_executable(xtbloom_cuda_d4_test tests/cuda_d4_test.cu)
    target_link_libraries(xtbloom_cuda_d4_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_d4_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_d4_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_d4_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.d4 COMMAND xtbloom_cuda_d4_test)

    add_executable(xtbloom_cuda_periodic_embedding_test
      tests/cuda_periodic_embedding_test.cu
    )
    target_link_libraries(xtbloom_cuda_periodic_embedding_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_periodic_embedding_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_periodic_embedding_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_periodic_embedding_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.periodic_embedding
      COMMAND xtbloom_cuda_periodic_embedding_test
    )

    add_executable(xtbloom_cuda_scc_test tests/cuda_scc_test.cu)
    target_link_libraries(xtbloom_cuda_scc_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_scc_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc COMMAND xtbloom_cuda_scc_test)

    add_executable(xtbloom_cuda_plan_schema_test tests/cuda_plan_schema_test.cu)
    target_link_libraries(xtbloom_cuda_plan_schema_test PRIVATE
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_plan_schema_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_plan_schema_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_plan_schema_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.plan_schema COMMAND xtbloom_cuda_plan_schema_test)

    add_executable(xtbloom_cuda_scc_bridge_test tests/cuda_scc_bridge_test.cu)
    target_link_libraries(xtbloom_cuda_scc_bridge_test PRIVATE
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_bridge_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_bridge_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_scc_bridge_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc_bridge COMMAND xtbloom_cuda_scc_bridge_test)

    add_executable(xtbloom_cuda_scc_iteration_control_test
      tests/cuda_scc_iteration_control_test.cu
    )
    target_link_libraries(xtbloom_cuda_scc_iteration_control_test PRIVATE
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_iteration_control_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_iteration_control_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_scc_iteration_control_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc_iteration_control
      COMMAND xtbloom_cuda_scc_iteration_control_test
    )

    add_executable(xtbloom_cuda_scc_iteration_binding_test
      tests/cuda_scc_iteration_binding_test.cu
    )
    target_link_libraries(xtbloom_cuda_scc_iteration_binding_test PRIVATE
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_iteration_binding_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_iteration_binding_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_scc_iteration_binding_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc_iteration_binding
      COMMAND xtbloom_cuda_scc_iteration_binding_test
    )

    add_executable(xtbloom_cuda_scc_publication_test
      tests/cuda_scc_publication_test.cu
    )
    target_link_libraries(xtbloom_cuda_scc_publication_test PRIVATE
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_publication_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_publication_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_scc_publication_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc_publication
      COMMAND xtbloom_cuda_scc_publication_test
    )

    add_executable(xtbloom_cuda_scc_iteration_composer_test
      tests/cuda_scc_iteration_composer_test.cu
      tests/support/gfn2_scc_test_case.cpp
    )
    target_link_libraries(xtbloom_cuda_scc_iteration_composer_test PRIVATE
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_iteration_composer_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_iteration_composer_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_scc_iteration_composer_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc_iteration_composer
      COMMAND xtbloom_cuda_scc_iteration_composer_test
    )

    add_executable(xtbloom_cuda_scc_iteration_arena_test
      tests/cuda_scc_iteration_arena_test.cu
    )
    target_link_libraries(xtbloom_cuda_scc_iteration_arena_test PRIVATE
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_iteration_arena_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_iteration_arena_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_scc_iteration_arena_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc_iteration_arena
      COMMAND xtbloom_cuda_scc_iteration_arena_test
    )

    add_executable(xtbloom_cuda_scc_iteration_initialize_test
      tests/cuda_scc_iteration_initialize_test.cu
    )
    target_link_libraries(xtbloom_cuda_scc_iteration_initialize_test PRIVATE
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_iteration_initialize_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_iteration_initialize_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_scc_iteration_initialize_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc_iteration_initialize
      COMMAND xtbloom_cuda_scc_iteration_initialize_test
    )

    add_executable(xtbloom_cuda_scc_iteration_reports_test
      tests/cuda_scc_iteration_reports_test.cu
    )
    target_link_libraries(xtbloom_cuda_scc_iteration_reports_test PRIVATE
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_iteration_reports_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_iteration_reports_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_scc_iteration_reports_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc_iteration_reports
      COMMAND xtbloom_cuda_scc_iteration_reports_test
    )

    add_executable(xtbloom_cuda_scc_setup_topology_test
      tests/cuda_scc_setup_topology_test.cu
    )
    target_link_libraries(xtbloom_cuda_scc_setup_topology_test PRIVATE
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_setup_topology_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_setup_topology_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_scc_setup_topology_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc_setup_topology
      COMMAND xtbloom_cuda_scc_setup_topology_test
    )

    add_executable(xtbloom_cuda_scc_setup_eigensolver_test
      tests/cuda_scc_setup_eigensolver_test.cu
      tests/support/gfn2_scc_test_case.cpp
    )
    target_link_libraries(xtbloom_cuda_scc_setup_eigensolver_test PRIVATE
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_setup_eigensolver_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_setup_eigensolver_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_scc_setup_eigensolver_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc_setup_eigensolver
      COMMAND xtbloom_cuda_scc_setup_eigensolver_test
    )

    add_executable(xtbloom_cuda_scc_setup_inputs_test
      tests/cuda_scc_setup_inputs_test.cu
      tests/support/gfn2_scc_test_case.cpp
    )
    target_link_libraries(xtbloom_cuda_scc_setup_inputs_test PRIVATE
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_setup_inputs_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_setup_inputs_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_scc_setup_inputs_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc_setup_inputs
      COMMAND xtbloom_cuda_scc_setup_inputs_test
    )

    add_executable(xtbloom_cuda_scc_iteration_production_test
      tests/cuda_scc_iteration_production_test.cu
      tests/support/gfn2_scc_test_case.cpp
    )
    target_link_libraries(xtbloom_cuda_scc_iteration_production_test PRIVATE
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_iteration_production_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_iteration_production_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_scc_iteration_production_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc_iteration_production
      COMMAND xtbloom_cuda_scc_iteration_production_test
    )
    add_test(NAME xtbloom.cuda.scc_iteration_large_singleton
      COMMAND xtbloom_cuda_scc_iteration_production_test --large-singleton-tridiagonal
    )

    # Focused #130 benchmark. It reuses the production SCC fixture but runs a
    # one-remaining-iteration state so the conditional WHILE body count and
    # latency can be compared directly with the bounded fallback.
    add_executable(xtbloom_cuda_scc_loop_benchmark
      tests/cuda_scc_iteration_production_test.cu
      tests/support/gfn2_scc_test_case.cpp
    )
    target_link_libraries(xtbloom_cuda_scc_loop_benchmark PRIVATE
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_loop_benchmark PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_loop_benchmark PRIVATE
      XTBLOOM_HAS_CUDA=1
      XTBLOOM_SCC_LOOP_BENCHMARK_ONLY=1
    )
    set_target_properties(xtbloom_cuda_scc_loop_benchmark PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(xtbloom_cuda_scc_iteration_failure_matrix_test
      tests/cuda_scc_iteration_failure_matrix_test.cu
      tests/support/gfn2_scc_test_case.cpp
    )
    target_link_libraries(xtbloom_cuda_scc_iteration_failure_matrix_test PRIVATE
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_iteration_failure_matrix_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_iteration_failure_matrix_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_scc_iteration_failure_matrix_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc_iteration_failure_matrix
      COMMAND xtbloom_cuda_scc_iteration_failure_matrix_test
    )

    add_executable(xtbloom_cuda_scc_electrostatic_components_test
      tests/cuda_scc_electrostatic_components_test.cu
    )
    target_link_libraries(xtbloom_cuda_scc_electrostatic_components_test PRIVATE
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_electrostatic_components_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_electrostatic_components_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_scc_electrostatic_components_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc_electrostatic_components
      COMMAND xtbloom_cuda_scc_electrostatic_components_test
    )

    add_executable(xtbloom_cuda_scc_d4_periodic_test
      tests/cuda_scc_d4_periodic_test.cu
    )
    target_link_libraries(xtbloom_cuda_scc_d4_periodic_test PRIVATE
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_d4_periodic_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_d4_periodic_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_scc_d4_periodic_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc_d4_periodic
      COMMAND xtbloom_cuda_scc_d4_periodic_test
    )

    add_executable(xtbloom_cuda_scc_free_energy_test
      tests/cuda_scc_free_energy_test.cu
    )
    target_link_libraries(xtbloom_cuda_scc_free_energy_test PRIVATE
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_free_energy_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_free_energy_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_scc_free_energy_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc_free_energy COMMAND xtbloom_cuda_scc_free_energy_test)

    add_executable(xtbloom_cuda_total_energy_test tests/cuda_total_energy_test.cu)
    target_link_libraries(xtbloom_cuda_total_energy_test PRIVATE
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_total_energy_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_total_energy_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_total_energy_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.total_energy COMMAND xtbloom_cuda_total_energy_test)

    add_executable(xtbloom_cuda_post_scc_potential_test
      tests/cuda_post_scc_potential_test.cu
    )
    target_link_libraries(xtbloom_cuda_post_scc_potential_test PRIVATE
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_post_scc_potential_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_post_scc_potential_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_post_scc_potential_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.post_scc_potential
      COMMAND xtbloom_cuda_post_scc_potential_test
    )

    add_executable(xtbloom_cuda_scc_energy_test tests/cuda_scc_energy_test.cu)
    target_link_libraries(xtbloom_cuda_scc_energy_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_energy_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_energy_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_scc_energy_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc_energy COMMAND xtbloom_cuda_scc_energy_test)

    add_executable(xtbloom_cuda_scc_classical_energy_test
      tests/cuda_scc_classical_energy_test.cu
    )
    target_link_libraries(xtbloom_cuda_scc_classical_energy_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_classical_energy_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_classical_energy_test
      PRIVATE XTBLOOM_HAS_CUDA=1
    )
    set_target_properties(xtbloom_cuda_scc_classical_energy_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc_classical_energy
      COMMAND xtbloom_cuda_scc_classical_energy_test
    )

    add_executable(xtbloom_cuda_scc_potential_test tests/cuda_scc_potential_test.cu)
    target_link_libraries(xtbloom_cuda_scc_potential_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_potential_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_potential_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_scc_potential_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc_potential COMMAND xtbloom_cuda_scc_potential_test)

    add_executable(xtbloom_cuda_scc_mixer_test tests/cuda_scc_mixer_test.cu)
    target_link_libraries(xtbloom_cuda_scc_mixer_test PRIVATE
      xtbloom::xtbloom
      xtbloom_cuda_backend
      xtbloom_gfn2_cpu
    )
    target_include_directories(xtbloom_cuda_scc_mixer_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(xtbloom_cuda_scc_mixer_test PRIVATE XTBLOOM_HAS_CUDA=1)
    set_target_properties(xtbloom_cuda_scc_mixer_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME xtbloom.cuda.scc_mixer COMMAND xtbloom_cuda_scc_mixer_test)
  endif()

  add_executable(xtbloom_repulsion_test tests/repulsion_test.cpp)
  target_link_libraries(xtbloom_repulsion_test PRIVATE xtbloom_gfn2_cpu)
  target_compile_features(xtbloom_repulsion_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.gfn2.repulsion COMMAND xtbloom_repulsion_test)

  add_executable(xtbloom_coordination_test tests/coordination_test.cpp)
  target_link_libraries(xtbloom_coordination_test PRIVATE xtbloom_gfn2_cpu)
  target_compile_features(xtbloom_coordination_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.gfn2.coordination COMMAND xtbloom_coordination_test)

  add_executable(xtbloom_d4_test tests/d4_test.cpp)
  target_link_libraries(xtbloom_d4_test PRIVATE xtbloom_gfn2_cpu)
  target_compile_features(xtbloom_d4_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.gfn2.d4 COMMAND xtbloom_d4_test)

  add_executable(xtbloom_basis_test tests/basis_test.cpp)
  target_link_libraries(xtbloom_basis_test PRIVATE xtbloom_gfn2_cpu)
  target_compile_features(xtbloom_basis_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.gfn2.basis COMMAND xtbloom_basis_test)

  add_executable(xtbloom_integrals_test tests/integrals_test.cpp)
  target_link_libraries(xtbloom_integrals_test PRIVATE xtbloom_gfn2_cpu)
  target_compile_features(xtbloom_integrals_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.gfn2.integrals COMMAND xtbloom_integrals_test)

  add_executable(xtbloom_multipole_integrals_test tests/multipole_integrals_test.cpp)
  target_link_libraries(xtbloom_multipole_integrals_test PRIVATE xtbloom_gfn2_cpu)
  target_compile_features(xtbloom_multipole_integrals_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.gfn2.multipole_integrals COMMAND xtbloom_multipole_integrals_test)

  add_executable(xtbloom_h0_test tests/h0_test.cpp)
  target_link_libraries(xtbloom_h0_test PRIVATE xtbloom_gfn2_cpu)
  target_compile_features(xtbloom_h0_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.gfn2.h0 COMMAND xtbloom_h0_test)

  add_executable(xtbloom_force_test tests/force_test.cpp)
  target_link_libraries(xtbloom_force_test PRIVATE xtbloom_gfn2_cpu)
  target_compile_features(xtbloom_force_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.gfn2.force COMMAND xtbloom_force_test)

  add_executable(xtbloom_external_point_charges_test tests/external_point_charges_test.cpp)
  target_link_libraries(xtbloom_external_point_charges_test PRIVATE xtbloom_gfn2_cpu)
  target_compile_features(xtbloom_external_point_charges_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.gfn2.external_point_charges COMMAND xtbloom_external_point_charges_test)

  add_executable(xtbloom_periodic_embedding_test tests/periodic_embedding_test.cpp)
  target_link_libraries(xtbloom_periodic_embedding_test PRIVATE xtbloom_gfn2_cpu Threads::Threads)
  target_compile_features(xtbloom_periodic_embedding_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.gfn2.periodic_embedding COMMAND xtbloom_periodic_embedding_test)

  add_executable(xtbloom_es3_test tests/es3_test.cpp)
  target_link_libraries(xtbloom_es3_test PRIVATE xtbloom_gfn2_cpu)
  target_compile_features(xtbloom_es3_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.gfn2.es3 COMMAND xtbloom_es3_test)

  add_executable(xtbloom_spin_test tests/spin_test.cpp)
  target_link_libraries(xtbloom_spin_test PRIVATE xtbloom_gfn2_cpu)
  target_compile_features(xtbloom_spin_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.gfn2.spin COMMAND xtbloom_spin_test)

  add_executable(xtbloom_es2_test tests/es2_test.cpp)
  target_link_libraries(xtbloom_es2_test PRIVATE xtbloom_gfn2_cpu)
  target_compile_features(xtbloom_es2_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.gfn2.es2 COMMAND xtbloom_es2_test)

  add_executable(xtbloom_aes2_test tests/aes2_test.cpp)
  target_link_libraries(xtbloom_aes2_test PRIVATE xtbloom_gfn2_cpu)
  target_compile_features(xtbloom_aes2_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.gfn2.aes2 COMMAND xtbloom_aes2_test)

  add_executable(xtbloom_wavefunction_test tests/wavefunction_test.cpp)
  target_link_libraries(xtbloom_wavefunction_test PRIVATE xtbloom_gfn2_cpu)
  target_compile_features(xtbloom_wavefunction_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.gfn2.wavefunction COMMAND xtbloom_wavefunction_test)

  add_executable(xtbloom_mulliken_test tests/mulliken_test.cpp)
  target_link_libraries(xtbloom_mulliken_test PRIVATE xtbloom_gfn2_cpu)
  target_compile_features(xtbloom_mulliken_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.gfn2.mulliken COMMAND xtbloom_mulliken_test)

  add_executable(xtbloom_scc_mixer_test tests/scc_mixer_test.cpp)
  target_link_libraries(xtbloom_scc_mixer_test PRIVATE xtbloom_gfn2_cpu Threads::Threads)
  target_compile_features(xtbloom_scc_mixer_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.gfn2.scc_mixer COMMAND xtbloom_scc_mixer_test)

  add_executable(xtbloom_scc_driver_test tests/scc_driver_test.cpp)
  target_link_libraries(xtbloom_scc_driver_test PRIVATE xtbloom_gfn2_cpu)
  target_compile_features(xtbloom_scc_driver_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.gfn2.scc_driver COMMAND xtbloom_scc_driver_test)

  # Focused #41 benchmark: per-iteration mixer history transactions must scale
  # with active-system history rather than the total batch history. Not a CTest
  # gate; run standalone to produce issue-scoped evidence.
  add_executable(xtbloom_scc_mixer_transaction_benchmark benchmarks/scc_mixer_transaction.cpp)
  target_link_libraries(xtbloom_scc_mixer_transaction_benchmark PRIVATE xtbloom_gfn2_cpu)
  target_compile_features(xtbloom_scc_mixer_transaction_benchmark PRIVATE cxx_std_17)

  if(XTBLOOM_CPU_LINALG_RUNTIME)
    add_executable(xtbloom_eigensolver_test tests/eigensolver_test.cpp)
    target_link_libraries(xtbloom_eigensolver_test PRIVATE
      xtbloom_gfn2_cpu
      ${XTBLOOM_CPU_LINALG_RUNTIME}
      Threads::Threads
    )
    target_compile_features(xtbloom_eigensolver_test PRIVATE cxx_std_17)
    if(XTBLOOM_CPU_LINALG_SYMBOL_PREFIX STREQUAL "scipy_")
      # The test spies call through to the configured provider directly; use
      # scipy-openblas32's collision-avoiding ABI names when applicable.
      target_compile_definitions(
        xtbloom_eigensolver_test PRIVATE XTBLOOM_TEST_SCIPY_PREFIXED_BLAS=1
      )
    endif()
    add_test(NAME xtbloom.gfn2.eigensolver COMMAND xtbloom_eigensolver_test)
    xtbloom_configure_linalg_runtime_test(
      xtbloom_eigensolver_test xtbloom.gfn2.eigensolver
    )

    if(XTBLOOM_CPU_LINALG_SHIM)
      # Host-isolated LP64 provider evidence (issue #30). This binary links only
      # xtbloom_gfn2_cpu, never a BLAS/MKL runtime, so it can prove that provider
      # libraries stay outside RTLD_DEFAULT and that xtbloom coexists with a
      # globally loaded ILP64 host before and after backend creation.
      add_executable(xtbloom_cpu_linalg_isolation_test
        tests/cpu_linalg_isolation_test.cpp
      )
      target_link_libraries(xtbloom_cpu_linalg_isolation_test PRIVATE
        xtbloom_gfn2_cpu
        ${CMAKE_DL_LIBS}
      )
      target_compile_features(xtbloom_cpu_linalg_isolation_test PRIVATE cxx_std_17)
      target_compile_definitions(
        xtbloom_cpu_linalg_isolation_test PRIVATE XTBLOOM_TEST_HAS_MKL_SHIM=1
      )
      add_test(
        NAME xtbloom.cpu.linalg_isolation
        COMMAND xtbloom_cpu_linalg_isolation_test
      )
      xtbloom_configure_linalg_runtime_test(
        xtbloom_cpu_linalg_isolation_test xtbloom.cpu.linalg_isolation
      )
      # The test itself selects the host interface in fresh children. Do not
      # preselect LP64 through the generic provider-test environment.
      set_tests_properties(xtbloom.cpu.linalg_isolation PROPERTIES
        ENVIRONMENT "MKL_THREADING_LAYER=SEQUENTIAL"
      )

      # Build deliberately unusable libraries with the same MKL component
      # SONAMEs. The adversarial test puts them first in LD_LIBRARY_PATH; only
      # the shim's configure-time DT_RPATH may select the verified cohort.
      set(xtbloom_mkl_decoy_targets "")
      foreach(component IN ITEMS intel_lp64 sequential core)
        set(decoy_target "xtbloom_mkl_${component}_decoy")
        add_library(${decoy_target} SHARED src/runtime/mkl_lp64_shim.cpp)
        target_compile_features(${decoy_target} PRIVATE cxx_std_17)
        set_target_properties(${decoy_target} PROPERTIES
          OUTPUT_NAME "mkl_${component}"
          SUFFIX "${xtbloom_mkl_runtime_suffix}"
          CXX_VISIBILITY_PRESET hidden
          VISIBILITY_INLINES_HIDDEN YES
          LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mkl-decoy"
        )
        list(APPEND xtbloom_mkl_decoy_targets ${decoy_target})
      endforeach()
      add_dependencies(
        xtbloom_cpu_linalg_isolation_test ${xtbloom_mkl_decoy_targets}
      )
      add_test(
        NAME xtbloom.cpu.linalg_isolation_search_path
        COMMAND xtbloom_cpu_linalg_isolation_test --correctness
      )
      set_tests_properties(xtbloom.cpu.linalg_isolation_search_path PROPERTIES
        ENVIRONMENT
          "MKL_THREADING_LAYER=SEQUENTIAL;LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/mkl-decoy:${XTBLOOM_CPU_LINALG_DIRECTORY}"
      )
    endif()
  elseif(NOT XTBLOOM_WHEEL_OPENBLAS)
    message(STATUS
      "xtbloom eigensolver test: disabled (set XTBLOOM_CPU_LINALG_LIBRARY or provide a system BLAS to enable)"
    )
  endif()

  if(XTBLOOM_WHEEL_OPENBLAS)
    # Prove the wheel provider is loaded through the sibling shim in a fresh
    # link-map namespace. The process-global decoy deliberately has the
    # provider's original SONAME but none of its symbols; successful inference
    # therefore proves base-namespace interposition cannot satisfy the shim.
    add_library(xtbloom_openblas_decoy SHARED
      src/runtime/openblas_lp64_shim.c
    )
    set_target_properties(xtbloom_openblas_decoy PROPERTIES
      OUTPUT_NAME scipy_openblas
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/openblas-decoy"
    )
    add_executable(xtbloom_wheel_openblas_isolation_test
      tests/cpu_linalg_isolation_test.cpp
    )
    target_link_libraries(xtbloom_wheel_openblas_isolation_test PRIVATE
      xtbloom_gfn2_cpu
      ${CMAKE_DL_LIBS}
    )
    target_compile_features(
      xtbloom_wheel_openblas_isolation_test PRIVATE cxx_std_17
    )
    target_compile_definitions(
      xtbloom_wheel_openblas_isolation_test
      PRIVATE XTBLOOM_TEST_HAS_WHEEL_OPENBLAS=1
    )
    add_dependencies(
      xtbloom_wheel_openblas_isolation_test
      xtbloom_openblas_lp64_shim
      xtbloom_openblas_decoy
    )
    add_test(
      NAME xtbloom.cpu.wheel_openblas_isolation
      COMMAND xtbloom_wheel_openblas_isolation_test
    )
    set_tests_properties(xtbloom.cpu.wheel_openblas_isolation PROPERTIES
      ENVIRONMENT
        "XTBLOOM_TEST_OPENBLAS_DECOY=$<TARGET_FILE:xtbloom_openblas_decoy>"
    )
  endif()

  if(XTBLOOM_CPU_LINALG_RUNTIME)
    # Issue #217 investigation gate: pinned temperature-continuation evidence
    # for the separated Me4N+ / Cl- ion pair on the committed tmacl fixture.
    # Requires the real LP64 eigensolver because tmacl has 42 orbitals.
    add_executable(xtbloom_scc_temperature_continuation_test
      tests/scc_temperature_continuation_test.cpp
    )
    target_compile_definitions(xtbloom_scc_temperature_continuation_test PRIVATE
      XTBLOOM_TMACL_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/conformance/inputs/tmacl.xyz"
    )
    target_link_libraries(xtbloom_scc_temperature_continuation_test PRIVATE
      xtbloom_gfn2_cpu
      Threads::Threads
    )
    target_compile_features(xtbloom_scc_temperature_continuation_test PRIVATE cxx_std_17)
    add_test(NAME xtbloom.gfn2.scc_temperature_continuation
      COMMAND xtbloom_scc_temperature_continuation_test
    )
    xtbloom_configure_linalg_runtime_test(
      xtbloom_scc_temperature_continuation_test xtbloom.gfn2.scc_temperature_continuation
    )

    # Issues #49/#50: harnesses that drive the production CPU GFN2 SCC driver
    # through the pinned restricted corpus and compare against the goldens.
    # These are now passing acceptance gates:
    #   * xtbloom_scc_trace_capture drives one case sequentially and compares the
    #     complete closed-loop trajectory with the cpu_closed_loop_v1 profile;
    #   * xtbloom_scc_trace_batch_capture runs several cases in one ragged batch
    #     and proves each lane equals its sequential trajectory, with early
    #     convergence and an optional controlled per-system failure lane;
    #   * xtbloom_scc_trace_replay injects one golden mixed state and compares a
    #     single independently executed iteration (cuda/cpu replay profile).
    # They require the host-isolated LP64 runtime for the eigensolver, so they
    # are only registered when that runtime is present.
    add_executable(xtbloom_scc_trace_capture
      tests/scc_trace_capture.cpp
    )
    target_include_directories(xtbloom_scc_trace_capture PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(xtbloom_scc_trace_capture PRIVATE
      xtbloom_gfn2_cpu
      Threads::Threads
    )
    target_compile_features(xtbloom_scc_trace_capture PRIVATE cxx_std_17)

    add_executable(xtbloom_scc_trace_batch_capture
      tests/scc_trace_batch_capture.cpp
    )
    target_include_directories(xtbloom_scc_trace_batch_capture PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(xtbloom_scc_trace_batch_capture PRIVATE
      xtbloom_gfn2_cpu
      Threads::Threads
    )
    target_compile_features(xtbloom_scc_trace_batch_capture PRIVATE cxx_std_17)

    add_executable(xtbloom_scc_trace_replay
      tests/scc_trace_replay.cpp
    )
    target_include_directories(xtbloom_scc_trace_replay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(xtbloom_scc_trace_replay PRIVATE
      xtbloom_gfn2_cpu
      Threads::Threads
    )
    target_compile_features(xtbloom_scc_trace_replay PRIVATE cxx_std_17)

    add_executable(xtbloom_scc_trace_mixer
      tests/scc_trace_mixer_replay.cpp
    )
    target_include_directories(xtbloom_scc_trace_mixer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(xtbloom_scc_trace_mixer PRIVATE
      xtbloom_gfn2_cpu
      Threads::Threads
    )
    target_compile_features(xtbloom_scc_trace_mixer PRIVATE cxx_std_17)

    # Native ragged-batch structure test: early-convergence freezing, per-lane
    # status/iteration/flags, controlled per-system failure isolation, and
    # disjoint per-system output placement under the CPU driver.  Registered
    # in every CPU configuration so sanitizer builds cover ragged execution.
    add_executable(xtbloom_scc_trace_harness_native_test
      tests/scc_trace_harness_native_test.cpp
    )
    target_compile_definitions(xtbloom_scc_trace_harness_native_test PRIVATE
      XTBLOOM_SCC_TRACE_SPEC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/conformance/scc-traces/specs"
    )
    target_include_directories(xtbloom_scc_trace_harness_native_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    target_link_libraries(xtbloom_scc_trace_harness_native_test PRIVATE
      xtbloom_gfn2_cpu
      Threads::Threads
    )
    target_compile_features(xtbloom_scc_trace_harness_native_test PRIVATE cxx_std_17)
    add_test(NAME xtbloom.gfn2.scc_trace_ragged_batch_native
      COMMAND xtbloom_scc_trace_harness_native_test
    )
    xtbloom_configure_linalg_runtime_test(
      xtbloom_scc_trace_harness_native_test xtbloom.gfn2.scc_trace_ragged_batch_native
    )

    # Sequential closed-loop gate: all five restricted goldens must match the
    # full per-iteration trajectory within the cpu_closed_loop_v1 profile.
    add_test(
      NAME xtbloom.gfn2.scc_trace_cpu_closed_loop
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tools/oracle/tblite_scc_trace/xtbloom_scc_cpu_trace.py
              --capture $<TARGET_FILE:xtbloom_scc_trace_capture>
              --corpus-dir ${CMAKE_CURRENT_SOURCE_DIR}/data/conformance/scc-traces
              --work-dir ${CMAKE_CURRENT_BINARY_DIR}/scc-trace-capture
    )
    xtbloom_configure_linalg_runtime_test(
      xtbloom_scc_trace_capture xtbloom.gfn2.scc_trace_cpu_closed_loop
    )

    # Ragged batch gate: h3_plus (converges early), nenacl (slowest), and the
    # six-point-charge water dimer run in one heterogeneous batch and each lane
    # must equal its pinned sequential trajectory.  --poison 3 adds a controlled
    # preparation-failure lane that must not corrupt or suppress the peers.
    add_test(
      NAME xtbloom.gfn2.scc_trace_ragged_batch
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tools/oracle/tblite_scc_trace/xtbloom_scc_cpu_trace.py
              --batch-capture $<TARGET_FILE:xtbloom_scc_trace_batch_capture>
              --corpus-dir ${CMAKE_CURRENT_SOURCE_DIR}/data/conformance/scc-traces
              --work-dir ${CMAKE_CURRENT_BINARY_DIR}/scc-trace-batch
              --batch-cases h3_plus nenacl water_dimer_6pc_hardness
    )
    xtbloom_configure_linalg_runtime_test(
      xtbloom_scc_trace_batch_capture xtbloom.gfn2.scc_trace_ragged_batch
    )

    # Single-iteration replay gate: every golden iteration replayed from its
    # injected mixed state must equal the pinned iteration snapshot.
    add_test(
      NAME xtbloom.gfn2.scc_trace_cpu_replay
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tools/oracle/tblite_scc_trace/xtbloom_scc_cpu_trace.py
              --replay $<TARGET_FILE:xtbloom_scc_trace_replay>
              --corpus-dir ${CMAKE_CURRENT_SOURCE_DIR}/data/conformance/scc-traces
              --work-dir ${CMAKE_CURRENT_BINARY_DIR}/scc-trace-replay
              --replay-cases h3_plus nenacl water_dimer_6pc_hardness
    )
    xtbloom_configure_linalg_runtime_test(
      xtbloom_scc_trace_replay xtbloom.gfn2.scc_trace_cpu_replay
    )

    # Independent Broyden mixer gate: replay the pinned golden residual
    # sequence through xtbloom's production mixer alone and check every state
    # transition reproduces the golden next mixed state.
    add_test(
      NAME xtbloom.gfn2.scc_trace_mixer_replay
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tools/oracle/tblite_scc_trace/xtbloom_scc_cpu_trace.py
              --mixer $<TARGET_FILE:xtbloom_scc_trace_mixer>
              --corpus-dir ${CMAKE_CURRENT_SOURCE_DIR}/data/conformance/scc-traces
              --work-dir ${CMAKE_CURRENT_BINARY_DIR}/scc-trace-mixer
              --mixer-cases h3_plus nenacl water_dimer_6pc_hardness
    )
    xtbloom_configure_linalg_runtime_test(
      xtbloom_scc_trace_mixer xtbloom.gfn2.scc_trace_mixer_replay
    )
  endif()

  add_executable(xtbloom_parameters_header_test tests/parameters/gfn2_header_test.cpp)
  target_include_directories(xtbloom_parameters_header_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_features(xtbloom_parameters_header_test PRIVATE cxx_std_17)
  add_test(NAME xtbloom.parameters.header COMMAND xtbloom_parameters_header_test)

  add_test(
    NAME xtbloom.parameters.generated
    COMMAND ${Python3_EXECUTABLE} tools/parameters/generate_gfn2.py --check
  )
  add_test(
    NAME xtbloom.parameters.python
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
            -s tests/parameters -p "test_*.py" -v
  )
  add_test(
    NAME xtbloom.conformance.manifest
    COMMAND ${Python3_EXECUTABLE} tools/conformance/xtbloom_conformance.py check
  )
  add_test(
    NAME xtbloom.conformance.tmacl_evidence
    COMMAND ${Python3_EXECUTABLE} tools/conformance/tmacl_evidence.py
  )
  add_test(
    NAME xtbloom.conformance.python
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
            -s tests/conformance -p "test_*.py" -v
  )
  if(BUILD_SHARED_LIBS AND XTBLOOM_CPU_LINALG_RUNTIME)
    add_test(
      NAME xtbloom.conformance.public_cpu
      COMMAND ${Python3_EXECUTABLE} tools/conformance/xtbloom_public_api.py
              --library $<TARGET_FILE:xtbloom> --backend cpu --memory-mode host
              --actual-dir ${CMAKE_CURRENT_BINARY_DIR}/conformance/public-api
    )
    set_tests_properties(xtbloom.conformance.public_cpu PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      ENVIRONMENT "MKL_INTERFACE_LAYER=LP64;MKL_THREADING_LAYER=SEQUENTIAL"
      ENVIRONMENT_MODIFICATION
        "${XTBLOOM_RUNTIME_LIBRARY_PATH_VARIABLE}=path_list_prepend:${XTBLOOM_CPU_LINALG_DIRECTORY}"
      LABELS "conformance;public-api"
    )
    add_test(
      NAME xtbloom.conformance.invariants_cpu
      COMMAND ${Python3_EXECUTABLE} tools/conformance/xtbloom_invariants.py
              --library $<TARGET_FILE:xtbloom> --backend cpu --memory-mode host
    )
    set_tests_properties(xtbloom.conformance.invariants_cpu PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      ENVIRONMENT "MKL_INTERFACE_LAYER=LP64;MKL_THREADING_LAYER=SEQUENTIAL"
      ENVIRONMENT_MODIFICATION
        "${XTBLOOM_RUNTIME_LIBRARY_PATH_VARIABLE}=path_list_prepend:${XTBLOOM_CPU_LINALG_DIRECTORY}"
      LABELS "conformance;public-api"
    )
    if(XTBLOOM_HAS_CUDA)
      add_test(
        NAME xtbloom.conformance.public_cuda
        COMMAND ${Python3_EXECUTABLE} tools/conformance/xtbloom_public_api.py
                --library $<TARGET_FILE:xtbloom> --backend cuda --memory-mode host
                --actual-dir ${CMAKE_CURRENT_BINARY_DIR}/conformance/public-api
                --skip-backend-unavailable
      )
      set_tests_properties(xtbloom.conformance.public_cuda PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        ENVIRONMENT "MKL_INTERFACE_LAYER=LP64;MKL_THREADING_LAYER=SEQUENTIAL"
        ENVIRONMENT_MODIFICATION
          "${XTBLOOM_RUNTIME_LIBRARY_PATH_VARIABLE}=path_list_prepend:${XTBLOOM_CPU_LINALG_DIRECTORY}"
        LABELS "conformance;public-api"
        SKIP_RETURN_CODE 77
      )
      add_test(
        NAME xtbloom.conformance.invariants_cuda
        COMMAND ${Python3_EXECUTABLE} tools/conformance/xtbloom_invariants.py
                --library $<TARGET_FILE:xtbloom> --backend cuda --memory-mode host
                --skip-backend-unavailable
      )
      set_tests_properties(xtbloom.conformance.invariants_cuda PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        ENVIRONMENT "MKL_INTERFACE_LAYER=LP64;MKL_THREADING_LAYER=SEQUENTIAL"
        ENVIRONMENT_MODIFICATION
          "${XTBLOOM_RUNTIME_LIBRARY_PATH_VARIABLE}=path_list_prepend:${XTBLOOM_CPU_LINALG_DIRECTORY}"
        LABELS "conformance;public-api"
        SKIP_RETURN_CODE 77
      )
      foreach(memory_mode IN ITEMS device mixed)
        add_test(
          NAME xtbloom.conformance.public_cuda_${memory_mode}
          COMMAND ${Python3_EXECUTABLE} tools/conformance/xtbloom_public_api.py
                  --library $<TARGET_FILE:xtbloom> --backend cuda
                  --memory-mode ${memory_mode}
                  --actual-dir ${CMAKE_CURRENT_BINARY_DIR}/conformance/public-api
                  --skip-backend-unavailable
        )
        set_tests_properties(xtbloom.conformance.public_cuda_${memory_mode} PROPERTIES
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          ENVIRONMENT "MKL_INTERFACE_LAYER=LP64;MKL_THREADING_LAYER=SEQUENTIAL"
          ENVIRONMENT_MODIFICATION
            "${XTBLOOM_RUNTIME_LIBRARY_PATH_VARIABLE}=path_list_prepend:${XTBLOOM_CPU_LINALG_DIRECTORY}"
          LABELS "conformance;public-api"
          SKIP_RETURN_CODE 77
        )
        add_test(
          NAME xtbloom.conformance.invariants_cuda_${memory_mode}
          COMMAND ${Python3_EXECUTABLE} tools/conformance/xtbloom_invariants.py
                  --library $<TARGET_FILE:xtbloom> --backend cuda
                  --memory-mode ${memory_mode}
                  --skip-backend-unavailable
        )
        set_tests_properties(xtbloom.conformance.invariants_cuda_${memory_mode} PROPERTIES
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          ENVIRONMENT "MKL_INTERFACE_LAYER=LP64;MKL_THREADING_LAYER=SEQUENTIAL"
          ENVIRONMENT_MODIFICATION
            "${XTBLOOM_RUNTIME_LIBRARY_PATH_VARIABLE}=path_list_prepend:${XTBLOOM_CPU_LINALG_DIRECTORY}"
          LABELS "conformance;public-api"
          SKIP_RETURN_CODE 77
        )
      endforeach()
    endif()
  endif()
  add_test(
    NAME xtbloom.licensing
    COMMAND ${Python3_EXECUTABLE} tools/licensing/check_licenses.py
            --source-root ${CMAKE_CURRENT_SOURCE_DIR}
  )
  add_test(
    NAME xtbloom.licensing_unit
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
            -s tests/licensing -p "test_*.py" -v
  )
  add_test(
    NAME xtbloom.oracle.tblite_observer_patch
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
            -s tests/oracle -p "test_*.py" -v
  )
  add_test(
    NAME xtbloom.oracle.scc_corpus
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/oracle/tblite_scc_trace/generate_scc_corpus.py
            --source-root ${CMAKE_CURRENT_SOURCE_DIR}
            --corpus-dir ${CMAKE_CURRENT_SOURCE_DIR}/data/conformance/scc-traces
            --check
  )
  set_tests_properties(xtbloom.oracle.scc_corpus PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    LABELS "conformance;oracle"
  )
  set_tests_properties(
    xtbloom.parameters.generated
    xtbloom.parameters.python
    xtbloom.licensing
    xtbloom.licensing_unit
    xtbloom.conformance.manifest
    xtbloom.conformance.tmacl_evidence
    xtbloom.conformance.python
    xtbloom.oracle.tblite_observer_patch
    PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()

install(TARGETS xtbloom EXPORT xtbloomTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${XTBLOOM_GENERATED_INCLUDE_DIR}/xtbloom/version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xtbloom
)
# Legal material is part of the installed SDK contract. Keep project terms,
# third-party notices, and machine-readable provenance together so native
# consumers receive the same audit trail as source and Python distributions.
install(FILES LICENSE CUDA_MKL_LINKING_EXCEPTION THIRD_PARTY_NOTICES.md
  DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/xtbloom
)
install(DIRECTORY LICENSES/
  DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/xtbloom/third-party
)
install(FILES
  data/parameters/manifest.json
  data/parameters/sto_manifest.json
  data/parameters/spin_manifest.json
  data/parameters/d4_manifest.json
  data/parameters/mctc_manifest.json
  cmake/3rdparty/implib_manifest.json
  cmake/3rdparty/scipy_openblas32_manifest.json
  DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/xtbloom/provenance
)
install(FILES data/parameters/d4.NOTICE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/xtbloom/third-party/d4
)
install(DIRECTORY data/parameters/licenses/
  DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/xtbloom/third-party/d4
)
install(EXPORT xtbloomTargets
  FILE xtbloomTargets.cmake
  NAMESPACE xtbloom::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xtbloom
)

# A CUDA-enabled build records the compatible major version of the CUDA
# runtime it dlopens so the installed CMake config can warn consumers that the
# runtime they have may be incompatible (the library itself no longer links
# against the toolkit; see src/runtime/cuda_dlopen.c).
if(NOT XTBLOOM_HAS_CUDA)
  set(XTBLOOM_CUDA_TOOLKIT_MAJOR 0)
  set(XTBLOOM_CUDA_TOOLKIT_MINOR 0)
endif()
configure_package_config_file(
  cmake/xtbloomConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/xtbloomConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xtbloom
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/xtbloomConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/xtbloomConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/xtbloomConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xtbloom
)

message(STATUS "xtbloom product version: ${XTBLOOM_VERSION_FULL}")
message(STATUS "xtbloom CUDA backend: ${XTBLOOM_HAS_CUDA}")

if(XTBLOOM_BUILD_WEB_DEMO)
  add_subdirectory(web)
endif()
