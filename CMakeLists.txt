cmake_minimum_required(VERSION 3.24)

project(
  gpuxtb
  VERSION 0.1.0
  DESCRIPTION "Batched CPU and GPU inference for GFN-xTB"
  LANGUAGES C CXX
)

include(CheckLanguage)
include(CheckLibraryExists)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Threads REQUIRED)

set(GPUXTB_ENABLE_CUDA "AUTO" CACHE STRING "Build the CUDA backend: AUTO, ON, or OFF")
set_property(CACHE GPUXTB_ENABLE_CUDA PROPERTY STRINGS AUTO ON OFF)
option(GPUXTB_BUILD_TESTS "Build gpuxtb tests" ${PROJECT_IS_TOP_LEVEL})
option(GPUXTB_PYTHON_WHEEL
  "Install an unversioned single-copy shared library for a Python wheel" OFF)
set(GPUXTB_MKL_RT_LIBRARY "" CACHE FILEPATH
  "Optional shared LP64 BLAS/LAPACK runtime for the CPU eigensolver (e.g. libmkl_rt.so or libopenblas.so). When unset, compatible runtimes are auto-discovered with find_package(BLAS/LAPACK)."
)

string(TOUPPER "${GPUXTB_ENABLE_CUDA}" GPUXTB_ENABLE_CUDA_NORMALIZED)
if(NOT GPUXTB_ENABLE_CUDA_NORMALIZED MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "GPUXTB_ENABLE_CUDA must be AUTO, ON, or OFF")
endif()

set(GPUXTB_HAS_CUDA OFF)
if(NOT GPUXTB_ENABLE_CUDA_NORMALIZED STREQUAL "OFF")
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(GPUXTB_HAS_CUDA ON)
  elseif(GPUXTB_ENABLE_CUDA_NORMALIZED STREQUAL "ON")
    message(FATAL_ERROR "GPUXTB_ENABLE_CUDA=ON, but no CUDA compiler was found")
  endif()
endif()

# The eigensolver requires one dlopen-able runtime that provides LAPACKE and
# CBLAS together. FindBLAS/FindLAPACK may also report split reference libraries
# (for example liblapack.so plus libblas.so), so validate every candidate as a
# self-contained runtime before baking its path into the library or enabling
# production-backend tests.
function(gpuxtb_cpu_linalg_has_symbol_set candidate symbol_prefix result)
  set(symbol_set_complete ON)
  foreach(symbol IN ITEMS
      LAPACKE_dpotrf_work
      LAPACKE_dpocon_work
      LAPACKE_dsyevd_work
      cblas_dtrsm
      cblas_dgemm
  )
    unset(GPUXTB_CPU_LINALG_SYMBOL_FOUND CACHE)
    check_library_exists(
      "${candidate}" "${symbol_prefix}${symbol}" ""
      GPUXTB_CPU_LINALG_SYMBOL_FOUND
    )
    if(NOT GPUXTB_CPU_LINALG_SYMBOL_FOUND)
      set(symbol_set_complete OFF)
      unset(GPUXTB_CPU_LINALG_SYMBOL_FOUND CACHE)
      break()
    endif()
    unset(GPUXTB_CPU_LINALG_SYMBOL_FOUND CACHE)
  endforeach()
  set(${result} ${symbol_set_complete} PARENT_SCOPE)
endfunction()

function(gpuxtb_validate_cpu_linalg_runtime candidate result symbol_prefix_result)
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
  # call the provider directly with gpuxtb's 32-bit LapackInt declarations.
  file(STRINGS "${candidate_realpath}" ilp64_markers REGEX "USE64BITINT" LIMIT_COUNT 1)
  if(ilp64_markers)
    set(${result} OFF PARENT_SCOPE)
    set(${symbol_prefix_result} "" PARENT_SCOPE)
    return()
  endif()

  gpuxtb_cpu_linalg_has_symbol_set("${candidate}" "" has_standard_symbols)
  if(has_standard_symbols)
    set(${result} ON PARENT_SCOPE)
    set(${symbol_prefix_result} "" PARENT_SCOPE)
    return()
  endif()

  # scipy-openblas32 deliberately prefixes its public ABI with ``scipy_`` to
  # avoid interposing on another BLAS already loaded by NumPy or the host.
  gpuxtb_cpu_linalg_has_symbol_set("${candidate}" "scipy_" has_scipy_symbols)
  set(${result} ${has_scipy_symbols} PARENT_SCOPE)
  if(has_scipy_symbols)
    set(${symbol_prefix_result} "scipy_" PARENT_SCOPE)
  else()
    set(${symbol_prefix_result} "" PARENT_SCOPE)
  endif()
endfunction()

# The CPU eigensolver dlopens an LP64 BLAS/LAPACK runtime (MKL on x86_64,
# OpenBLAS where Intel MKL does not exist, e.g. linux/aarch64). Bake a verified
# absolute path into the library so a non-system runtime can be found without
# LD_LIBRARY_PATH. This remains a lazy dlopen: libgpuxtb never gains a hard
# DT_NEEDED dependency that would make the CUDA backend unloadable when no BLAS
# is installed.
set(GPUXTB_CPU_LINALG_RUNTIME "")
set(GPUXTB_CPU_LINALG_SYMBOL_PREFIX "")
if(GPUXTB_MKL_RT_LIBRARY)
  gpuxtb_validate_cpu_linalg_runtime(
    "${GPUXTB_MKL_RT_LIBRARY}"
    configured_runtime_is_compatible
    configured_runtime_symbol_prefix
  )
  if(NOT configured_runtime_is_compatible)
    message(FATAL_ERROR
      "GPUXTB_MKL_RT_LIBRARY=${GPUXTB_MKL_RT_LIBRARY} does not provide a "
      "complete LP64 LAPACKE+CBLAS runtime"
    )
  endif()
  set(GPUXTB_CPU_LINALG_RUNTIME "${GPUXTB_MKL_RT_LIBRARY}")
  set(GPUXTB_CPU_LINALG_SYMBOL_PREFIX "${configured_runtime_symbol_prefix}")
else()
  if(DEFINED BLA_SIZEOF_INTEGER)
    set(GPUXTB_SAVED_BLA_SIZEOF_INTEGER "${BLA_SIZEOF_INTEGER}")
    set(GPUXTB_HAD_BLA_SIZEOF_INTEGER ON)
  else()
    set(GPUXTB_HAD_BLA_SIZEOF_INTEGER OFF)
  endif()
  # gpuxtb's public and internal eigensolver ABI uses 32-bit LAPACK integers;
  # prevent FindBLAS/FindLAPACK from selecting an ILP64 provider automatically.
  set(BLA_SIZEOF_INTEGER 4)
  find_package(BLAS QUIET)
  if(BLAS_FOUND)
    find_package(LAPACK QUIET)
  endif()
  if(GPUXTB_HAD_BLA_SIZEOF_INTEGER)
    set(BLA_SIZEOF_INTEGER "${GPUXTB_SAVED_BLA_SIZEOF_INTEGER}")
  else()
    unset(BLA_SIZEOF_INTEGER)
  endif()
  set(GPUXTB_CPU_LINALG_CANDIDATES "")
  foreach(candidate IN LISTS LAPACK_LIBRARIES BLAS_LIBRARIES)
    if(NOT IS_ABSOLUTE "${candidate}")
      continue()
    endif()
    string(TOLOWER "${candidate}" candidate_lower)
    if(candidate_lower MATCHES "intel|mkl_rt|openblas|scipy_openblas")
      list(APPEND GPUXTB_CPU_LINALG_CANDIDATES "${candidate}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES GPUXTB_CPU_LINALG_CANDIDATES)
  foreach(candidate IN LISTS GPUXTB_CPU_LINALG_CANDIDATES)
    gpuxtb_validate_cpu_linalg_runtime(
      "${candidate}" candidate_is_compatible candidate_symbol_prefix
    )
    if(candidate_is_compatible)
      set(GPUXTB_CPU_LINALG_RUNTIME "${candidate}")
      set(GPUXTB_CPU_LINALG_SYMBOL_PREFIX "${candidate_symbol_prefix}")
      break()
    endif()
    message(STATUS
      "gpuxtb ignored incomplete CPU linear-algebra runtime: ${candidate}"
    )
  endforeach()
endif()
if(GPUXTB_CPU_LINALG_RUNTIME)
  message(STATUS "gpuxtb CPU eigensolver BLAS/LAPACK runtime: ${GPUXTB_CPU_LINALG_RUNTIME}")
endif()

add_library(gpuxtb_gfn2_cpu OBJECT
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
target_compile_features(gpuxtb_gfn2_cpu PUBLIC cxx_std_17)
target_include_directories(gpuxtb_gfn2_cpu PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
set_target_properties(gpuxtb_gfn2_cpu PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  POSITION_INDEPENDENT_CODE ON
  VISIBILITY_INLINES_HIDDEN YES
)
if(CMAKE_DL_LIBS)
  target_link_libraries(gpuxtb_gfn2_cpu PUBLIC ${CMAKE_DL_LIBS})
endif()
if(GPUXTB_CPU_LINALG_RUNTIME)
  target_compile_definitions(gpuxtb_gfn2_cpu PRIVATE
    "GPUXTB_CONFIGURED_CPU_LINALG_RUNTIME=\"${GPUXTB_CPU_LINALG_RUNTIME}\""
  )
endif()

add_library(gpuxtb
  src/api.cpp
  src/runtime/backend.cpp
  src/runtime/gfn2_cpu_execution.cpp
  src/runtime/validation.cpp
  $<TARGET_OBJECTS:gpuxtb_gfn2_cpu>
)
add_library(gpuxtb::gpuxtb ALIAS gpuxtb)

target_compile_features(gpuxtb PUBLIC cxx_std_17)
target_include_directories(gpuxtb
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_definitions(gpuxtb PRIVATE GPUXTB_BUILDING_LIBRARY)
target_link_libraries(gpuxtb PRIVATE Threads::Threads)
if(CMAKE_DL_LIBS)
  target_link_libraries(gpuxtb PRIVATE ${CMAKE_DL_LIBS})
endif()
set_target_properties(gpuxtb PROPERTIES
  CUDA_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
)
if(NOT GPUXTB_PYTHON_WHEEL)
  # Native SDK installs retain the normal SONAME chain.  Wheel archives often
  # dereference those symlinks and would otherwise store the large CUDA shared
  # object three times, so wheel builds deliberately install one unversioned
  # library instead.
  set_target_properties(gpuxtb PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
  )
endif()
if(UNIX AND NOT APPLE)
  target_link_options(gpuxtb PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpuxtb.map"
  )
  set_property(TARGET gpuxtb APPEND PROPERTY
    LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpuxtb.map
  )
endif()

if(GPUXTB_HAS_CUDA)
  # implib's hand-written trampolines are GNU ELF assembly for Linux.  Keep the
  # accepted architecture aliases explicit so an unsupported wheel cannot
  # configure successfully and then fail at load time.
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "the CUDA dlopen backend currently requires Linux ELF")
  endif()
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _gpuxtb_cuda_processor)
  if(_gpuxtb_cuda_processor MATCHES "^(x86_64|amd64)$")
    set(GPUXTB_CUDA_IMPLIB_TARGET x86_64)
  elseif(_gpuxtb_cuda_processor MATCHES "^(aarch64|arm64)$")
    set(GPUXTB_CUDA_IMPLIB_TARGET aarch64)
  else()
    message(FATAL_ERROR
      "the CUDA dlopen backend supports only Linux x86_64 and aarch64, not "
      "'${CMAKE_SYSTEM_PROCESSOR}'")
  endif()

  # The generated shims contain all CUDA-facing definitions needed by the
  # backend.  CUDA device linking still contributes cudadevrt for RDC, but the
  # host link must not silently reintroduce cudart_static.
  set(CMAKE_CUDA_RUNTIME_LIBRARY None)
  enable_language(C ASM)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  find_program(GPUXTB_READELF NAMES readelf REQUIRED)

  set(GPUXTB_CUDA_IMPLIB_GEN
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/3rdparty/implib/implib-gen.py")
  set(GPUXTB_CUDA_IMPLIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/cuda_implib")
  file(MAKE_DIRECTORY "${GPUXTB_CUDA_IMPLIB_DIR}")

  # Curate the object-level names actually referenced by gpuxtb.  CUDA headers
  # normalize several source API names to versioned ELF aliases; listing those
  # aliases here keeps -z defs useful and prevents a full-library catch-all from
  # masking a new dependency or an ABI-unsafe fallback category.
  set(GPUXTB_CUDART_SYMBOLS
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
  set(GPUXTB_CUBLAS_SYMBOLS
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
  )
  set(GPUXTB_CUSOLVER_SYMBOLS
    cusolverDnCreate
    cusolverDnCreateParams
    cusolverDnDestroy
    cusolverDnDestroyParams
    cusolverDnDpotrfBatched
    cusolverDnGetStream
    cusolverDnSetStream
    cusolverDnXsyevBatched
    cusolverDnXsyevBatched_bufferSize
    cusolverGetProperty
  )
  set(GPUXTB_CUDA_DRIVER_SYMBOLS
    cuGetErrorString
    cuMemGetAddressRange_v2
  )

  function(gpuxtb_write_implib_symbol_file output_path)
    file(WRITE "${output_path}" "")
    foreach(_gpuxtb_symbol IN LISTS ARGN)
      file(APPEND "${output_path}" "${_gpuxtb_symbol}\n")
    endforeach()
  endfunction()

  function(gpuxtb_read_elf_soname library_path output_variable)
    execute_process(
      COMMAND "${GPUXTB_READELF}" -dW "${library_path}"
      RESULT_VARIABLE _gpuxtb_readelf_status
      OUTPUT_VARIABLE _gpuxtb_readelf_output
      ERROR_VARIABLE _gpuxtb_readelf_error)
    if(NOT _gpuxtb_readelf_status EQUAL 0)
      message(FATAL_ERROR
        "failed to read CUDA library SONAME from '${library_path}': "
        "${_gpuxtb_readelf_error}")
    endif()
    string(REGEX MATCH "\\(SONAME\\)[^[]*\\[([^]]+)\\]"
      _gpuxtb_soname_match "${_gpuxtb_readelf_output}")
    if(NOT _gpuxtb_soname_match)
      message(FATAL_ERROR "CUDA library '${library_path}' has no ELF SONAME")
    endif()
    set(${output_variable} "${CMAKE_MATCH_1}" PARENT_SCOPE)
  endfunction()

  foreach(_gpuxtb_cuda_target IN ITEMS CUDA::cudart CUDA::cublas CUDA::cusolver)
    get_target_property(_gpuxtb_cuda_location
      "${_gpuxtb_cuda_target}" IMPORTED_LOCATION)
    if(NOT _gpuxtb_cuda_location)
      message(FATAL_ERROR
        "cannot locate ${_gpuxtb_cuda_target} for dlopen shim generation")
    endif()
    list(APPEND _gpuxtb_cuda_implib_inputs "${_gpuxtb_cuda_location}")
  endforeach()
  if(NOT CUDA_cuda_driver_LIBRARY)
    message(FATAL_ERROR "cannot locate CUDA::cuda_driver for dlopen shim generation")
  endif()
  list(APPEND _gpuxtb_cuda_implib_inputs "${CUDA_cuda_driver_LIBRARY}")
  set(_gpuxtb_cuda_implib_bases
    libcudart.so libcublas.so libcusolver.so libcuda.so)
  set(_gpuxtb_cuda_symbol_sets
    GPUXTB_CUDART_SYMBOLS
    GPUXTB_CUBLAS_SYMBOLS
    GPUXTB_CUSOLVER_SYMBOLS
    GPUXTB_CUDA_DRIVER_SYMBOLS)

  set(GPUXTB_CUDA_DYNLOADER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/cuda_dlopen.c")
  set(_gpuxtb_cuda_sonames "")
  foreach(_gpuxtb_cuda_input _gpuxtb_cuda_base _gpuxtb_cuda_symbol_set
      IN ZIP_LISTS
        _gpuxtb_cuda_implib_inputs
        _gpuxtb_cuda_implib_bases
        _gpuxtb_cuda_symbol_sets)
    get_filename_component(_gpuxtb_cuda_input_base "${_gpuxtb_cuda_input}" NAME)
    if(NOT _gpuxtb_cuda_input_base STREQUAL _gpuxtb_cuda_base)
      message(FATAL_ERROR
        "expected ${_gpuxtb_cuda_base} from FindCUDAToolkit, got "
        "'${_gpuxtb_cuda_input_base}'")
    endif()
    set(_gpuxtb_cuda_symbol_file
      "${GPUXTB_CUDA_IMPLIB_DIR}/${_gpuxtb_cuda_base}.symbols")
    gpuxtb_write_implib_symbol_file(
      "${_gpuxtb_cuda_symbol_file}" ${${_gpuxtb_cuda_symbol_set}})
    if(_gpuxtb_cuda_base STREQUAL "libcuda.so")
      set(_gpuxtb_cuda_soname "libcuda.so.1")
    else()
      gpuxtb_read_elf_soname("${_gpuxtb_cuda_input}" _gpuxtb_cuda_soname)
    endif()
    list(APPEND _gpuxtb_cuda_sonames "${_gpuxtb_cuda_soname}")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" "${GPUXTB_CUDA_IMPLIB_GEN}"
              "${_gpuxtb_cuda_input}"
              --target "${GPUXTB_CUDA_IMPLIB_TARGET}"
              --no-dlopen
              --lazy-load
              --library-load-name "${_gpuxtb_cuda_soname}"
              --dlsym-callback gpu_xtb_cuda_dlsym
              --symbol-list "${_gpuxtb_cuda_symbol_file}"
              --outdir "${GPUXTB_CUDA_IMPLIB_DIR}"
      RESULT_VARIABLE _gpuxtb_cuda_implib_result
      OUTPUT_VARIABLE _gpuxtb_cuda_implib_output
      ERROR_VARIABLE _gpuxtb_cuda_implib_error
      COMMAND_ERROR_IS_FATAL ANY)
    list(APPEND GPUXTB_CUDA_DYNLOADER_SOURCES
      "${GPUXTB_CUDA_IMPLIB_DIR}/${_gpuxtb_cuda_base}.tramp.S"
      "${GPUXTB_CUDA_IMPLIB_DIR}/${_gpuxtb_cuda_base}.init.c")
  endforeach()
  list(GET _gpuxtb_cuda_sonames 0 GPUXTB_CUDART_SONAME)
  list(GET _gpuxtb_cuda_sonames 1 GPUXTB_CUBLAS_SONAME)
  list(GET _gpuxtb_cuda_sonames 2 GPUXTB_CUSOLVER_SONAME)
  list(GET _gpuxtb_cuda_sonames 3 GPUXTB_CUDA_DRIVER_SONAME)

  # Runtime bootstrap consumes the same configure-time list used to generate
  # the trampolines, so cohort preflight cannot drift from the link closure.
  set(GPUXTB_CUDA_DLOPEN_SYMBOLS_HEADER
    "${GPUXTB_CUDA_IMPLIB_DIR}/cuda_dlopen_symbols.h")
  file(WRITE "${GPUXTB_CUDA_DLOPEN_SYMBOLS_HEADER}"
    "/* Generated by CMake; do not edit. */\n#pragma once\n")
  foreach(_gpuxtb_cuda_name IN ITEMS CUDART CUBLAS CUSOLVER CUDA_DRIVER)
    string(TOLOWER "${_gpuxtb_cuda_name}" _gpuxtb_cuda_name_lower)
    file(APPEND "${GPUXTB_CUDA_DLOPEN_SYMBOLS_HEADER}"
      "static const char* const gpu_xtb_${_gpuxtb_cuda_name_lower}_required_symbols[] = {\n")
    foreach(_gpuxtb_symbol IN LISTS GPUXTB_${_gpuxtb_cuda_name}_SYMBOLS)
      file(APPEND "${GPUXTB_CUDA_DLOPEN_SYMBOLS_HEADER}"
        "    \"${_gpuxtb_symbol}\",\n")
    endforeach()
    file(APPEND "${GPUXTB_CUDA_DLOPEN_SYMBOLS_HEADER}"
      "    (const char*)0,\n};\n")
  endforeach()

  add_library(gpuxtb_cuda_backend OBJECT
    ${GPUXTB_CUDA_DYNLOADER_SOURCES}
    src/backends/cuda/cuda_runtime.cu
    src/runtime/cuda_descriptor_validation.cu
    src/runtime/gfn2_cuda_topology_staging.cu
    src/runtime/gfn2_cuda_execution.cu
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
  target_include_directories(gpuxtb_cuda_backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_include_directories(gpuxtb_cuda_backend PRIVATE
    ${GPUXTB_CUDA_IMPLIB_DIR}
  )
  # The startup bootstrap uses pthread_once and a diagnostic mutex. Preserve
  # that dependency for every executable that consumes the backend objects,
  # not just for the final gpuxtb library target.
  target_link_libraries(gpuxtb_cuda_backend PUBLIC Threads::Threads)
  # The CUDA runtime/math/driver shared libraries are deliberately NOT linked.
  # The priority-101 bootstrap pre-resolves the complete shim cohort before
  # ordinary NVCC registration constructors run.
  target_compile_definitions(gpuxtb_cuda_backend PRIVATE
    GPUXTB_HAS_CUDA=1
    GPUXTB_CUDART_SONAME="${GPUXTB_CUDART_SONAME}"
    GPUXTB_CUBLAS_SONAME="${GPUXTB_CUBLAS_SONAME}"
    GPUXTB_CUSOLVER_SONAME="${GPUXTB_CUSOLVER_SONAME}"
    GPUXTB_CUDA_DRIVER_SONAME="${GPUXTB_CUDA_DRIVER_SONAME}"
  )
  if(GPUXTB_BUILD_TESTS)
    # White-box CUDA kernels exercise otherwise unreachable transactional
    # overflow boundaries without shipping test entry points in release builds.
    target_compile_definitions(gpuxtb_cuda_backend PRIVATE GPUXTB_CUDA_TEST_HOOKS=1)
  endif()
  set_target_properties(gpuxtb_cuda_backend PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_VISIBILITY_PRESET hidden
    CUDA_RUNTIME_LIBRARY None
    C_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
    VISIBILITY_INLINES_HIDDEN YES
  )

  target_sources(gpuxtb PRIVATE
    $<TARGET_OBJECTS:gpuxtb_cuda_backend>
  )
  # The NVIDIA runtime/math/driver libraries are intentionally not linked or
  # exported transitively; startup pre-resolution fills hidden trampoline
  # tables and the installed library does not demand them at load time.
  target_compile_definitions(gpuxtb PRIVATE GPUXTB_HAS_CUDA=1)
  set_target_properties(gpuxtb PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY None
  )
  target_link_options(gpuxtb PRIVATE "LINKER:-z,defs")
endif()

if(GPUXTB_BUILD_TESTS)
  enable_testing()
  find_package(Threads REQUIRED)
  find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)

  add_test(
    NAME gpuxtb.benchmarks.self_test
    COMMAND ${Python3_EXECUTABLE} -m unittest -v benchmarks.test_run
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  if(GPUXTB_CPU_LINALG_RUNTIME)
    get_filename_component(GPUXTB_CPU_LINALG_DIRECTORY
      "${GPUXTB_CPU_LINALG_RUNTIME}" DIRECTORY
    )
    if(WIN32)
      set(GPUXTB_RUNTIME_LIBRARY_PATH_VARIABLE PATH)
    elseif(APPLE)
      set(GPUXTB_RUNTIME_LIBRARY_PATH_VARIABLE DYLD_LIBRARY_PATH)
    else()
      set(GPUXTB_RUNTIME_LIBRARY_PATH_VARIABLE LD_LIBRARY_PATH)
    endif()

    function(gpuxtb_configure_linalg_runtime_test target_name test_name)
      # The production eigensolver dlopens the BLAS/LAPACK runtime by path and
      # soname. Give CTest (including sanitizer builds) an explicit loader path
      # instead of relying only on the executable's build RPATH being inherited
      # by dlopen. The MKL env vars are harmless for OpenBLAS runtimes.
      set_target_properties(${target_name} PROPERTIES
        BUILD_RPATH "${GPUXTB_CPU_LINALG_DIRECTORY}"
      )
      set_tests_properties(${test_name} PROPERTIES
        ENVIRONMENT "MKL_INTERFACE_LAYER=LP64;MKL_THREADING_LAYER=SEQUENTIAL"
        ENVIRONMENT_MODIFICATION
          "${GPUXTB_RUNTIME_LIBRARY_PATH_VARIABLE}=path_list_prepend:${GPUXTB_CPU_LINALG_DIRECTORY}"
      )
    endfunction()
  endif()

  add_executable(gpuxtb_c_api_test tests/c_api_test.c)
  target_link_libraries(gpuxtb_c_api_test PRIVATE gpuxtb::gpuxtb)
  set_target_properties(gpuxtb_c_api_test PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED YES)
  add_test(NAME gpuxtb.c_api COMMAND gpuxtb_c_api_test)

  add_executable(gpuxtb_runtime_test tests/runtime_test.cpp)
  target_link_libraries(gpuxtb_runtime_test PRIVATE gpuxtb::gpuxtb)
  target_compile_features(gpuxtb_runtime_test PRIVATE cxx_std_17)
  if(GPUXTB_HAS_CUDA)
    target_compile_definitions(gpuxtb_runtime_test PRIVATE GPUXTB_TEST_HAS_CUDA=1)
  endif()
  add_test(NAME gpuxtb.runtime COMMAND gpuxtb_runtime_test)
  if(GPUXTB_CPU_LINALG_RUNTIME)
    gpuxtb_configure_linalg_runtime_test(gpuxtb_runtime_test gpuxtb.runtime)
  endif()

  if(GPUXTB_CPU_LINALG_RUNTIME)
    add_executable(gpuxtb_cpu_public_inference_test
      tests/cpu_public_inference_test.cpp
    )
    target_link_libraries(gpuxtb_cpu_public_inference_test PRIVATE
      gpuxtb::gpuxtb
      Threads::Threads
    )
    target_compile_features(gpuxtb_cpu_public_inference_test PRIVATE cxx_std_17)
    add_test(NAME gpuxtb.cpu.public_inference COMMAND gpuxtb_cpu_public_inference_test)
    gpuxtb_configure_linalg_runtime_test(
      gpuxtb_cpu_public_inference_test gpuxtb.cpu.public_inference
    )
  endif()

  if(BUILD_SHARED_LIBS AND UNIX AND NOT APPLE)
    add_test(
      NAME gpuxtb.abi_symbols
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/abi/check_symbols.py
              --nm ${CMAKE_NM} --library $<TARGET_FILE:gpuxtb>
    )
    if(GPUXTB_HAS_CUDA)
      find_program(GPUXTB_READELF NAMES readelf REQUIRED)
      add_test(
        NAME gpuxtb.abi_cuda_no_proprietary_needed
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/abi/check_cuda_dependencies.py
                --readelf ${GPUXTB_READELF} --library $<TARGET_FILE:gpuxtb>
      )
      add_test(
        NAME gpuxtb.abi_cuda_dependency_checker
        COMMAND ${Python3_EXECUTABLE} -m unittest discover
                -s ${CMAKE_CURRENT_SOURCE_DIR}/tests/abi
                -p test_check_cuda_dependencies.py -v
      )
      add_test(
        NAME gpuxtb.abi_cuda_direct_dso_load
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/abi/check_cuda_dlopen_runtime.py
                --library $<TARGET_FILE:gpuxtb> --mode load
      )
      add_test(
        NAME gpuxtb.abi_cuda_no_runtime_concurrent
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/abi/check_cuda_dlopen_runtime.py
                --library $<TARGET_FILE:gpuxtb>
                --mode concurrent-unavailable --threads 16 --calls-per-thread 16
      )
      set_tests_properties(gpuxtb.abi_cuda_no_runtime_concurrent PROPERTIES
        ENVIRONMENT "GPUXTB_CUDA_FORCE_FALLBACK=1"
      )
    endif()
  endif()

  add_executable(gpuxtb_batch_validation_test
    tests/batch_validation_test.cpp
    src/runtime/validation.cpp
  )
  target_include_directories(gpuxtb_batch_validation_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_compile_features(gpuxtb_batch_validation_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.batch_validation COMMAND gpuxtb_batch_validation_test)

  add_executable(gpuxtb_plan_schema_test tests/plan_schema_test.cpp)
  target_link_libraries(gpuxtb_plan_schema_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_plan_schema_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.plan_schema COMMAND gpuxtb_plan_schema_test)

  if(GPUXTB_HAS_CUDA)
    add_executable(gpuxtb_cuda_descriptor_validation_test
      tests/cuda_descriptor_validation_test.cu
    )
    target_link_libraries(gpuxtb_cuda_descriptor_validation_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
      CUDA::cudart
    )
    target_include_directories(gpuxtb_cuda_descriptor_validation_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_descriptor_validation_test PRIVATE
      GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_descriptor_validation_test PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
    )
    add_test(NAME gpuxtb.cuda.descriptor_validation
      COMMAND gpuxtb_cuda_descriptor_validation_test
    )

    add_executable(gpuxtb_cuda_gfn2_topology_staging_test
      tests/cuda_gfn2_topology_staging_test.cu
    )
    target_link_libraries(gpuxtb_cuda_gfn2_topology_staging_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
      CUDA::cudart
    )
    target_include_directories(gpuxtb_cuda_gfn2_topology_staging_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_gfn2_topology_staging_test PRIVATE
      GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_gfn2_topology_staging_test PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
    )
    add_test(NAME gpuxtb.cuda.gfn2_topology_staging
      COMMAND gpuxtb_cuda_gfn2_topology_staging_test
    )

    add_executable(gpuxtb_cuda_gfn2_public_result_bridge_test
      tests/cuda_gfn2_public_result_bridge_test.cu
    )
    target_link_libraries(gpuxtb_cuda_gfn2_public_result_bridge_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
      CUDA::cudart
    )
    target_include_directories(gpuxtb_cuda_gfn2_public_result_bridge_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_gfn2_public_result_bridge_test PRIVATE
      GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_gfn2_public_result_bridge_test PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
    )
    add_test(NAME gpuxtb.cuda.gfn2_public_result_bridge
      COMMAND gpuxtb_cuda_gfn2_public_result_bridge_test
    )

    add_executable(gpuxtb_cuda_public_api_test
      tests/cuda_public_api_test.cu
      tests/support/gfn2_scc_test_case.cpp
    )
    target_link_libraries(gpuxtb_cuda_public_api_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_gfn2_cpu
      CUDA::cudart
      Threads::Threads
    )
    target_include_directories(gpuxtb_cuda_public_api_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_public_api_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_public_api_test PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
    )
    add_test(NAME gpuxtb.cuda.public_api COMMAND gpuxtb_cuda_public_api_test)
    if(GPUXTB_CPU_LINALG_RUNTIME)
      gpuxtb_configure_linalg_runtime_test(
        gpuxtb_cuda_public_api_test gpuxtb.cuda.public_api
      )
    endif()

    add_executable(gpuxtb_cuda_public_unrestricted_test
      tests/cuda_public_unrestricted_test.cu
    )
    target_link_libraries(gpuxtb_cuda_public_unrestricted_test PRIVATE
      gpuxtb::gpuxtb
      CUDA::cudart
    )
    target_include_directories(gpuxtb_cuda_public_unrestricted_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_public_unrestricted_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_public_unrestricted_test PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      # Shared-library builds must load the cudart paired with the selected
      # CUDA compiler, even when another toolkit is the system default.
      BUILD_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>"
    )
    add_test(NAME gpuxtb.cuda.public_unrestricted
      COMMAND gpuxtb_cuda_public_unrestricted_test
    )

    add_executable(gpuxtb_cuda_atomics_test tests/cuda_atomics_test.cu)
    target_link_libraries(gpuxtb_cuda_atomics_test PRIVATE CUDA::cudart)
    target_include_directories(gpuxtb_cuda_atomics_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_atomics_test PRIVATE
      GPUXTB_FORCE_FP64_ATOMIC_CAS=1
    )
    set_target_properties(gpuxtb_cuda_atomics_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
    )
    add_test(NAME gpuxtb.cuda.atomics COMMAND gpuxtb_cuda_atomics_test)

    add_executable(gpuxtb_cuda_parameters_test tests/cuda_parameters_test.cu)
    target_link_libraries(gpuxtb_cuda_parameters_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_parameters_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_parameters_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_parameters_test PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
    )
    add_test(NAME gpuxtb.cuda.parameters COMMAND gpuxtb_cuda_parameters_test)

    add_executable(gpuxtb_cuda_repulsion_test tests/cuda_repulsion_test.cu)
    target_link_libraries(gpuxtb_cuda_repulsion_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_repulsion_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_repulsion_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_repulsion_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.repulsion COMMAND gpuxtb_cuda_repulsion_test)

    add_executable(gpuxtb_cuda_external_point_charges_test
      tests/cuda_external_point_charges_test.cu
    )
    target_link_libraries(gpuxtb_cuda_external_point_charges_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_external_point_charges_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_external_point_charges_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_external_point_charges_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.external_point_charges
      COMMAND gpuxtb_cuda_external_point_charges_test
    )

    add_executable(gpuxtb_cuda_classical_force_test
      tests/cuda_classical_force_test.cu
    )
    target_link_libraries(gpuxtb_cuda_classical_force_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_classical_force_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_classical_force_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_classical_force_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.classical_force
      COMMAND gpuxtb_cuda_classical_force_test
    )

    add_executable(gpuxtb_cuda_force_composition_test
      tests/cuda_force_composition_test.cu
    )
    target_link_libraries(gpuxtb_cuda_force_composition_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_force_composition_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_force_composition_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_force_composition_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.force_composition
      COMMAND gpuxtb_cuda_force_composition_test
    )

    add_executable(gpuxtb_cuda_geometry_test tests/cuda_geometry_test.cu)
    target_link_libraries(gpuxtb_cuda_geometry_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_geometry_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_geometry_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_geometry_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.geometry COMMAND gpuxtb_cuda_geometry_test)

    add_executable(gpuxtb_cuda_integrals_test tests/cuda_integrals_test.cu)
    target_link_libraries(gpuxtb_cuda_integrals_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_integrals_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_integrals_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_integrals_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.integrals COMMAND gpuxtb_cuda_integrals_test)

    add_executable(gpuxtb_cuda_integral_force_test tests/cuda_integral_force_test.cu)
    target_link_libraries(gpuxtb_cuda_integral_force_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_integral_force_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_integral_force_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_integral_force_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.integral_force COMMAND gpuxtb_cuda_integral_force_test)

    add_executable(gpuxtb_cuda_h0_force_test tests/cuda_h0_force_test.cu)
    target_link_libraries(gpuxtb_cuda_h0_force_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_h0_force_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_h0_force_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_h0_force_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.h0_force COMMAND gpuxtb_cuda_h0_force_test)

    add_executable(gpuxtb_cuda_hamiltonian_force_test tests/cuda_hamiltonian_force_test.cu)
    target_link_libraries(gpuxtb_cuda_hamiltonian_force_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_hamiltonian_force_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_hamiltonian_force_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_hamiltonian_force_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.hamiltonian_force COMMAND gpuxtb_cuda_hamiltonian_force_test)

    add_executable(gpuxtb_cuda_hamiltonian_test tests/cuda_hamiltonian_test.cu)
    target_link_libraries(gpuxtb_cuda_hamiltonian_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_hamiltonian_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_hamiltonian_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_hamiltonian_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.hamiltonian COMMAND gpuxtb_cuda_hamiltonian_test)

    add_executable(gpuxtb_cuda_mulliken_test tests/cuda_mulliken_test.cu)
    target_link_libraries(gpuxtb_cuda_mulliken_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_mulliken_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_mulliken_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_mulliken_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.mulliken COMMAND gpuxtb_cuda_mulliken_test)

    add_executable(gpuxtb_cuda_spin_test tests/cuda_spin_test.cu)
    target_link_libraries(gpuxtb_cuda_spin_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_spin_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_spin_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_spin_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.spin COMMAND gpuxtb_cuda_spin_test)

    add_executable(gpuxtb_cuda_occupations_test tests/cuda_occupations_test.cu)
    target_link_libraries(gpuxtb_cuda_occupations_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_occupations_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_occupations_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_occupations_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.occupations COMMAND gpuxtb_cuda_occupations_test)

    add_executable(gpuxtb_cuda_density_test tests/cuda_density_test.cu)
    target_link_libraries(gpuxtb_cuda_density_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_density_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_density_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_density_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.density COMMAND gpuxtb_cuda_density_test)

    add_executable(gpuxtb_cuda_electronic_gradient_test
      tests/cuda_electronic_gradient_test.cu
    )
    target_link_libraries(gpuxtb_cuda_electronic_gradient_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_electronic_gradient_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_electronic_gradient_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_electronic_gradient_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.electronic_gradient COMMAND gpuxtb_cuda_electronic_gradient_test)

    add_executable(gpuxtb_cuda_energy_force_execution_test
      tests/cuda_energy_force_execution_test.cu
    )
    target_link_libraries(gpuxtb_cuda_energy_force_execution_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_energy_force_execution_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_energy_force_execution_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_energy_force_execution_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.energy_force_execution
      COMMAND gpuxtb_cuda_energy_force_execution_test
    )

    add_executable(gpuxtb_cuda_gfn2_terminal_execution_test
      tests/cuda_gfn2_terminal_execution_test.cu
    )
    target_link_libraries(gpuxtb_cuda_gfn2_terminal_execution_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_gfn2_terminal_execution_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_gfn2_terminal_execution_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_gfn2_terminal_execution_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.gfn2_terminal_execution
      COMMAND gpuxtb_cuda_gfn2_terminal_execution_test
    )

    add_executable(gpuxtb_cuda_gfn2_runtime_owner_test
      tests/cuda_gfn2_runtime_owner_test.cu
      tests/support/gfn2_scc_test_case.cpp
      # This white-box test exercises the internal context owner directly.
      # Shared-library builds hide that implementation symbol by design, so
      # compile the owner into the test instead of weakening the public ABI.
      src/runtime/backend.cpp
      src/runtime/gfn2_cpu_execution.cpp
    )
    target_link_libraries(gpuxtb_cuda_gfn2_runtime_owner_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_gfn2_runtime_owner_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_gfn2_runtime_owner_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_gfn2_runtime_owner_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.gfn2_runtime_owner
      COMMAND gpuxtb_cuda_gfn2_runtime_owner_test
    )

    add_executable(gpuxtb_cuda_gfn2_runtime_parity_test
      tests/cuda_gfn2_runtime_parity_test.cu
      tests/support/gfn2_scc_test_case.cpp
    )
    target_link_libraries(gpuxtb_cuda_gfn2_runtime_parity_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_gfn2_runtime_parity_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_gfn2_runtime_parity_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_gfn2_runtime_parity_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.gfn2_runtime_parity
      COMMAND gpuxtb_cuda_gfn2_runtime_parity_test
    )
    if(GPUXTB_CPU_LINALG_RUNTIME)
      gpuxtb_configure_linalg_runtime_test(
        gpuxtb_cuda_gfn2_runtime_parity_test gpuxtb.cuda.gfn2_runtime_parity
      )
    endif()

    add_executable(gpuxtb_cuda_gfn2_runtime_graph_test
      tests/cuda_gfn2_runtime_graph_test.cu
      tests/support/gfn2_scc_test_case.cpp
    )
    target_link_libraries(gpuxtb_cuda_gfn2_runtime_graph_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_gfn2_runtime_graph_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_gfn2_runtime_graph_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_gfn2_runtime_graph_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.gfn2_runtime_graph
      COMMAND gpuxtb_cuda_gfn2_runtime_graph_test
    )

    add_executable(gpuxtb_cuda_gfn2_runtime_cache_diagnostics_test
      tests/cuda_gfn2_runtime_cache_diagnostics_test.cu
      tests/support/gfn2_scc_test_case.cpp
    )
    target_link_libraries(gpuxtb_cuda_gfn2_runtime_cache_diagnostics_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_gfn2_runtime_cache_diagnostics_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_gfn2_runtime_cache_diagnostics_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_gfn2_runtime_cache_diagnostics_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.gfn2_runtime_cache_diagnostics
      COMMAND gpuxtb_cuda_gfn2_runtime_cache_diagnostics_test
    )

    add_executable(gpuxtb_cuda_gfn2_preprocessing_test
      tests/cuda_gfn2_preprocessing_test.cu
    )
    target_link_libraries(gpuxtb_cuda_gfn2_preprocessing_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_gfn2_preprocessing_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_gfn2_preprocessing_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_gfn2_preprocessing_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.gfn2_preprocessing
      COMMAND gpuxtb_cuda_gfn2_preprocessing_test
    )

    add_executable(gpuxtb_cuda_eigensolver_test tests/cuda_eigensolver_test.cu)
    target_link_libraries(gpuxtb_cuda_eigensolver_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_eigensolver_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_eigensolver_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_eigensolver_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.eigensolver COMMAND gpuxtb_cuda_eigensolver_test)

    add_executable(gpuxtb_cuda_es3_test tests/cuda_es3_test.cu)
    target_link_libraries(gpuxtb_cuda_es3_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_es3_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_es3_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_es3_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.es3 COMMAND gpuxtb_cuda_es3_test)

    add_executable(gpuxtb_cuda_es2_test tests/cuda_es2_test.cu)
    target_link_libraries(gpuxtb_cuda_es2_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_es2_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_es2_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_es2_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.es2 COMMAND gpuxtb_cuda_es2_test)

    add_executable(gpuxtb_cuda_aes2_test tests/cuda_aes2_test.cu)
    target_link_libraries(gpuxtb_cuda_aes2_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_aes2_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_aes2_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_aes2_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.aes2 COMMAND gpuxtb_cuda_aes2_test)

    add_executable(gpuxtb_cuda_d4_test tests/cuda_d4_test.cu)
    target_link_libraries(gpuxtb_cuda_d4_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_d4_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_d4_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_d4_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.d4 COMMAND gpuxtb_cuda_d4_test)

    add_executable(gpuxtb_cuda_periodic_embedding_test
      tests/cuda_periodic_embedding_test.cu
    )
    target_link_libraries(gpuxtb_cuda_periodic_embedding_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_periodic_embedding_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_periodic_embedding_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_periodic_embedding_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.periodic_embedding
      COMMAND gpuxtb_cuda_periodic_embedding_test
    )

    add_executable(gpuxtb_cuda_scc_test tests/cuda_scc_test.cu)
    target_link_libraries(gpuxtb_cuda_scc_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_scc_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc COMMAND gpuxtb_cuda_scc_test)

    add_executable(gpuxtb_cuda_plan_schema_test tests/cuda_plan_schema_test.cu)
    target_link_libraries(gpuxtb_cuda_plan_schema_test PRIVATE
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_plan_schema_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_plan_schema_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_plan_schema_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.plan_schema COMMAND gpuxtb_cuda_plan_schema_test)

    add_executable(gpuxtb_cuda_scc_bridge_test tests/cuda_scc_bridge_test.cu)
    target_link_libraries(gpuxtb_cuda_scc_bridge_test PRIVATE
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_bridge_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_bridge_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_scc_bridge_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc_bridge COMMAND gpuxtb_cuda_scc_bridge_test)

    add_executable(gpuxtb_cuda_scc_iteration_control_test
      tests/cuda_scc_iteration_control_test.cu
    )
    target_link_libraries(gpuxtb_cuda_scc_iteration_control_test PRIVATE
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_iteration_control_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_iteration_control_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_scc_iteration_control_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc_iteration_control
      COMMAND gpuxtb_cuda_scc_iteration_control_test
    )

    add_executable(gpuxtb_cuda_scc_iteration_binding_test
      tests/cuda_scc_iteration_binding_test.cu
    )
    target_link_libraries(gpuxtb_cuda_scc_iteration_binding_test PRIVATE
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_iteration_binding_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_iteration_binding_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_scc_iteration_binding_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc_iteration_binding
      COMMAND gpuxtb_cuda_scc_iteration_binding_test
    )

    add_executable(gpuxtb_cuda_scc_publication_test
      tests/cuda_scc_publication_test.cu
    )
    target_link_libraries(gpuxtb_cuda_scc_publication_test PRIVATE
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_publication_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_publication_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_scc_publication_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc_publication
      COMMAND gpuxtb_cuda_scc_publication_test
    )

    add_executable(gpuxtb_cuda_scc_iteration_composer_test
      tests/cuda_scc_iteration_composer_test.cu
      tests/support/gfn2_scc_test_case.cpp
    )
    target_link_libraries(gpuxtb_cuda_scc_iteration_composer_test PRIVATE
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_iteration_composer_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_iteration_composer_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_scc_iteration_composer_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc_iteration_composer
      COMMAND gpuxtb_cuda_scc_iteration_composer_test
    )

    add_executable(gpuxtb_cuda_scc_iteration_arena_test
      tests/cuda_scc_iteration_arena_test.cu
    )
    target_link_libraries(gpuxtb_cuda_scc_iteration_arena_test PRIVATE
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_iteration_arena_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_iteration_arena_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_scc_iteration_arena_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc_iteration_arena
      COMMAND gpuxtb_cuda_scc_iteration_arena_test
    )

    add_executable(gpuxtb_cuda_scc_iteration_initialize_test
      tests/cuda_scc_iteration_initialize_test.cu
    )
    target_link_libraries(gpuxtb_cuda_scc_iteration_initialize_test PRIVATE
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_iteration_initialize_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_iteration_initialize_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_scc_iteration_initialize_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc_iteration_initialize
      COMMAND gpuxtb_cuda_scc_iteration_initialize_test
    )

    add_executable(gpuxtb_cuda_scc_iteration_reports_test
      tests/cuda_scc_iteration_reports_test.cu
    )
    target_link_libraries(gpuxtb_cuda_scc_iteration_reports_test PRIVATE
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_iteration_reports_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_iteration_reports_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_scc_iteration_reports_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc_iteration_reports
      COMMAND gpuxtb_cuda_scc_iteration_reports_test
    )

    add_executable(gpuxtb_cuda_scc_setup_topology_test
      tests/cuda_scc_setup_topology_test.cu
    )
    target_link_libraries(gpuxtb_cuda_scc_setup_topology_test PRIVATE
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_setup_topology_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_setup_topology_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_scc_setup_topology_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc_setup_topology
      COMMAND gpuxtb_cuda_scc_setup_topology_test
    )

    add_executable(gpuxtb_cuda_scc_setup_eigensolver_test
      tests/cuda_scc_setup_eigensolver_test.cu
      tests/support/gfn2_scc_test_case.cpp
    )
    target_link_libraries(gpuxtb_cuda_scc_setup_eigensolver_test PRIVATE
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_setup_eigensolver_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_setup_eigensolver_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_scc_setup_eigensolver_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc_setup_eigensolver
      COMMAND gpuxtb_cuda_scc_setup_eigensolver_test
    )

    add_executable(gpuxtb_cuda_scc_setup_inputs_test
      tests/cuda_scc_setup_inputs_test.cu
      tests/support/gfn2_scc_test_case.cpp
    )
    target_link_libraries(gpuxtb_cuda_scc_setup_inputs_test PRIVATE
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_setup_inputs_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_setup_inputs_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_scc_setup_inputs_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc_setup_inputs
      COMMAND gpuxtb_cuda_scc_setup_inputs_test
    )

    add_executable(gpuxtb_cuda_scc_iteration_production_test
      tests/cuda_scc_iteration_production_test.cu
      tests/support/gfn2_scc_test_case.cpp
    )
    target_link_libraries(gpuxtb_cuda_scc_iteration_production_test PRIVATE
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_iteration_production_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_iteration_production_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_scc_iteration_production_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc_iteration_production
      COMMAND gpuxtb_cuda_scc_iteration_production_test
    )

    # Focused #130 benchmark. It reuses the production SCC fixture but runs a
    # one-remaining-iteration state so the conditional WHILE body count and
    # latency can be compared directly with the bounded fallback.
    add_executable(gpuxtb_cuda_scc_loop_benchmark
      tests/cuda_scc_iteration_production_test.cu
      tests/support/gfn2_scc_test_case.cpp
    )
    target_link_libraries(gpuxtb_cuda_scc_loop_benchmark PRIVATE
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_loop_benchmark PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_loop_benchmark PRIVATE
      GPUXTB_HAS_CUDA=1
      GPUXTB_SCC_LOOP_BENCHMARK_ONLY=1
    )
    set_target_properties(gpuxtb_cuda_scc_loop_benchmark PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )

    add_executable(gpuxtb_cuda_scc_iteration_failure_matrix_test
      tests/cuda_scc_iteration_failure_matrix_test.cu
      tests/support/gfn2_scc_test_case.cpp
    )
    target_link_libraries(gpuxtb_cuda_scc_iteration_failure_matrix_test PRIVATE
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_iteration_failure_matrix_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_iteration_failure_matrix_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_scc_iteration_failure_matrix_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc_iteration_failure_matrix
      COMMAND gpuxtb_cuda_scc_iteration_failure_matrix_test
    )

    add_executable(gpuxtb_cuda_scc_electrostatic_components_test
      tests/cuda_scc_electrostatic_components_test.cu
    )
    target_link_libraries(gpuxtb_cuda_scc_electrostatic_components_test PRIVATE
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_electrostatic_components_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_electrostatic_components_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_scc_electrostatic_components_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc_electrostatic_components
      COMMAND gpuxtb_cuda_scc_electrostatic_components_test
    )

    add_executable(gpuxtb_cuda_scc_d4_periodic_test
      tests/cuda_scc_d4_periodic_test.cu
    )
    target_link_libraries(gpuxtb_cuda_scc_d4_periodic_test PRIVATE
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_d4_periodic_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_d4_periodic_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_scc_d4_periodic_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc_d4_periodic
      COMMAND gpuxtb_cuda_scc_d4_periodic_test
    )

    add_executable(gpuxtb_cuda_scc_free_energy_test
      tests/cuda_scc_free_energy_test.cu
    )
    target_link_libraries(gpuxtb_cuda_scc_free_energy_test PRIVATE
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_free_energy_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_free_energy_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_scc_free_energy_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc_free_energy COMMAND gpuxtb_cuda_scc_free_energy_test)

    add_executable(gpuxtb_cuda_total_energy_test tests/cuda_total_energy_test.cu)
    target_link_libraries(gpuxtb_cuda_total_energy_test PRIVATE
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_total_energy_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_total_energy_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_total_energy_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.total_energy COMMAND gpuxtb_cuda_total_energy_test)

    add_executable(gpuxtb_cuda_post_scc_potential_test
      tests/cuda_post_scc_potential_test.cu
    )
    target_link_libraries(gpuxtb_cuda_post_scc_potential_test PRIVATE
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_post_scc_potential_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_post_scc_potential_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_post_scc_potential_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.post_scc_potential
      COMMAND gpuxtb_cuda_post_scc_potential_test
    )

    add_executable(gpuxtb_cuda_scc_energy_test tests/cuda_scc_energy_test.cu)
    target_link_libraries(gpuxtb_cuda_scc_energy_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_energy_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_energy_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_scc_energy_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc_energy COMMAND gpuxtb_cuda_scc_energy_test)

    add_executable(gpuxtb_cuda_scc_classical_energy_test
      tests/cuda_scc_classical_energy_test.cu
    )
    target_link_libraries(gpuxtb_cuda_scc_classical_energy_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_classical_energy_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_classical_energy_test
      PRIVATE GPUXTB_HAS_CUDA=1
    )
    set_target_properties(gpuxtb_cuda_scc_classical_energy_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc_classical_energy
      COMMAND gpuxtb_cuda_scc_classical_energy_test
    )

    add_executable(gpuxtb_cuda_scc_potential_test tests/cuda_scc_potential_test.cu)
    target_link_libraries(gpuxtb_cuda_scc_potential_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_potential_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_potential_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_scc_potential_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc_potential COMMAND gpuxtb_cuda_scc_potential_test)

    add_executable(gpuxtb_cuda_scc_mixer_test tests/cuda_scc_mixer_test.cu)
    target_link_libraries(gpuxtb_cuda_scc_mixer_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
      gpuxtb_gfn2_cpu
    )
    target_include_directories(gpuxtb_cuda_scc_mixer_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(gpuxtb_cuda_scc_mixer_test PRIVATE GPUXTB_HAS_CUDA=1)
    set_target_properties(gpuxtb_cuda_scc_mixer_test PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME gpuxtb.cuda.scc_mixer COMMAND gpuxtb_cuda_scc_mixer_test)
  endif()

  add_executable(gpuxtb_repulsion_test tests/repulsion_test.cpp)
  target_link_libraries(gpuxtb_repulsion_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_repulsion_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.repulsion COMMAND gpuxtb_repulsion_test)

  add_executable(gpuxtb_coordination_test tests/coordination_test.cpp)
  target_link_libraries(gpuxtb_coordination_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_coordination_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.coordination COMMAND gpuxtb_coordination_test)

  add_executable(gpuxtb_d4_test tests/d4_test.cpp)
  target_link_libraries(gpuxtb_d4_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_d4_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.d4 COMMAND gpuxtb_d4_test)

  add_executable(gpuxtb_basis_test tests/basis_test.cpp)
  target_link_libraries(gpuxtb_basis_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_basis_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.basis COMMAND gpuxtb_basis_test)

  add_executable(gpuxtb_integrals_test tests/integrals_test.cpp)
  target_link_libraries(gpuxtb_integrals_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_integrals_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.integrals COMMAND gpuxtb_integrals_test)

  add_executable(gpuxtb_multipole_integrals_test tests/multipole_integrals_test.cpp)
  target_link_libraries(gpuxtb_multipole_integrals_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_multipole_integrals_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.multipole_integrals COMMAND gpuxtb_multipole_integrals_test)

  add_executable(gpuxtb_h0_test tests/h0_test.cpp)
  target_link_libraries(gpuxtb_h0_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_h0_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.h0 COMMAND gpuxtb_h0_test)

  add_executable(gpuxtb_force_test tests/force_test.cpp)
  target_link_libraries(gpuxtb_force_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_force_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.force COMMAND gpuxtb_force_test)

  add_executable(gpuxtb_external_point_charges_test tests/external_point_charges_test.cpp)
  target_link_libraries(gpuxtb_external_point_charges_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_external_point_charges_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.external_point_charges COMMAND gpuxtb_external_point_charges_test)

  add_executable(gpuxtb_periodic_embedding_test tests/periodic_embedding_test.cpp)
  target_link_libraries(gpuxtb_periodic_embedding_test PRIVATE gpuxtb_gfn2_cpu Threads::Threads)
  target_compile_features(gpuxtb_periodic_embedding_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.periodic_embedding COMMAND gpuxtb_periodic_embedding_test)

  add_executable(gpuxtb_es3_test tests/es3_test.cpp)
  target_link_libraries(gpuxtb_es3_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_es3_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.es3 COMMAND gpuxtb_es3_test)

  add_executable(gpuxtb_spin_test tests/spin_test.cpp)
  target_link_libraries(gpuxtb_spin_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_spin_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.spin COMMAND gpuxtb_spin_test)

  add_executable(gpuxtb_es2_test tests/es2_test.cpp)
  target_link_libraries(gpuxtb_es2_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_es2_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.es2 COMMAND gpuxtb_es2_test)

  add_executable(gpuxtb_aes2_test tests/aes2_test.cpp)
  target_link_libraries(gpuxtb_aes2_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_aes2_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.aes2 COMMAND gpuxtb_aes2_test)

  add_executable(gpuxtb_wavefunction_test tests/wavefunction_test.cpp)
  target_link_libraries(gpuxtb_wavefunction_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_wavefunction_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.wavefunction COMMAND gpuxtb_wavefunction_test)

  add_executable(gpuxtb_mulliken_test tests/mulliken_test.cpp)
  target_link_libraries(gpuxtb_mulliken_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_mulliken_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.mulliken COMMAND gpuxtb_mulliken_test)

  add_executable(gpuxtb_scc_mixer_test tests/scc_mixer_test.cpp)
  target_link_libraries(gpuxtb_scc_mixer_test PRIVATE gpuxtb_gfn2_cpu Threads::Threads)
  target_compile_features(gpuxtb_scc_mixer_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.scc_mixer COMMAND gpuxtb_scc_mixer_test)

  add_executable(gpuxtb_scc_driver_test tests/scc_driver_test.cpp)
  target_link_libraries(gpuxtb_scc_driver_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_scc_driver_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.scc_driver COMMAND gpuxtb_scc_driver_test)

  if(GPUXTB_CPU_LINALG_RUNTIME)
    add_executable(gpuxtb_eigensolver_test tests/eigensolver_test.cpp)
    target_link_libraries(gpuxtb_eigensolver_test PRIVATE
      gpuxtb_gfn2_cpu
      ${GPUXTB_CPU_LINALG_RUNTIME}
      Threads::Threads
    )
    target_compile_features(gpuxtb_eigensolver_test PRIVATE cxx_std_17)
    if(GPUXTB_CPU_LINALG_SYMBOL_PREFIX STREQUAL "scipy_")
      # The test spies call through to the configured provider directly; use
      # scipy-openblas32's collision-avoiding ABI names when applicable.
      target_compile_definitions(
        gpuxtb_eigensolver_test PRIVATE GPUXTB_TEST_SCIPY_PREFIXED_BLAS=1
      )
    endif()
    add_test(NAME gpuxtb.gfn2.eigensolver COMMAND gpuxtb_eigensolver_test)
    gpuxtb_configure_linalg_runtime_test(
      gpuxtb_eigensolver_test gpuxtb.gfn2.eigensolver
    )
  else()
    message(STATUS
      "gpuxtb eigensolver test: disabled (set GPUXTB_MKL_RT_LIBRARY or provide a system BLAS to enable)"
    )
  endif()

  add_executable(gpuxtb_parameters_header_test tests/parameters/gfn2_header_test.cpp)
  target_include_directories(gpuxtb_parameters_header_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_features(gpuxtb_parameters_header_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.parameters.header COMMAND gpuxtb_parameters_header_test)

  add_test(
    NAME gpuxtb.parameters.generated
    COMMAND ${Python3_EXECUTABLE} tools/parameters/generate_gfn2.py --check
  )
  add_test(
    NAME gpuxtb.parameters.python
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
            -s tests/parameters -p "test_*.py" -v
  )
  add_test(
    NAME gpuxtb.conformance.manifest
    COMMAND ${Python3_EXECUTABLE} tools/conformance/gpuxtb_conformance.py check
  )
  add_test(
    NAME gpuxtb.conformance.python
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
            -s tests/conformance -p "test_*.py" -v
  )
  if(BUILD_SHARED_LIBS AND GPUXTB_CPU_LINALG_RUNTIME)
    add_test(
      NAME gpuxtb.conformance.public_cpu
      COMMAND ${Python3_EXECUTABLE} tools/conformance/gpuxtb_public_api.py
              --library $<TARGET_FILE:gpuxtb> --backend cpu --memory-mode host
              --actual-dir ${CMAKE_CURRENT_BINARY_DIR}/conformance/public-api
    )
    set_tests_properties(gpuxtb.conformance.public_cpu PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      ENVIRONMENT "MKL_INTERFACE_LAYER=LP64;MKL_THREADING_LAYER=SEQUENTIAL"
      ENVIRONMENT_MODIFICATION
        "${GPUXTB_RUNTIME_LIBRARY_PATH_VARIABLE}=path_list_prepend:${GPUXTB_CPU_LINALG_DIRECTORY}"
      LABELS "conformance;public-api"
    )
    add_test(
      NAME gpuxtb.conformance.invariants_cpu
      COMMAND ${Python3_EXECUTABLE} tools/conformance/gpuxtb_invariants.py
              --library $<TARGET_FILE:gpuxtb> --backend cpu --memory-mode host
    )
    set_tests_properties(gpuxtb.conformance.invariants_cpu PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      ENVIRONMENT "MKL_INTERFACE_LAYER=LP64;MKL_THREADING_LAYER=SEQUENTIAL"
      ENVIRONMENT_MODIFICATION
        "${GPUXTB_RUNTIME_LIBRARY_PATH_VARIABLE}=path_list_prepend:${GPUXTB_CPU_LINALG_DIRECTORY}"
      LABELS "conformance;public-api"
    )
    if(GPUXTB_HAS_CUDA)
      add_test(
        NAME gpuxtb.conformance.public_cuda
        COMMAND ${Python3_EXECUTABLE} tools/conformance/gpuxtb_public_api.py
                --library $<TARGET_FILE:gpuxtb> --backend cuda --memory-mode host
                --actual-dir ${CMAKE_CURRENT_BINARY_DIR}/conformance/public-api
                --skip-backend-unavailable
      )
      set_tests_properties(gpuxtb.conformance.public_cuda PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        ENVIRONMENT "MKL_INTERFACE_LAYER=LP64;MKL_THREADING_LAYER=SEQUENTIAL"
        ENVIRONMENT_MODIFICATION
          "${GPUXTB_RUNTIME_LIBRARY_PATH_VARIABLE}=path_list_prepend:${GPUXTB_CPU_LINALG_DIRECTORY}"
        LABELS "conformance;public-api"
        SKIP_RETURN_CODE 77
      )
      add_test(
        NAME gpuxtb.conformance.invariants_cuda
        COMMAND ${Python3_EXECUTABLE} tools/conformance/gpuxtb_invariants.py
                --library $<TARGET_FILE:gpuxtb> --backend cuda --memory-mode host
                --skip-backend-unavailable
      )
      set_tests_properties(gpuxtb.conformance.invariants_cuda PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        ENVIRONMENT "MKL_INTERFACE_LAYER=LP64;MKL_THREADING_LAYER=SEQUENTIAL"
        ENVIRONMENT_MODIFICATION
          "${GPUXTB_RUNTIME_LIBRARY_PATH_VARIABLE}=path_list_prepend:${GPUXTB_CPU_LINALG_DIRECTORY}"
        LABELS "conformance;public-api"
        SKIP_RETURN_CODE 77
      )
      foreach(memory_mode IN ITEMS device mixed)
        add_test(
          NAME gpuxtb.conformance.public_cuda_${memory_mode}
          COMMAND ${Python3_EXECUTABLE} tools/conformance/gpuxtb_public_api.py
                  --library $<TARGET_FILE:gpuxtb> --backend cuda
                  --memory-mode ${memory_mode}
                  --actual-dir ${CMAKE_CURRENT_BINARY_DIR}/conformance/public-api
                  --skip-backend-unavailable
        )
        set_tests_properties(gpuxtb.conformance.public_cuda_${memory_mode} PROPERTIES
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          ENVIRONMENT "MKL_INTERFACE_LAYER=LP64;MKL_THREADING_LAYER=SEQUENTIAL"
          ENVIRONMENT_MODIFICATION
            "${GPUXTB_RUNTIME_LIBRARY_PATH_VARIABLE}=path_list_prepend:${GPUXTB_CPU_LINALG_DIRECTORY}"
          LABELS "conformance;public-api"
          SKIP_RETURN_CODE 77
        )
        add_test(
          NAME gpuxtb.conformance.invariants_cuda_${memory_mode}
          COMMAND ${Python3_EXECUTABLE} tools/conformance/gpuxtb_invariants.py
                  --library $<TARGET_FILE:gpuxtb> --backend cuda
                  --memory-mode ${memory_mode}
                  --skip-backend-unavailable
        )
        set_tests_properties(gpuxtb.conformance.invariants_cuda_${memory_mode} PROPERTIES
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          ENVIRONMENT "MKL_INTERFACE_LAYER=LP64;MKL_THREADING_LAYER=SEQUENTIAL"
          ENVIRONMENT_MODIFICATION
            "${GPUXTB_RUNTIME_LIBRARY_PATH_VARIABLE}=path_list_prepend:${GPUXTB_CPU_LINALG_DIRECTORY}"
          LABELS "conformance;public-api"
          SKIP_RETURN_CODE 77
        )
      endforeach()
    endif()
  endif()
  add_test(
    NAME gpuxtb.licensing
    COMMAND ${Python3_EXECUTABLE} tools/licensing/check_licenses.py
            --source-root ${CMAKE_CURRENT_SOURCE_DIR}
  )
  add_test(
    NAME gpuxtb.licensing_unit
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
            -s tests/licensing -p "test_*.py" -v
  )
  add_test(
    NAME gpuxtb.oracle.tblite_observer_patch
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
            -s tests/oracle -p "test_*.py" -v
  )
  set_tests_properties(
    gpuxtb.parameters.generated
    gpuxtb.parameters.python
    gpuxtb.licensing
    gpuxtb.licensing_unit
    gpuxtb.conformance.manifest
    gpuxtb.conformance.python
    gpuxtb.oracle.tblite_observer_patch
    PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()

install(TARGETS gpuxtb EXPORT gpuxtbTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# Legal material is part of the installed SDK contract. Keep project terms,
# third-party notices, and machine-readable provenance together so native
# consumers receive the same audit trail as source and Python distributions.
install(FILES LICENSE THIRD_PARTY_NOTICES.md
  DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/gpuxtb
)
install(DIRECTORY LICENSES/
  DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/gpuxtb/third-party
)
install(FILES
  data/parameters/manifest.json
  data/parameters/spin_manifest.json
  data/parameters/d4_manifest.json
  data/parameters/mctc_manifest.json
  cmake/3rdparty/implib_manifest.json
  DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/gpuxtb/provenance
)
install(FILES data/parameters/d4.NOTICE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/gpuxtb/third-party/d4
)
install(DIRECTORY data/parameters/licenses/
  DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/gpuxtb/third-party/d4
)
install(EXPORT gpuxtbTargets
  FILE gpuxtbTargets.cmake
  NAMESPACE gpuxtb::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpuxtb
)

# A CUDA-enabled build records the compatible major version of the CUDA
# runtime it dlopens so the installed CMake config can warn consumers that the
# runtime they have may be incompatible (the library itself no longer links
# against the toolkit; see src/runtime/cuda_dlopen.c).
if(GPUXTB_HAS_CUDA)
  set(GPUXTB_CUDA_TOOLKIT_MAJOR ${CUDAToolkit_VERSION_MAJOR})
  set(GPUXTB_CUDA_TOOLKIT_MINOR ${CUDAToolkit_VERSION_MINOR})
else()
  set(GPUXTB_CUDA_TOOLKIT_MAJOR 0)
  set(GPUXTB_CUDA_TOOLKIT_MINOR 0)
endif()
configure_package_config_file(
  cmake/gpuxtbConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/gpuxtbConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpuxtb
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/gpuxtbConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/gpuxtbConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/gpuxtbConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpuxtb
)

message(STATUS "gpuxtb CUDA backend: ${GPUXTB_HAS_CUDA}")
