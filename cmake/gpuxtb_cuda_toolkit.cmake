include_guard(GLOBAL)

# Resolve the compiler-owned CUDA metadata separately from FindCUDAToolkit.
# CMake's package module requires libcudart even though gpuxtb only uses it to
# discover optional provider shared libraries for implib generation.
function(gpuxtb_resolve_cuda_toolkit_metadata
    output_version output_major output_minor output_include_dirs)
  if(CUDAToolkit_FOUND)
    set(_gpuxtb_cuda_version "${CUDAToolkit_VERSION}")
    set(_gpuxtb_cuda_major "${CUDAToolkit_VERSION_MAJOR}")
    set(_gpuxtb_cuda_minor "${CUDAToolkit_VERSION_MINOR}")
    set(_gpuxtb_cuda_include_dirs "${CUDAToolkit_INCLUDE_DIRS}")
  else()
    set(_gpuxtb_cuda_version "${CMAKE_CUDA_COMPILER_TOOLKIT_VERSION}")
    if(NOT _gpuxtb_cuda_version)
      set(_gpuxtb_cuda_version "${CMAKE_CUDA_COMPILER_VERSION}")
    endif()
    set(_gpuxtb_cuda_include_dirs "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
  endif()

  if(NOT _gpuxtb_cuda_version)
    message(FATAL_ERROR
      "the CUDA compiler did not report its toolkit version")
  endif()
  if(NOT _gpuxtb_cuda_include_dirs)
    message(FATAL_ERROR
      "the CUDA compiler did not report its toolkit include directories")
  endif()
  if(NOT _gpuxtb_cuda_major OR NOT _gpuxtb_cuda_minor)
    if(NOT _gpuxtb_cuda_version MATCHES "^([0-9]+)\\.([0-9]+)")
      message(FATAL_ERROR
        "cannot parse CUDA toolkit version '${_gpuxtb_cuda_version}'")
    endif()
    set(_gpuxtb_cuda_major "${CMAKE_MATCH_1}")
    set(_gpuxtb_cuda_minor "${CMAKE_MATCH_2}")
  endif()

  set(${output_version} "${_gpuxtb_cuda_version}" PARENT_SCOPE)
  set(${output_major} "${_gpuxtb_cuda_major}" PARENT_SCOPE)
  set(${output_minor} "${_gpuxtb_cuda_minor}" PARENT_SCOPE)
  set(${output_include_dirs} "${_gpuxtb_cuda_include_dirs}" PARENT_SCOPE)
endfunction()

# The self-declared host API still consumes the CUDA runtime's public data
# types. Verify the exact header-only contract before compiling the backend.
function(gpuxtb_validate_cuda_runtime_headers)
  foreach(_gpuxtb_cuda_header IN ITEMS cuda_runtime_api.h library_types.h)
    set(_gpuxtb_cuda_header_found OFF)
    foreach(_gpuxtb_cuda_include_dir IN LISTS ARGN)
      if(EXISTS "${_gpuxtb_cuda_include_dir}/${_gpuxtb_cuda_header}")
        set(_gpuxtb_cuda_header_found ON)
        break()
      endif()
    endforeach()
    if(NOT _gpuxtb_cuda_header_found)
      message(FATAL_ERROR
        "the CUDA compiler include set does not contain ${_gpuxtb_cuda_header}")
    endif()
  endforeach()
endfunction()

# CUDA provider products do not all use the toolkit major as their ELF major.
# Keep an audited table instead of guessing future provider SONAMEs.
function(gpuxtb_cuda_default_provider_sonames
    toolkit_major output_cudart output_cublas output_cusolver output_driver)
  if(toolkit_major EQUAL 12)
    set(_gpuxtb_cudart_soname "libcudart.so.12")
    set(_gpuxtb_cublas_soname "libcublas.so.12")
    set(_gpuxtb_cusolver_soname "libcusolver.so.11")
  elseif(toolkit_major EQUAL 13)
    set(_gpuxtb_cudart_soname "libcudart.so.13")
    set(_gpuxtb_cublas_soname "libcublas.so.13")
    set(_gpuxtb_cusolver_soname "libcusolver.so.12")
  else()
    set(_gpuxtb_cudart_soname "")
    set(_gpuxtb_cublas_soname "")
    set(_gpuxtb_cusolver_soname "")
  endif()

  set(${output_cudart} "${_gpuxtb_cudart_soname}" PARENT_SCOPE)
  set(${output_cublas} "${_gpuxtb_cublas_soname}" PARENT_SCOPE)
  set(${output_cusolver} "${_gpuxtb_cusolver_soname}" PARENT_SCOPE)
  set(${output_driver} "libcuda.so.1" PARENT_SCOPE)
endfunction()

# Cache an audited SONAME default without freezing it across toolkit changes.
# The internal companion records whether gpuxtb created the public entry. An
# entry that predates the marker may be an explicit command-line pin, even when
# its text equals the default, so it is conservatively kept user-owned.
function(gpuxtb_cache_cuda_provider_soname variable default_value description)
  set(_gpuxtb_auto_default_variable "${variable}_AUTO_DEFAULT")
  set(_gpuxtb_explicit_marker "__GPUXTB_EXPLICIT_OVERRIDE__")
  if(NOT DEFINED CACHE{${variable}})
    set(${variable} "${default_value}" CACHE STRING "${description}")
    set(${_gpuxtb_auto_default_variable} "${default_value}" CACHE INTERNAL
      "Last automatically selected value for ${variable}" FORCE)
  elseif(NOT DEFINED CACHE{${_gpuxtb_auto_default_variable}})
    set(${_gpuxtb_auto_default_variable} "${_gpuxtb_explicit_marker}"
      CACHE INTERNAL "Origin marker for ${variable}" FORCE)
  else()
    get_property(_gpuxtb_previous_auto_default
      CACHE "${_gpuxtb_auto_default_variable}" PROPERTY VALUE)
    if("${_gpuxtb_previous_auto_default}" STREQUAL "${_gpuxtb_explicit_marker}")
      return()
    elseif("${${variable}}" STREQUAL "${_gpuxtb_previous_auto_default}")
      set(${variable} "${default_value}" CACHE STRING "${description}" FORCE)
      set(${_gpuxtb_auto_default_variable} "${default_value}" CACHE INTERNAL
        "Last automatically selected value for ${variable}" FORCE)
    else()
      set(${_gpuxtb_auto_default_variable} "${_gpuxtb_explicit_marker}"
        CACHE INTERNAL "Origin marker for ${variable}" FORCE)
    endif()
  endif()
endfunction()
