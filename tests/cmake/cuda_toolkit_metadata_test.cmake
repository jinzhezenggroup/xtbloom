cmake_minimum_required(VERSION 3.24)

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/xtbloom_cuda_toolkit.cmake")

function(xtbloom_assert_equal actual expected description)
  if(NOT "${actual}" STREQUAL "${expected}")
    message(FATAL_ERROR
      "${description}: expected '${expected}', got '${actual}'")
  endif()
endfunction()

# FindCUDAToolkit may fail solely because libcudart is absent. The enabled
# compiler still supplies the version and include metadata needed by xtbloom.
set(CUDAToolkit_FOUND OFF)
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "13.0.48")
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/cuda/include;/cuda/extra")
xtbloom_resolve_cuda_toolkit_metadata(
  _version _major _minor _include_dirs)
xtbloom_assert_equal("${_version}" "13.0.48" "compiler toolkit version")
xtbloom_assert_equal("${_major}" "13" "compiler toolkit major")
xtbloom_assert_equal("${_minor}" "0" "compiler toolkit minor")
xtbloom_assert_equal(
  "${_include_dirs}" "/cuda/include;/cuda/extra" "compiler include set")

# CMake 3.24 does not need to expose the more specific toolkit-version
# variable because the CUDA compiler version carries the same major/minor.
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "")
set(CMAKE_CUDA_COMPILER_VERSION "12.4.131")
xtbloom_resolve_cuda_toolkit_metadata(
  _version _major _minor _include_dirs)
xtbloom_assert_equal("${_version}" "12.4.131" "compiler version fallback")
xtbloom_assert_equal("${_major}" "12" "fallback toolkit major")
xtbloom_assert_equal("${_minor}" "4" "fallback toolkit minor")

set(CUDAToolkit_FOUND ON)
set(CUDAToolkit_VERSION "12.9.1")
set(CUDAToolkit_VERSION_MAJOR "12")
set(CUDAToolkit_VERSION_MINOR "9")
set(CUDAToolkit_INCLUDE_DIRS "/provider/include")
xtbloom_resolve_cuda_toolkit_metadata(
  _version _major _minor _include_dirs)
xtbloom_assert_equal("${_version}" "12.9.1" "package toolkit version")
xtbloom_assert_equal("${_major}" "12" "package toolkit major")
xtbloom_assert_equal("${_minor}" "9" "package toolkit minor")
xtbloom_assert_equal(
  "${_include_dirs}" "/provider/include" "package include set")

xtbloom_cuda_default_provider_sonames(
  12 _cudart _cublas _cusolver _driver)
xtbloom_assert_equal("${_cudart}" "libcudart.so.12" "CUDA 12 cudart SONAME")
xtbloom_assert_equal("${_cublas}" "libcublas.so.12" "CUDA 12 cuBLAS SONAME")
xtbloom_assert_equal("${_cusolver}" "libcusolver.so.11" "CUDA 12 cuSOLVER SONAME")
xtbloom_assert_equal("${_driver}" "libcuda.so.1" "CUDA driver SONAME")

xtbloom_cuda_default_provider_sonames(
  13 _cudart _cublas _cusolver _driver)
xtbloom_assert_equal("${_cudart}" "libcudart.so.13" "CUDA 13 cudart SONAME")
xtbloom_assert_equal("${_cublas}" "libcublas.so.13" "CUDA 13 cuBLAS SONAME")
xtbloom_assert_equal("${_cusolver}" "libcusolver.so.12" "CUDA 13 cuSOLVER SONAME")
xtbloom_assert_equal("${_driver}" "libcuda.so.1" "CUDA driver SONAME")

xtbloom_cuda_default_provider_sonames(
  14 _cudart _cublas _cusolver _driver)
xtbloom_assert_equal("${_cudart}" "" "unknown cudart SONAME")
xtbloom_assert_equal("${_cublas}" "" "unknown cuBLAS SONAME")
xtbloom_assert_equal("${_cusolver}" "" "unknown cuSOLVER SONAME")
xtbloom_assert_equal("${_driver}" "libcuda.so.1" "unknown-major driver SONAME")

# Generated cache values advance with the toolkit major.
unset(XTBLOOM_TEST_CUDART_SONAME CACHE)
unset(XTBLOOM_TEST_CUDART_SONAME_AUTO_DEFAULT CACHE)
xtbloom_cache_cuda_provider_soname(
  XTBLOOM_TEST_CUDART_SONAME "libcudart.so.12" "test SONAME")
xtbloom_assert_equal(
  "${XTBLOOM_TEST_CUDART_SONAME}" "libcudart.so.12" "initial cached default")
xtbloom_cache_cuda_provider_soname(
  XTBLOOM_TEST_CUDART_SONAME "libcudart.so.13" "test SONAME")
xtbloom_assert_equal(
  "${XTBLOOM_TEST_CUDART_SONAME}" "libcudart.so.13" "refreshed cached default")

# A pre-defined cache entry is an explicit override even when its text equals
# the current generated default, so a later toolkit change must not rewrite it.
unset(XTBLOOM_TEST_CUDART_SONAME CACHE)
unset(XTBLOOM_TEST_CUDART_SONAME_AUTO_DEFAULT CACHE)
set(XTBLOOM_TEST_CUDART_SONAME "libcudart.so.12" CACHE STRING
  "test SONAME")
xtbloom_cache_cuda_provider_soname(
  XTBLOOM_TEST_CUDART_SONAME "libcudart.so.12" "test SONAME")
xtbloom_cache_cuda_provider_soname(
  XTBLOOM_TEST_CUDART_SONAME "libcudart.so.13" "test SONAME")
xtbloom_assert_equal(
  "${XTBLOOM_TEST_CUDART_SONAME}" "libcudart.so.12"
  "equal-to-default explicit override")

# Changing a previously generated public value also transfers ownership to the
# user and remains authoritative across subsequent toolkit changes.
unset(XTBLOOM_TEST_CUDART_SONAME CACHE)
unset(XTBLOOM_TEST_CUDART_SONAME_AUTO_DEFAULT CACHE)
xtbloom_cache_cuda_provider_soname(
  XTBLOOM_TEST_CUDART_SONAME "libcudart.so.12" "test SONAME")
set(XTBLOOM_TEST_CUDART_SONAME "libcudart-custom.so" CACHE STRING
  "test SONAME" FORCE)
xtbloom_cache_cuda_provider_soname(
  XTBLOOM_TEST_CUDART_SONAME "libcudart.so.14" "test SONAME")
xtbloom_assert_equal(
  "${XTBLOOM_TEST_CUDART_SONAME}" "libcudart-custom.so" "explicit override")
