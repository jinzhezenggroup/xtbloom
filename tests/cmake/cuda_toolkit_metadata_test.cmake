cmake_minimum_required(VERSION 3.24)

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/gpuxtb_cuda_toolkit.cmake")

function(gpuxtb_assert_equal actual expected description)
  if(NOT "${actual}" STREQUAL "${expected}")
    message(FATAL_ERROR
      "${description}: expected '${expected}', got '${actual}'")
  endif()
endfunction()

# FindCUDAToolkit may fail solely because libcudart is absent. The enabled
# compiler still supplies the version and include metadata needed by gpuxtb.
set(CUDAToolkit_FOUND OFF)
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "13.0.48")
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/cuda/include;/cuda/extra")
gpuxtb_resolve_cuda_toolkit_metadata(
  _version _major _minor _include_dirs)
gpuxtb_assert_equal("${_version}" "13.0.48" "compiler toolkit version")
gpuxtb_assert_equal("${_major}" "13" "compiler toolkit major")
gpuxtb_assert_equal("${_minor}" "0" "compiler toolkit minor")
gpuxtb_assert_equal(
  "${_include_dirs}" "/cuda/include;/cuda/extra" "compiler include set")

# CMake 3.24 does not need to expose the more specific toolkit-version
# variable because the CUDA compiler version carries the same major/minor.
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "")
set(CMAKE_CUDA_COMPILER_VERSION "12.4.131")
gpuxtb_resolve_cuda_toolkit_metadata(
  _version _major _minor _include_dirs)
gpuxtb_assert_equal("${_version}" "12.4.131" "compiler version fallback")
gpuxtb_assert_equal("${_major}" "12" "fallback toolkit major")
gpuxtb_assert_equal("${_minor}" "4" "fallback toolkit minor")

set(CUDAToolkit_FOUND ON)
set(CUDAToolkit_VERSION "12.9.1")
set(CUDAToolkit_VERSION_MAJOR "12")
set(CUDAToolkit_VERSION_MINOR "9")
set(CUDAToolkit_INCLUDE_DIRS "/provider/include")
gpuxtb_resolve_cuda_toolkit_metadata(
  _version _major _minor _include_dirs)
gpuxtb_assert_equal("${_version}" "12.9.1" "package toolkit version")
gpuxtb_assert_equal("${_major}" "12" "package toolkit major")
gpuxtb_assert_equal("${_minor}" "9" "package toolkit minor")
gpuxtb_assert_equal(
  "${_include_dirs}" "/provider/include" "package include set")

gpuxtb_cuda_default_provider_sonames(
  12 _cudart _cublas _cusolver _driver)
gpuxtb_assert_equal("${_cudart}" "libcudart.so.12" "CUDA 12 cudart SONAME")
gpuxtb_assert_equal("${_cublas}" "libcublas.so.12" "CUDA 12 cuBLAS SONAME")
gpuxtb_assert_equal("${_cusolver}" "libcusolver.so.11" "CUDA 12 cuSOLVER SONAME")
gpuxtb_assert_equal("${_driver}" "libcuda.so.1" "CUDA driver SONAME")

gpuxtb_cuda_default_provider_sonames(
  13 _cudart _cublas _cusolver _driver)
gpuxtb_assert_equal("${_cudart}" "libcudart.so.13" "CUDA 13 cudart SONAME")
gpuxtb_assert_equal("${_cublas}" "libcublas.so.13" "CUDA 13 cuBLAS SONAME")
gpuxtb_assert_equal("${_cusolver}" "libcusolver.so.12" "CUDA 13 cuSOLVER SONAME")
gpuxtb_assert_equal("${_driver}" "libcuda.so.1" "CUDA driver SONAME")

gpuxtb_cuda_default_provider_sonames(
  14 _cudart _cublas _cusolver _driver)
gpuxtb_assert_equal("${_cudart}" "" "unknown cudart SONAME")
gpuxtb_assert_equal("${_cublas}" "" "unknown cuBLAS SONAME")
gpuxtb_assert_equal("${_cusolver}" "" "unknown cuSOLVER SONAME")
gpuxtb_assert_equal("${_driver}" "libcuda.so.1" "unknown-major driver SONAME")

# Generated cache values advance with the toolkit major.
unset(GPUXTB_TEST_CUDART_SONAME CACHE)
unset(GPUXTB_TEST_CUDART_SONAME_AUTO_DEFAULT CACHE)
gpuxtb_cache_cuda_provider_soname(
  GPUXTB_TEST_CUDART_SONAME "libcudart.so.12" "test SONAME")
gpuxtb_assert_equal(
  "${GPUXTB_TEST_CUDART_SONAME}" "libcudart.so.12" "initial cached default")
gpuxtb_cache_cuda_provider_soname(
  GPUXTB_TEST_CUDART_SONAME "libcudart.so.13" "test SONAME")
gpuxtb_assert_equal(
  "${GPUXTB_TEST_CUDART_SONAME}" "libcudart.so.13" "refreshed cached default")

# A pre-defined cache entry is an explicit override even when its text equals
# the current generated default, so a later toolkit change must not rewrite it.
unset(GPUXTB_TEST_CUDART_SONAME CACHE)
unset(GPUXTB_TEST_CUDART_SONAME_AUTO_DEFAULT CACHE)
set(GPUXTB_TEST_CUDART_SONAME "libcudart.so.12" CACHE STRING
  "test SONAME")
gpuxtb_cache_cuda_provider_soname(
  GPUXTB_TEST_CUDART_SONAME "libcudart.so.12" "test SONAME")
gpuxtb_cache_cuda_provider_soname(
  GPUXTB_TEST_CUDART_SONAME "libcudart.so.13" "test SONAME")
gpuxtb_assert_equal(
  "${GPUXTB_TEST_CUDART_SONAME}" "libcudart.so.12"
  "equal-to-default explicit override")

# Changing a previously generated public value also transfers ownership to the
# user and remains authoritative across subsequent toolkit changes.
unset(GPUXTB_TEST_CUDART_SONAME CACHE)
unset(GPUXTB_TEST_CUDART_SONAME_AUTO_DEFAULT CACHE)
gpuxtb_cache_cuda_provider_soname(
  GPUXTB_TEST_CUDART_SONAME "libcudart.so.12" "test SONAME")
set(GPUXTB_TEST_CUDART_SONAME "libcudart-custom.so" CACHE STRING
  "test SONAME" FORCE)
gpuxtb_cache_cuda_provider_soname(
  GPUXTB_TEST_CUDART_SONAME "libcudart.so.14" "test SONAME")
gpuxtb_assert_equal(
  "${GPUXTB_TEST_CUDART_SONAME}" "libcudart-custom.so" "explicit override")
