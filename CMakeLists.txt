cmake_minimum_required(VERSION 3.24)

project(
  gpuxtb
  VERSION 0.1.0
  DESCRIPTION "Batched CPU and GPU inference for GFN-xTB"
  LANGUAGES C CXX
)

include(CheckLanguage)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(GPUXTB_ENABLE_CUDA "AUTO" CACHE STRING "Build the CUDA backend: AUTO, ON, or OFF")
set_property(CACHE GPUXTB_ENABLE_CUDA PROPERTY STRINGS AUTO ON OFF)
option(GPUXTB_BUILD_TESTS "Build gpuxtb tests" ${PROJECT_IS_TOP_LEVEL})

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

add_library(gpuxtb
  src/api.cpp
  src/runtime/backend.cpp
)
add_library(gpuxtb::gpuxtb ALIAS gpuxtb)

target_compile_features(gpuxtb PUBLIC cxx_std_17)
target_include_directories(gpuxtb
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_definitions(gpuxtb PRIVATE GPUXTB_BUILDING_LIBRARY)
set_target_properties(gpuxtb PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

if(GPUXTB_HAS_CUDA)
  target_sources(gpuxtb PRIVATE src/backends/cuda/cuda_runtime.cu)
  target_link_libraries(gpuxtb PRIVATE CUDA::cudart)
  target_compile_definitions(gpuxtb PRIVATE GPUXTB_HAS_CUDA=1)
  set_target_properties(gpuxtb PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
  )
endif()

if(GPUXTB_BUILD_TESTS)
  enable_testing()

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
endif()

install(TARGETS gpuxtb EXPORT gpuxtbTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT gpuxtbTargets
  FILE gpuxtbTargets.cmake
  NAMESPACE gpuxtb::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpuxtb
)

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
