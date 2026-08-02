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

add_library(gpuxtb_gfn2_cpu OBJECT
  src/model/gfn2/basis.cpp
  src/model/gfn2/coordination.cpp
  src/model/gfn2/es3.cpp
  src/model/gfn2/external_point_charges.cpp
  src/model/gfn2/h0.cpp
  src/model/gfn2/integrals.cpp
  src/model/gfn2/repulsion.cpp
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

add_library(gpuxtb
  src/api.cpp
  src/runtime/backend.cpp
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
set_target_properties(gpuxtb PROPERTIES
  CUDA_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)
if(UNIX AND NOT APPLE)
  target_link_options(gpuxtb PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpuxtb.map"
  )
  set_property(TARGET gpuxtb APPEND PROPERTY
    LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpuxtb.map
  )
endif()

if(GPUXTB_HAS_CUDA)
  add_library(gpuxtb_cuda_backend OBJECT
    src/backends/cuda/cuda_runtime.cu
    src/backends/cuda/gfn2_es3.cu
    src/backends/cuda/gfn2_external_point_charges.cu
    src/backends/cuda/gfn2_parameters.cu
    src/backends/cuda/gfn2_repulsion.cu
  )
  target_include_directories(gpuxtb_cuda_backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_link_libraries(gpuxtb_cuda_backend PUBLIC CUDA::cudart)
  target_compile_definitions(gpuxtb_cuda_backend PRIVATE GPUXTB_HAS_CUDA=1)
  set_target_properties(gpuxtb_cuda_backend PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
    VISIBILITY_INLINES_HIDDEN YES
  )

  target_sources(gpuxtb PRIVATE $<TARGET_OBJECTS:gpuxtb_cuda_backend>)
  target_link_libraries(gpuxtb PRIVATE CUDA::cudart)
  target_compile_definitions(gpuxtb PRIVATE GPUXTB_HAS_CUDA=1)
  set_target_properties(gpuxtb PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()

if(GPUXTB_BUILD_TESTS)
  enable_testing()
  find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)

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

  if(BUILD_SHARED_LIBS AND UNIX AND NOT APPLE)
    add_test(
      NAME gpuxtb.abi_symbols
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/abi/check_symbols.py
              --nm ${CMAKE_NM} --library $<TARGET_FILE:gpuxtb>
    )
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

  if(GPUXTB_HAS_CUDA)
    add_executable(gpuxtb_cuda_parameters_test tests/cuda_parameters_test.cu)
    target_link_libraries(gpuxtb_cuda_parameters_test PRIVATE
      gpuxtb::gpuxtb
      gpuxtb_cuda_backend
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
  endif()

  add_executable(gpuxtb_repulsion_test tests/repulsion_test.cpp)
  target_link_libraries(gpuxtb_repulsion_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_repulsion_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.repulsion COMMAND gpuxtb_repulsion_test)

  add_executable(gpuxtb_coordination_test tests/coordination_test.cpp)
  target_link_libraries(gpuxtb_coordination_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_coordination_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.coordination COMMAND gpuxtb_coordination_test)

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

  add_executable(gpuxtb_external_point_charges_test tests/external_point_charges_test.cpp)
  target_link_libraries(gpuxtb_external_point_charges_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_external_point_charges_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.external_point_charges COMMAND gpuxtb_external_point_charges_test)

  add_executable(gpuxtb_es3_test tests/es3_test.cpp)
  target_link_libraries(gpuxtb_es3_test PRIVATE gpuxtb_gfn2_cpu)
  target_compile_features(gpuxtb_es3_test PRIVATE cxx_std_17)
  add_test(NAME gpuxtb.gfn2.es3 COMMAND gpuxtb_es3_test)

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
  set_tests_properties(
    gpuxtb.parameters.generated
    gpuxtb.parameters.python
    gpuxtb.conformance.manifest
    gpuxtb.conformance.python
    PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
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
