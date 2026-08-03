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
set(GPUXTB_MKL_RT_LIBRARY "" CACHE FILEPATH
  "Optional MKL runtime library used to build the CPU eigensolver test"
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
if(CMAKE_DL_LIBS)
  target_link_libraries(gpuxtb PRIVATE ${CMAKE_DL_LIBS})
endif()
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
    src/backends/cuda/gfn2_terminal_classical_energy.cu
    src/backends/cuda/gfn2_total_energy.cu
  )
  target_include_directories(gpuxtb_cuda_backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_link_libraries(gpuxtb_cuda_backend PUBLIC
    CUDA::cublas
    CUDA::cudart
    CUDA::cusolver
  )
  target_compile_definitions(gpuxtb_cuda_backend PRIVATE GPUXTB_HAS_CUDA=1)
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
    POSITION_INDEPENDENT_CODE ON
    VISIBILITY_INLINES_HIDDEN YES
  )

  target_sources(gpuxtb PRIVATE $<TARGET_OBJECTS:gpuxtb_cuda_backend>)
  target_link_libraries(gpuxtb PRIVATE
    CUDA::cublas
    CUDA::cudart
    CUDA::cusolver
  )
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
  find_package(Threads REQUIRED)
  find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)

  if(GPUXTB_MKL_RT_LIBRARY)
    get_filename_component(GPUXTB_MKL_RT_DIRECTORY
      "${GPUXTB_MKL_RT_LIBRARY}" DIRECTORY
    )
    if(WIN32)
      set(GPUXTB_RUNTIME_LIBRARY_PATH_VARIABLE PATH)
    elseif(APPLE)
      set(GPUXTB_RUNTIME_LIBRARY_PATH_VARIABLE DYLD_LIBRARY_PATH)
    else()
      set(GPUXTB_RUNTIME_LIBRARY_PATH_VARIABLE LD_LIBRARY_PATH)
    endif()

    function(gpuxtb_configure_mkl_runtime_test target_name test_name)
      # The production eigensolver opens MKL by soname. Give CTest (including
      # sanitizer builds) an explicit loader path instead of relying only on
      # the executable's build RPATH being inherited by dlopen.
      set_target_properties(${target_name} PROPERTIES
        BUILD_RPATH "${GPUXTB_MKL_RT_DIRECTORY}"
      )
      set_tests_properties(${test_name} PROPERTIES
        ENVIRONMENT "MKL_INTERFACE_LAYER=LP64;MKL_THREADING_LAYER=SEQUENTIAL"
        ENVIRONMENT_MODIFICATION
          "${GPUXTB_RUNTIME_LIBRARY_PATH_VARIABLE}=path_list_prepend:${GPUXTB_MKL_RT_DIRECTORY}"
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
  if(GPUXTB_MKL_RT_LIBRARY)
    gpuxtb_configure_mkl_runtime_test(gpuxtb_runtime_test gpuxtb.runtime)
  endif()

  if(GPUXTB_MKL_RT_LIBRARY)
    add_executable(gpuxtb_cpu_public_inference_test
      tests/cpu_public_inference_test.cpp
    )
    target_link_libraries(gpuxtb_cpu_public_inference_test PRIVATE gpuxtb::gpuxtb)
    target_compile_features(gpuxtb_cpu_public_inference_test PRIVATE cxx_std_17)
    add_test(NAME gpuxtb.cpu.public_inference COMMAND gpuxtb_cpu_public_inference_test)
    gpuxtb_configure_mkl_runtime_test(
      gpuxtb_cpu_public_inference_test gpuxtb.cpu.public_inference
    )
  endif()

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

  if(GPUXTB_MKL_RT_LIBRARY)
    add_executable(gpuxtb_eigensolver_test tests/eigensolver_test.cpp)
    target_link_libraries(gpuxtb_eigensolver_test PRIVATE
      gpuxtb_gfn2_cpu
      ${GPUXTB_MKL_RT_LIBRARY}
      Threads::Threads
    )
    target_compile_features(gpuxtb_eigensolver_test PRIVATE cxx_std_17)
    add_test(NAME gpuxtb.gfn2.eigensolver COMMAND gpuxtb_eigensolver_test)
    gpuxtb_configure_mkl_runtime_test(
      gpuxtb_eigensolver_test gpuxtb.gfn2.eigensolver
    )
  else()
    message(STATUS
      "gpuxtb eigensolver test: disabled (set GPUXTB_MKL_RT_LIBRARY to enable)"
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
  add_test(
    NAME gpuxtb.oracle.tblite_observer_patch
    COMMAND ${Python3_EXECUTABLE} -m unittest discover
            -s tests/oracle -p "test_*.py" -v
  )
  set_tests_properties(
    gpuxtb.parameters.generated
    gpuxtb.parameters.python
    gpuxtb.conformance.manifest
    gpuxtb.conformance.python
    gpuxtb.oracle.tblite_observer_patch
    PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()

install(TARGETS gpuxtb EXPORT gpuxtbTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
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
