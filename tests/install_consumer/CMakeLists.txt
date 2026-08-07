cmake_minimum_required(VERSION 3.24)

project(gpuxtb_install_consumer LANGUAGES C CXX)

find_package(gpuxtb CONFIG REQUIRED)

add_executable(gpuxtb_install_consumer main.c)
target_link_libraries(gpuxtb_install_consumer PRIVATE gpuxtb::gpuxtb)
set_target_properties(
  gpuxtb_install_consumer
  PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED YES
)

option(GPUXTB_STAGE_MKL_SHIM
  "Stage an installed gpuxtb MKL provider shim beside this test consumer" OFF
)
if(GPUXTB_STAGE_MKL_SHIM)
  if(NOT TARGET gpuxtb::mkl_lp64_shim)
    message(FATAL_ERROR
      "GPUXTB_STAGE_MKL_SHIM requires an install built with the isolated MKL provider"
    )
  endif()
  add_custom_command(TARGET gpuxtb_install_consumer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE:gpuxtb::mkl_lp64_shim>"
      "$<TARGET_FILE_DIR:gpuxtb_install_consumer>"
    COMMENT "Staging the installed gpuxtb MKL provider shim"
  )
endif()
