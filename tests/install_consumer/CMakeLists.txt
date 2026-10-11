cmake_minimum_required(VERSION 3.24)

project(xtbloom_install_consumer LANGUAGES C CXX)

find_package(xtbloom CONFIG REQUIRED)

add_executable(xtbloom_install_consumer main.c)
target_link_libraries(xtbloom_install_consumer PRIVATE xtbloom::xtbloom)
set_target_properties(
  xtbloom_install_consumer
  PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED YES
)

option(XTBLOOM_STAGE_MKL_SHIM
  "Stage an installed xtbloom MKL provider shim beside this test consumer" OFF
)
if(XTBLOOM_STAGE_MKL_SHIM)
  if(NOT TARGET xtbloom::mkl_lp64_shim OR
     NOT TARGET xtbloom::mkl_pthread_tss_bridge)
    message(FATAL_ERROR
      "XTBLOOM_STAGE_MKL_SHIM requires an install built with the complete isolated MKL provider"
    )
  endif()
  add_custom_command(TARGET xtbloom_install_consumer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE:xtbloom::mkl_pthread_tss_bridge>"
      "$<TARGET_FILE_DIR:xtbloom_install_consumer>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE:xtbloom::mkl_lp64_shim>"
      "$<TARGET_FILE_DIR:xtbloom_install_consumer>"
    COMMENT "Staging the installed xtbloom MKL pthread bridge and provider shim"
  )
endif()
