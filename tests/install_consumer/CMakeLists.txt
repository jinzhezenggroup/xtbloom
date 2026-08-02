cmake_minimum_required(VERSION 3.24)

project(gpuxtb_install_consumer LANGUAGES C CXX)

find_package(gpuxtb CONFIG REQUIRED)

add_executable(gpuxtb_install_consumer main.c)
target_link_libraries(gpuxtb_install_consumer PRIVATE gpuxtb::gpuxtb)
set_target_properties(
  gpuxtb_install_consumer
  PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED YES
)
