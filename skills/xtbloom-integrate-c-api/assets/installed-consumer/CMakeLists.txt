cmake_minimum_required(VERSION 3.24)

project(xtbloom_c_consumer LANGUAGES C CXX)

find_package(xtbloom CONFIG REQUIRED)

add_executable(xtbloom_c_consumer main.c)
target_link_libraries(xtbloom_c_consumer PRIVATE xtbloom::xtbloom)
target_compile_features(xtbloom_c_consumer PRIVATE c_std_11)

if(MSVC)
  target_compile_options(xtbloom_c_consumer PRIVATE /W4)
else()
  target_compile_options(xtbloom_c_consumer PRIVATE -Wall -Wextra -Wpedantic)
endif()
