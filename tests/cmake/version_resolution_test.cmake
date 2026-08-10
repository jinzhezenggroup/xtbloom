cmake_minimum_required(VERSION 3.24)

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/xtbloom_version.cmake")

if(DEFINED XTBLOOM_TEST_INVALID_TAG)
  _xtbloom_versions_from_tag(
    "${XTBLOOM_TEST_INVALID_TAG}" ignored_release ignored_full
  )
  message(FATAL_ERROR "invalid tag was unexpectedly accepted")
endif()

if(DEFINED XTBLOOM_TEST_INVALID_METADATA)
  _xtbloom_versions_from_metadata(
    "${XTBLOOM_TEST_INVALID_METADATA}" "test metadata"
    ignored_release ignored_full
  )
  message(FATAL_ERROR "invalid metadata version was unexpectedly accepted")
endif()

if(DEFINED XTBLOOM_TEST_SKBUILD_VERSION)
  set(SKBUILD_PROJECT_VERSION "${XTBLOOM_TEST_SKBUILD_VERSION}")
  set(SKBUILD_PROJECT_VERSION_FULL "${XTBLOOM_TEST_SKBUILD_VERSION_FULL}")
  xtbloom_resolve_version(observed_release observed_full)
  if(NOT observed_release STREQUAL XTBLOOM_TEST_EXPECTED_NATIVE_VERSION OR
     NOT observed_full STREQUAL XTBLOOM_TEST_EXPECTED_NATIVE_VERSION)
    message(FATAL_ERROR
      "scikit-build metadata ${SKBUILD_PROJECT_VERSION_FULL}: expected native "
      "${XTBLOOM_TEST_EXPECTED_NATIVE_VERSION}, got "
      "${observed_release}/${observed_full}"
    )
  endif()
  return()
endif()

function(assert_version tag expected_release expected_full)
  _xtbloom_versions_from_tag("${tag}" observed_release observed_full)
  if(NOT observed_release STREQUAL expected_release)
    message(FATAL_ERROR
      "${tag}: expected release ${expected_release}, got ${observed_release}"
    )
  endif()
  if(NOT observed_full STREQUAL expected_full)
    message(FATAL_ERROR
      "${tag}: expected full ${expected_full}, got ${observed_full}"
    )
  endif()
endfunction()

assert_version("v0.0.0" "0.0.0" "0.0.0")
assert_version("v1.2.3" "1.2.3" "1.2.3")
assert_version("v2048.16.32" "2048.16.32" "2048.16.32")

function(assert_metadata_version metadata expected_native)
  _xtbloom_versions_from_metadata(
    "${metadata}" "test metadata" observed_release observed_full
  )
  if(NOT observed_release STREQUAL expected_native OR
     NOT observed_full STREQUAL expected_native)
    message(FATAL_ERROR
      "metadata ${metadata}: expected native ${expected_native}, got "
      "${observed_release}/${observed_full}"
    )
  endif()
endfunction()

assert_metadata_version("0.0.0" "0.0.0")
assert_metadata_version("0.0.0.post1.dev20+g0123456" "0.0.0")
assert_metadata_version("0.0.0.post1.dev0+g0123456.d20260810" "0.0.0")
assert_metadata_version("1.2.3.post1.dev7+g0123456789" "1.2.3")
assert_metadata_version(
  "2048.16.32.post1.dev1+gabcdef012.d20260810" "2048.16.32"
)

execute_process(
  COMMAND ${CMAKE_COMMAND}
          -DXTBLOOM_TEST_SKBUILD_VERSION=1.2.3
          -DXTBLOOM_TEST_SKBUILD_VERSION_FULL=1.2.3.post1.dev7+g0123456789
          -DXTBLOOM_TEST_EXPECTED_NATIVE_VERSION=1.2.3
          -P ${CMAKE_CURRENT_LIST_FILE}
  RESULT_VARIABLE skbuild_status
  OUTPUT_VARIABLE skbuild_output
  ERROR_VARIABLE skbuild_error
)
if(NOT skbuild_status EQUAL 0)
  message(FATAL_ERROR
    "scikit-build development metadata was rejected:\n"
    "${skbuild_output}${skbuild_error}"
  )
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -DXTBLOOM_TEST_INVALID_TAG=v01.2.3
          -P ${CMAKE_CURRENT_LIST_FILE}
  RESULT_VARIABLE invalid_tag_status
  OUTPUT_QUIET
  ERROR_QUIET
)
if(invalid_tag_status EQUAL 0)
  message(FATAL_ERROR "tag components with leading zeroes were accepted")
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -DXTBLOOM_TEST_INVALID_METADATA=01.2.3
          -P ${CMAKE_CURRENT_LIST_FILE}
  RESULT_VARIABLE invalid_metadata_status
  OUTPUT_QUIET
  ERROR_QUIET
)
if(invalid_metadata_status EQUAL 0)
  message(FATAL_ERROR "metadata components with leading zeroes were accepted")
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -DXTBLOOM_TEST_INVALID_METADATA=1.2.4.dev7
          -P ${CMAKE_CURRENT_LIST_FILE}
  RESULT_VARIABLE guessed_next_dev_status
  OUTPUT_QUIET
  ERROR_QUIET
)
if(guessed_next_dev_status EQUAL 0)
  message(FATAL_ERROR "next-release development metadata was accepted")
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -DXTBLOOM_TEST_INVALID_METADATA=1.2.3.post2.dev7
          -P ${CMAKE_CURRENT_LIST_FILE}
  RESULT_VARIABLE post_release_status
  OUTPUT_QUIET
  ERROR_QUIET
)
if(post_release_status EQUAL 0)
  message(FATAL_ERROR "unsupported post-development metadata was accepted")
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -DXTBLOOM_TEST_INVALID_METADATA=1.2.3.post1.dev7
          -P ${CMAKE_CURRENT_LIST_FILE}
  RESULT_VARIABLE missing_node_status
  OUTPUT_QUIET
  ERROR_QUIET
)
if(missing_node_status EQUAL 0)
  message(FATAL_ERROR "development metadata without Git identity was accepted")
endif()

# Exercise the expanded Git archive metadata path without scikit-build or Git.
# The same full describe value identifies the Python artifact and carries the
# nearest strict tag from which CMake reconstructs the native product version.
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef archive_suffix)
set(archive_root "${CMAKE_CURRENT_BINARY_DIR}/xtbloom-archive-${archive_suffix}")
file(MAKE_DIRECTORY "${archive_root}")
file(WRITE "${archive_root}/.git_archival.txt"
  "node: 0123456789abcdef0123456789abcdef01234567\n"
  "node-date: 2026-08-10T22:14:05+08:00\n"
  "describe-name: v1.2.3-7-g0123456789abcdef0123456789abcdef01234567\n"
)
file(WRITE "${archive_root}/CMakeLists.txt"
  "cmake_minimum_required(VERSION 3.24)\n"
  "include(\"${CMAKE_CURRENT_LIST_DIR}/../../cmake/xtbloom_version.cmake\")\n"
  "xtbloom_resolve_version(release_version full_version)\n"
  "if(NOT release_version STREQUAL \"1.2.3\" OR "
  "NOT full_version STREQUAL \"1.2.3\")\n"
  "  message(FATAL_ERROR \"archive resolved \${release_version}/\${full_version}\")\n"
  "endif()\n"
  "project(version_archive_fixture VERSION \${release_version} LANGUAGES NONE)\n"
)
execute_process(
  COMMAND ${CMAKE_COMMAND} -S "${archive_root}" -B "${archive_root}/build"
  RESULT_VARIABLE archive_status
  OUTPUT_VARIABLE archive_output
  ERROR_VARIABLE archive_error
)
file(REMOVE_RECURSE "${archive_root}")
if(NOT archive_status EQUAL 0)
  message(FATAL_ERROR
    "expanded Git archive version resolution failed:\n"
    "${archive_output}${archive_error}"
  )
endif()

# Exercise repository selection, not only the tag parser. The whole v*
# namespace is reserved for product versions, so a nearer malformed tag must
# block configuration instead of falling back to an older valid release.
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef test_suffix)
set(test_root "${CMAKE_CURRENT_BINARY_DIR}/xtbloom-version-${test_suffix}")
file(MAKE_DIRECTORY "${test_root}")
file(WRITE "${test_root}/CMakeLists.txt"
  "cmake_minimum_required(VERSION 3.24)\n"
  "include(\"${CMAKE_CURRENT_LIST_DIR}/../../cmake/xtbloom_version.cmake\")\n"
  "xtbloom_resolve_version(release_version full_version)\n"
  "project(version_fixture VERSION \${release_version} LANGUAGES NONE)\n"
)
file(WRITE "${test_root}/marker.txt" "valid release\n")

function(run_fixture_git)
  execute_process(
    COMMAND git ${ARGN}
    WORKING_DIRECTORY "${test_root}"
    RESULT_VARIABLE git_status
    OUTPUT_VARIABLE git_output
    ERROR_VARIABLE git_error
  )
  if(NOT git_status EQUAL 0)
    file(REMOVE_RECURSE "${test_root}")
    message(FATAL_ERROR
      "fixture git ${ARGN} failed: ${git_error}${git_output}"
    )
  endif()
endfunction()

run_fixture_git(init --quiet)
run_fixture_git(config user.name "xTBloom version test")
run_fixture_git(config user.email "version-test@example.invalid")
run_fixture_git(add .)
run_fixture_git(commit --quiet -m "valid release")
run_fixture_git(tag v0.0.0)
file(APPEND "${test_root}/marker.txt" "malformed release\n")
run_fixture_git(add marker.txt)
run_fixture_git(commit --quiet -m "malformed release tag")
run_fixture_git(tag v1.2)

execute_process(
  COMMAND ${CMAKE_COMMAND} -S "${test_root}" -B "${test_root}/build"
  RESULT_VARIABLE malformed_repository_status
  OUTPUT_VARIABLE malformed_repository_output
  ERROR_VARIABLE malformed_repository_error
)
file(REMOVE_RECURSE "${test_root}")
if(malformed_repository_status EQUAL 0)
  message(FATAL_ERROR
    "repository with nearer malformed v tag unexpectedly configured:\n"
    "${malformed_repository_output}${malformed_repository_error}"
  )
endif()
