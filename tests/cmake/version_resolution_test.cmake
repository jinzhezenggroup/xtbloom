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
