# Resolve xTBloom's product version from one tag-derived source.
#
# Python distributions use revision-aware setuptools-scm versions, while the
# embedded native product keeps the nearest strict vMAJOR.MINOR.PATCH tag.
# Commit distance, object IDs, and worktree dirtiness identify Python artifacts
# only; they never enter the native CMake/header/C API version. The C ABI
# generation and ELF SONAME remain separate manual decisions.

function(_xtbloom_validate_resolved_version release_version full_version source_name)
  if(NOT release_version MATCHES
      "^(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$")
    message(FATAL_ERROR
      "${source_name} produced invalid numeric release version '${release_version}'"
    )
  endif()
  if(NOT full_version STREQUAL release_version)
    message(FATAL_ERROR
      "${source_name} produced non-tag product version '${full_version}' "
      "instead of '${release_version}'"
    )
  endif()
endfunction()

function(_xtbloom_versions_from_tag tag_value out_release out_full)
  string(STRIP "${tag_value}" tag_value)
  if(NOT tag_value MATCHES
      "^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$")
    message(FATAL_ERROR
      "Git tag '${tag_value}' does not match strict vMAJOR.MINOR.PATCH"
    )
  endif()

  set(release_version "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")

  _xtbloom_validate_resolved_version(
    "${release_version}" "${release_version}" "Git tag"
  )
  set(${out_release} "${release_version}" PARENT_SCOPE)
  set(${out_full} "${release_version}" PARENT_SCOPE)
endfunction()

function(_xtbloom_versions_from_metadata full_version source_name out_release out_full)
  string(STRIP "${full_version}" full_version)
  set(release_pattern
    "^(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$"
  )
  string(CONCAT development_pattern
    "^(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)"
    "\\.post1\\.dev(0|[1-9][0-9]*)"
    "\\+g[0-9a-f]+"
    "(\\.d[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9])?$"
  )
  if(full_version MATCHES "${release_pattern}")
    set(release_version "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
  elseif(full_version MATCHES "${development_pattern}")
    # no-guess-dev adds post1.devN without changing Version.release. Keeping
    # that tuple unchanged lets the wheel identify its source commit without
    # changing the native product tag embedded in the same artifact.
    set(release_version "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
  else()
    message(FATAL_ERROR
      "${source_name} version '${full_version}' is neither a strict release "
      "nor a supported setuptools-scm development version"
    )
  endif()
  _xtbloom_validate_resolved_version(
    "${release_version}" "${release_version}" "${source_name} native version"
  )
  set(${out_release} "${release_version}" PARENT_SCOPE)
  set(${out_full} "${release_version}" PARENT_SCOPE)
endfunction()

function(xtbloom_resolve_version out_release out_full)
  if(DEFINED SKBUILD_PROJECT_VERSION AND DEFINED SKBUILD_PROJECT_VERSION_FULL)
    _xtbloom_versions_from_metadata(
      "${SKBUILD_PROJECT_VERSION_FULL}" "scikit-build-core"
      native_release native_full
    )
    if(NOT SKBUILD_PROJECT_VERSION STREQUAL native_release)
      message(FATAL_ERROR
        "scikit-build-core release tuple '${SKBUILD_PROJECT_VERSION}' does not "
        "match full Python version '${SKBUILD_PROJECT_VERSION_FULL}'"
      )
    endif()
    set(${out_release} "${native_release}" PARENT_SCOPE)
    set(${out_full} "${native_full}" PARENT_SCOPE)
    return()
  endif()

  # Do not accidentally inherit a parent repository when configuring an
  # unpacked sdist beneath another checkout. Git is authoritative only when
  # this source directory is itself the worktree root.
  find_program(XTBLOOM_GIT_EXECUTABLE NAMES git)
  if(XTBLOOM_GIT_EXECUTABLE)
    execute_process(
      COMMAND "${XTBLOOM_GIT_EXECUTABLE}" rev-parse --show-toplevel
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      RESULT_VARIABLE git_root_status
      OUTPUT_VARIABLE git_root
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(git_root_status EQUAL 0)
      file(REAL_PATH "${git_root}" git_root_real)
      file(REAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}" source_root_real)
      if(git_root_real STREQUAL source_root_real)
        execute_process(
          COMMAND "${XTBLOOM_GIT_EXECUTABLE}" rev-parse --is-shallow-repository
          WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
          RESULT_VARIABLE shallow_status
          OUTPUT_VARIABLE shallow_value
          ERROR_VARIABLE shallow_error
          OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(NOT shallow_status EQUAL 0)
          message(FATAL_ERROR "failed to inspect Git history depth: ${shallow_error}")
        endif()
        if(shallow_value STREQUAL "true")
          message(FATAL_ERROR
            "xTBloom version resolution requires complete Git tag history; shallow clone rejected"
          )
        endif()

        execute_process(
          COMMAND "${XTBLOOM_GIT_EXECUTABLE}" describe
                  --tags --match "v*" --abbrev=0
          WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
          RESULT_VARIABLE describe_status
          OUTPUT_VARIABLE describe_value
          ERROR_VARIABLE describe_error
          OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(NOT describe_status EQUAL 0)
          message(FATAL_ERROR
            "xTBloom requires a reachable strict vMAJOR.MINOR.PATCH Git tag: ${describe_error}"
          )
        endif()
        _xtbloom_versions_from_tag(
          "${describe_value}" release_version full_version
        )
        set(${out_release} "${release_version}" PARENT_SCOPE)
        set(${out_full} "${full_version}" PARENT_SCOPE)
        return()
      endif()
    endif()
  endif()

  set(pkg_info "${CMAKE_CURRENT_SOURCE_DIR}/PKG-INFO")
  if(EXISTS "${pkg_info}")
    file(STRINGS "${pkg_info}" pkg_version_line REGEX "^Version:[ \\t]*" LIMIT_COUNT 1)
    if(NOT pkg_version_line)
      message(FATAL_ERROR "${pkg_info} does not contain a Version field")
    endif()
    string(REGEX REPLACE "^Version:[ \\t]*" "" pkg_version "${pkg_version_line}")
    _xtbloom_versions_from_metadata(
      "${pkg_version}" "PKG-INFO" release_version full_version
    )
    set(${out_release} "${release_version}" PARENT_SCOPE)
    set(${out_full} "${full_version}" PARENT_SCOPE)
    return()
  endif()

  set(archival_file "${CMAKE_CURRENT_SOURCE_DIR}/.git_archival.txt")
  if(EXISTS "${archival_file}")
    file(STRINGS "${archival_file}" archival_describe_line
      REGEX "^describe-name:[ \\t]*" LIMIT_COUNT 1
    )
    string(REGEX REPLACE "^describe-name:[ \\t]*" ""
      archival_describe "${archival_describe_line}"
    )
    if(archival_describe AND NOT archival_describe MATCHES "\\$Format:")
      # The full describe value feeds setuptools-scm's Python identity. Strip
      # its distance and node suffix to recover the unchanged native tag.
      string(REGEX REPLACE "-[0-9]+-g[0-9a-f]+$" ""
        archival_tag "${archival_describe}"
      )
      _xtbloom_versions_from_tag(
        "${archival_tag}" release_version full_version
      )
      set(${out_release} "${release_version}" PARENT_SCOPE)
      set(${out_full} "${full_version}" PARENT_SCOPE)
      return()
    endif()
  endif()

  message(FATAL_ERROR
    "cannot resolve xTBloom version: provide a full Git checkout, expanded Git archive, "
    "or sdist PKG-INFO; no fallback version is permitted"
  )
endfunction()
