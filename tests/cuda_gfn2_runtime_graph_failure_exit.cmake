if(NOT DEFINED TEST_BINARY)
  message(FATAL_ERROR "TEST_BINARY must name the runtime-graph executable")
endif()

execute_process(
  COMMAND "${TEST_BINARY}" --check-failure-exit-only
  RESULT_VARIABLE failure_exit
  OUTPUT_VARIABLE probe_stdout
  ERROR_VARIABLE probe_stderr
)

if(NOT failure_exit STREQUAL "1")
  message(FATAL_ERROR
    "CHECK failure must exit exactly 1, not ${failure_exit}: ${probe_stdout}${probe_stderr}")
endif()
