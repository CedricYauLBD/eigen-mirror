# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

bs_configure("LAPACK summary without Python" "${BS_CONSUMER_DIR}/lapack_summary"
             "${WORK_DIR}/build" "-DEIGEN_SOURCE_DIR=${EIGEN_SOURCE_DIR}")
# The numerical fixture creates its output; the summary must wait for it under -j.
bs_run(WHAT "registered LAPACK tests without Python"
       COMMAND "${CMAKE_COMMAND}" -E chdir "${WORK_DIR}/build"
               "${CMAKE_CTEST_COMMAND}" -C "${BUILD_CONFIG}" --parallel 2
               --output-on-failure --no-tests=error)

set(summary "${EIGEN_SOURCE_DIR}/lapack/testing_summary.cmake")
set(report "${WORK_DIR}/report.txt")
set(manifest "${WORK_DIR}/outputs.txt")
set(first "${WORK_DIR}/first output.out")
set(second "${WORK_DIR}/second.out")
file(WRITE "${first}" "Note; unrelated run information\nAll tests passed ( 100 tests run)\n")
file(WRITE "${second}" "All tests passed ( 20 tests run)\n")
file(WRITE "${WORK_DIR}/stale.out" "999 out of 999 tests failed\n")
file(WRITE "${manifest}" "${first}\n${second}\n")
bs_run(WHAT "successful LAPACK summary" OUTPUT_VARIABLE output
       COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}"
               "-DREPORT_FILE=${report}" -P "${summary}")
if(NOT output MATCHES "TOTAL: 120 tests, 0 numerical errors, 0 other errors")
  bs_fail("incorrect successful summary: ${output}")
endif()
file(READ "${report}" contents)
bs_assert_streq("${contents}" "Note; unrelated run information\nAll tests passed ( 100 tests run)\nAll tests passed ( 20 tests run)\n"
                "combined LAPACK report")

foreach(case numerical diagnostics xerbla error_exits info_summary error_code recorded combined_errors empty mixed_empty missing)
  file(WRITE "${manifest}" "${first}\n")
  if(case STREQUAL "numerical")
    file(WRITE "${first}" "3 out of 20 tests failed ( 20 tests run)\n")
    set(expected "20 tests, 3 numerical errors, 0 other errors")
  elseif(case STREQUAL "diagnostics")
    file(WRITE "${first}" "100 tests run)\nillegal argument\nDGETRF returned INFO = 1\n")
    set(expected "100 tests, 0 numerical errors, 2 other errors")
  elseif(case STREQUAL "xerbla")
    file(WRITE "${first}" "100 tests run)\n*** XERBLA was called with SRNAME = DGETRS instead of DGETRF ***\n")
    set(expected "100 tests, 0 numerical errors, 1 other errors")
  elseif(case STREQUAL "error_exits")
    file(WRITE "${first}" "100 tests run)\n*** DGE routines failed the tests of the error exits ***\n")
    set(expected "100 tests, 0 numerical errors, 1 other errors")
  elseif(case STREQUAL "info_summary")
    file(WRITE "${first}" "*** DGETRF returned with INFO = 2 instead of 1\nAll tests for DGE routines passed the threshold ( 100 tests run)\n1 error messages recorded\n")
    set(expected "100 tests, 0 numerical errors, 1 other errors")
  elseif(case STREQUAL "error_code" OR case STREQUAL "recorded" OR case STREQUAL "combined_errors")
    file(WRITE "${first}" "")
    if(NOT case STREQUAL "recorded")
      file(APPEND "${first}" " *** Error code from DGETRF = 1 for M = 2, N = 2, type 1\n")
    endif()
    file(APPEND "${first}" " All tests for DGE routines passed the threshold ( 100 tests run)\n")
    if(NOT case STREQUAL "error_code")
      file(APPEND "${first}" " 1 error messages recorded\n")
    endif()
    set(expected "100 tests, 0 numerical errors, 1 other errors")
  elseif(case STREQUAL "empty" OR case STREQUAL "mixed_empty")
    file(WRITE "${first}" "")
    if(case STREQUAL "mixed_empty")
      file(WRITE "${manifest}" "${second}\n${first}\n")
    endif()
    set(expected "No LAPACK numerical tests were analyzed")
  else()
    file(REMOVE "${first}")
    set(expected "Missing LAPACK test output")
  endif()
  bs_run(WHAT "LAPACK summary: ${case}" EXPECT_RESULT FAILURE OUTPUT_VARIABLE output
         COMMAND "${CMAKE_COMMAND}" "-DOUTPUT_MANIFEST=${manifest}" "-DREPORT_FILE=${report}" -P "${summary}")
  if(NOT output MATCHES "${expected}")
    bs_fail("wrong failure for ${case}: ${output}")
  endif()
endforeach()
