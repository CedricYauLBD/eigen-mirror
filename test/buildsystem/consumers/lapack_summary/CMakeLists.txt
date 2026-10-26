# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.18)
project(LapackSummary CXX)
enable_testing()
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

# Exercise LAPACK's real registration without a Fortran compiler or downloads.
include(CheckLanguage)
macro(check_language language)
  set(CMAKE_Fortran_COMPILER "unused-fortran")
endmacro()
macro(enable_language language)
endmacro()
set(EIGEN_BUILD_LAPACK ON)
set(EIGEN_BUILD_BLAS ON)
set(EIGEN_ENABLE_LAPACK_TESTS ON)
set(EIGEN_INSTALL OFF)
add_library(eigen_blas_static INTERFACE)
add_library(Eigen3::Eigen INTERFACE IMPORTED)
add_custom_target(buildtests)
foreach(precision s d)
  add_executable(xlintst${precision} IMPORTED)
  set_target_properties(xlintst${precision} PROPERTIES IMPORTED_LOCATION "${CMAKE_COMMAND}")
endforeach()

set(fixture "${CMAKE_CURRENT_BINARY_DIR}/lapack")
file(MAKE_DIRECTORY "${fixture}/reference" "${fixture}/testing")
file(GLOB sources "${EIGEN_SOURCE_DIR}/lapack/*.cpp")
file(COPY ${sources} "${EIGEN_SOURCE_DIR}/lapack/CMakeLists.txt"
          "${EIGEN_SOURCE_DIR}/lapack/testing_summary.cmake" DESTINATION "${fixture}")
foreach(directory MATGEN LIN EIG)
  file(MAKE_DIRECTORY "${fixture}/testing/${directory}")
  file(WRITE "${fixture}/testing/${directory}/CMakeLists.txt" "")
endforeach()
file(WRITE "${fixture}/testing/stest.in" "All tests passed ( 100 tests run)\n")
file(WRITE "${fixture}/testing/dtest.in" "All tests passed ( 20 tests run)\n")
file(WRITE "${fixture}/testing/runtest.cmake"
     "configure_file(\"\${INPUT}\" \"\${OUTPUT}\" COPYONLY)\n")

# Fail if configuration attempts to discover any Python interpreter.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/modules")
foreach(module Python PythonInterp Python2 Python3)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/modules/Find${module}.cmake"
       "message(FATAL_ERROR \"LAPACK tests must not require Python\")\n")
endforeach()
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/modules")
add_subdirectory("${fixture}" lapack-build)
