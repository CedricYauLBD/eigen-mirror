# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# Checks the routine split that the LAPACK test drivers link against:
#  - every routine eigen_lapack.def exports is defined by an Eigen library,
#  - every Fortran-callable routine eigen_lapack defines is exported,
#  - the reference LAPACK library defines none of Eigen's routines, so the
#    drivers cannot silently test a reference copy instead of Eigen's,
#  - on ELF platforms, the shared eigen_lapack exports every listed routine.

cmake_minimum_required(VERSION 3.18)

foreach(variable NM DEF_FILE EIGEN_LAPACK EIGEN_BLAS REFERENCE)
  if(NOT DEFINED ${variable})
    message(FATAL_ERROR "LAPACK symbol check needs -D${variable}")
  endif()
endforeach()
if(NOT DEFINED SYMBOL_PREFIX)
  set(SYMBOL_PREFIX "")
endif()

# Fortran-callable global symbols that <library> defines, without SYMBOL_PREFIX.
function(fortran_symbols out library)
  if(NOT EXISTS "${library}")
    message(FATAL_ERROR "Missing library: ${library}")
  endif()
  execute_process(COMMAND "${NM}" ${ARGN} "${library}"
                  OUTPUT_VARIABLE listing RESULT_VARIABLE status ERROR_VARIABLE errors)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "${NM} failed on ${library}: ${errors}")
  endif()
  # Weak definitions count: Eigen's xerbla_ is weak so that a program can replace it.
  string(REGEX MATCHALL "[ \t][TDRBWV][ \t]+${SYMBOL_PREFIX}[a-z][a-z0-9_]*_\r?\n" matches "${listing}")
  set(symbols "")
  foreach(match IN LISTS matches)
    string(REGEX REPLACE "^[ \t][TDRBWV][ \t]+${SYMBOL_PREFIX}([a-z0-9_]+)[\r\n]*$" "\\1" symbol "${match}")
    list(APPEND symbols "${symbol}")
  endforeach()
  list(REMOVE_DUPLICATES symbols)
  set(${out} "${symbols}" PARENT_SCOPE)
endfunction()

file(STRINGS "${DEF_FILE}" exports REGEX "^[ \t]*[a-z0-9_]+_[ \t]*$")
set(exported "")
foreach(export IN LISTS exports)
  string(STRIP "${export}" export)
  list(APPEND exported "${export}")
endforeach()
if(NOT exported)
  message(FATAL_ERROR "No routines found in ${DEF_FILE}")
endif()

fortran_symbols(lapack_defined "${EIGEN_LAPACK}" -g)
fortran_symbols(blas_defined "${EIGEN_BLAS}" -g)
fortran_symbols(reference_defined "${REFERENCE}" -g)

set(errors "")
set(eigen_defined ";${lapack_defined};${blas_defined};")
foreach(symbol IN LISTS exported)
  string(FIND "${eigen_defined}" ";${symbol};" found)
  if(found EQUAL -1)
    string(APPEND errors "eigen_lapack.def exports ${symbol}, which no Eigen library defines\n")
  endif()
endforeach()
set(exported_set ";${exported};")
foreach(symbol IN LISTS lapack_defined)
  string(FIND "${exported_set}" ";${symbol};" found)
  if(found EQUAL -1)
    string(APPEND errors "eigen_lapack defines ${symbol}, which eigen_lapack.def does not export\n")
  endif()
endforeach()
set(reference_set ";${reference_defined};")
foreach(symbol IN LISTS lapack_defined blas_defined)
  string(FIND "${reference_set}" ";${symbol};" found)
  if(NOT found EQUAL -1)
    string(APPEND errors "${symbol} is defined by both Eigen and the reference LAPACK library\n")
  endif()
endforeach()

if(DEFINED EIGEN_LAPACK_SHARED)
  fortran_symbols(lapack_exported "${EIGEN_LAPACK_SHARED}" -D --defined-only)
  fortran_symbols(blas_exported "${EIGEN_BLAS_SHARED}" -D --defined-only)
  set(shared_exported ";${lapack_exported};${blas_exported};")
  foreach(symbol IN LISTS exported)
    string(FIND "${shared_exported}" ";${symbol};" found)
    if(found EQUAL -1)
      string(APPEND errors "The shared eigen_lapack library does not export ${symbol}; "
                           "declare it with EIGEN_LAPACK_API in lapack.h\n")
    endif()
  endforeach()
endif()

list(LENGTH exported exported_count)
list(LENGTH reference_defined reference_count)
message(STATUS "${exported_count} routines from Eigen, ${reference_count} from reference LAPACK")
if(errors)
  message(FATAL_ERROR "${errors}")
endif()
