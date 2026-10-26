The LAPACK compatibility library is enabled with `EIGEN_BUILD_LAPACK` and
requires `EIGEN_BUILD_BLAS`. Its optional numerical tests
(`EIGEN_ENABLE_LAPACK_TESTS`) require a Fortran compiler and the LAPACK test data.

The tests and their summary run through CMake/CTest without Python.
`LAPACK_Test_Summary` waits for the registered numerical tests and combines
only their expected output files into `TESTING/testing_results.txt` in the
LAPACK build directory. It fails when the output reports numerical errors,
illegal-argument or INFO diagnostics, when an expected output is missing, or
when no numerical tests were analyzed. Run the numerical tests before selecting
the summary alone.
