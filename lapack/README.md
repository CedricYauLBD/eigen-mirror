# Eigen's LAPACK library

This directory builds a library that implements a subset of LAPACK's Fortran 77
interface in C++, on top of Eigen's dense decompositions. A program that calls
only the routines listed below can link it in place of reference LAPACK.
Eigen's C++ API does not use it.

## Interface

- Symbols follow the Fortran 77 calling convention: a lowercase name with a
  trailing underscore (`dgetrf_`), every argument passed by pointer,
  column-major matrices with a leading dimension, and complex numbers stored as
  (real, imaginary) pairs. Dimensions, pivot indices and `INFO` are 32-bit
  `int`. Unlike Eigen's BLAS library, this library has no ILP64 build.
- An invalid `i`-th argument makes the routine call `XERBLA` and return
  `INFO = -i`, as in reference LAPACK. Eigen's BLAS library provides `XERBLA`
  as a weak symbol that prints the error and returns, where reference `XERBLA`
  stops the program.
- The driver routines allocate their own workspace. They use `WORK` only to
  answer a workspace query (`LWORK = -1`). `xSYEV` still requires
  `LWORK >= max(1, 3*N-1)`.

## Building

The library is enabled with `EIGEN_BUILD_LAPACK` and requires
`EIGEN_BUILD_BLAS`. Both default to `ON` when Eigen is the top-level CMake
project. The `lapack` target builds `eigen_lapack_static`, which links
`eigen_blas_static`. Where the platform supports shared libraries
(`EIGEN_BUILD_SHARED_LIBS`), it also builds `eigen_lapack`, which links
`eigen_blas`. `EIGEN_INSTALL` installs both.

## Routines

The tables follow the classification in Chapter 2 of the
[LAPACK Users' Guide](https://www.netlib.org/lapack/lug/), third edition:

- A driver routine solves a complete problem.
- A computational routine performs one step of a driver's work.
- An auxiliary routine is a building block of the other two.

An `x` in a routine name stands for a precision prefix. The Precisions column
lists the prefixes that exist:

- S: real single.
- D: real double.
- C: complex single.
- Z: complex double.

### Driver routines (Users' Guide §2.3)

| Problem | Routine | Precisions | Computes | Implemented with |
|---|---|---|---|---|
| Symmetric eigenproblem (SEP) | `xSYEV` | S, D | All eigenvalues and, optionally, the eigenvectors of a symmetric matrix | `SelfAdjointEigenSolver` |
| Singular value decomposition (SVD) | `xGESVD` | S, D, C, Z | `A = U Σ V^H`: the singular values and, optionally, the left and right singular vectors | `JacobiSVD` (two-sided Jacobi, not bidiagonal QR iteration) |
| | `xGESDD` | S, D, C, Z | The same, by divide and conquer | `BDCSVD` |

The library has no driver for any other problem class in §2.3:

- Linear equations (`xGESV`, `xPOSV`, ...).
- Linear least squares, and generalized linear least squares.
- Nonsymmetric eigenproblems.
- Generalized eigenvalue and singular value problems.
- The Hermitian eigenproblem (`CHEEV`, `ZHEEV`).

### Computational routines (Users' Guide §2.4)

| Problem | Routine | Precisions | Computes | Implemented with |
|---|---|---|---|---|
| Linear equations (§2.4.1), general matrix | `xGETRF` | S, D, C, Z | LU factorization with partial pivoting, `A = P L U` | `PartialPivLU`'s blocked kernel |
| | `xGETRS` | S, D, C, Z | Solves `A X = B`, `A^T X = B` or `A^H X = B` with the factors from `xGETRF` | Triangular solves |
| Linear equations (§2.4.1), symmetric or Hermitian positive definite matrix | `xPOTRF` | S, D, C, Z | Cholesky factorization, `A = U^H U` or `A = L L^H` | `LLT`'s blocked kernel |
| | `xPOTRS` | S, D, C, Z | Solves `A X = B` with the factor from `xPOTRF` | Triangular solves |

The library has no routines from §2.4.2 through §2.4.9. Those sections cover
orthogonal factorizations and their generalized forms, the symmetric,
nonsymmetric and generalized eigenproblems, the SVD, and the generalized SVD.

### Auxiliary routines (Users' Guide, Index of Auxiliary Routines)

The Users' Guide indexes auxiliary routines alphabetically. They are grouped
here by purpose.

| Purpose | Routine | Precisions | Computes |
|---|---|---|---|
| Householder reflectors | `xLARFG` | S, D, C, Z | An elementary reflector `H` with `H^H [α; x] = [β; 0]`, `β` real |
| | `xLARF` | S, D, C, Z | `H C` or `C H` for `H = I - τ v v^H` |
| | `xLARFT` | S, D, C, Z | The triangular factor `T` of a block reflector `H = I - V T V^H` |
| | `xLARFB` | S, D, C, Z | `H C`, `H^H C`, `C H` or `C H^H` for a block reflector `H` |
| Matrix scanning | `ILAxLR`, `ILAxLC` | S, D, C, Z | Index of the last nonzero row or column |
| Scalar arithmetic | `xLAPY2`, `xLAPY3` | S, D | `sqrt(x^2 + y^2)` and `sqrt(x^2 + y^2 + z^2)` without unnecessary overflow |
| | `xLADIV` | S, D, C, Z | Complex division without unnecessary overflow. S and D pass real and imaginary parts as separate arguments. |
| Vector operations | `xLACGV` | C, Z | Conjugates a vector |
| Machine parameters | `xLAMCH` | S, D | Floating-point parameters: machine epsilon, safe minimum, ... |
| | `xLAMC3` | S, D | `a + b`, forced through memory |
| Version | `ILAVER` | – | The reference LAPACK release that the routines follow, 3.12.1 |
| Timing | `SECOND`, `DSECND` | – | Process CPU time in seconds |

These routines are written directly to the reference LAPACK specification
rather than built on Eigen's decompositions. The Householder routines support
every combination of `SIDE`, `TRANS`, `DIRECT` and `STOREV`.

## Tests

`lapack_routines`, built with Eigen's other tests (`EIGEN_BUILD_TESTING`), checks Eigen's routines
directly from C++ and needs no Fortran compiler.

The optional numerical tests (`EIGEN_ENABLE_LAPACK_TESTS`) require a Fortran
compiler and the LAPACK test data. They use unmodified LAPACK 3.12.1 sources,
downloaded with a pinned SHA-256 hash. They exercise Eigen's BLAS/LAPACK
implementations, using reference LAPACK for the remaining routines. The
reference routines are compiled into `eigen_lapack_static`, so in a build with
the tests enabled that library is no longer a subset of LAPACK. All test cases remain enabled;
the summary reports known numerical and test-driver failures.

The tests and their summary run through CMake/CTest without Python.
`LAPACK_Test_Summary` waits for the registered numerical tests and combines
only their expected output files into `TESTING/testing_results.txt` in the
LAPACK build directory. It fails in any of these cases:

- The output reports numerical errors.
- The output reports illegal-argument or `INFO` diagnostics.
- An expected output is missing.
- No numerical tests were analyzed.

Selecting `LAPACK_Test_Summary` alone also runs its numerical tests through a
CTest fixture. If a numerical driver fails to execute successfully, CTest skips
the summary.

Balancing tests contribute their example counts. They fail on nonzero
`INFO`/`ILO` error counts or on non-finite maximum errors. Their finite maximum
errors remain in the combined report; these drivers do not specify a pass/fail
threshold.
