# Numerical-Accuracy Review

Use this checklist to review a diff that touches numerical code: scalar or packet math functions, decompositions,
eigensolvers, linear or iterative solvers, matrix functions, or the numerical tests for them. Run the review in a
sub-agent or a fresh session when the harness has one, so that it is not anchored on the author's reasoning. The
reviewer reports findings and does not edit files.

## First: load the standard

Read these from the checkout before looking at the diff. They are the review standard; this checklist only adds
procedure and lessons the guides do not record.

- [`numerics.md`](../numerics.md) — contracts, special values, error metrics, decompositions, packet accuracy,
  documented bounds and shortcut paths, subnormals/FTZ, provenance.
- [`testing.md`](../testing.md#numerical-assertions), section "Numerical Assertions" — tolerance idiom and the
  comparisons that accept everything.
- [`AGENTS.md`](../../AGENTS.md) — "Scalar, index, and storage genericity" and rule 7 (verification ships with
  behavior).

The checked-out tree wins over any guide or over this checklist. If they disagree, follow the tree and say so in the
report.

## Getting the diff

Use read-only commands: `git diff`, `git show`, `git log`, `git merge-base`, `rg`, `ls`. Review what the caller names:
a single commit against its parent (`git show <sha>`), or a range or base. With nothing named, review
`git diff $(git merge-base upstream/master HEAD)` (try `origin/master` if `upstream` does not exist) plus uncommitted
changes. Do not build, run tests, run benchmarks, or write files. Recommend the runs instead.

## What to check

### Special values (exact contract, never traded for speed)

For each changed scalar or packet function, list the cases and whether the diff and its tests handle them: `±0` with
sign, `±inf`, quiet NaN, the smallest subnormal and the normal/subnormal boundary, both sides of every domain boundary,
overflow/underflow thresholds, and large arguments where range reduction breaks down. The normative contract is ISO
C++ / the C library it incorporates, and IEEE 754 where IEC 60559 is claimed. Cite cppreference only as a summary. Eigen
does not promise floating-point exception flags; do not ask for them.

Check that tests use explicit predicates — `(numext::isnan)`, `(numext::isinf)` plus sign, `==` plus
`(numext::signbit)` — because approximate comparisons equate NaNs and ignore the sign of zero.

### Finite-input accuracy

- A new or changed ULP claim needs evidence against MPFR (`test/ulp_accuracy`, target `ulp_accuracy`, `--ref=mpfr`),
  not against the scalar standard library. Compare with `doc/CoeffwiseMathFunctionsTable.dox` and flag a regression or
  an unrecorded change.
- New polynomial or rational coefficients must be reproducible: function, domain, precision, error objective, tool
  version, and the Sollya command or script.
- An exactly specified operation (conversion, rounding, integer or bit operation) is held to exact results, not a ULP
  budget. The evidence is bit-exact tests at each threshold, from both sides, plus an exhaustive sweep against an
  independent reference where the input space allows (all 2^32 inputs of a 32-bit source).
- A speed-for-accuracy trade is acceptable only with a principled bound of a few ULPs on finite inputs. Surface every
  such trade explicitly for a human decision; never approve or reject one yourself.

### Packet paths

- Scalar, generic packet, and every changed backend must agree on special values.
- Each lane is independent: one lane's NaN, inf, or subnormal must not change another lane. Tests should place each
  special value alone among ordinary lanes; tests that pad with zeros never reach a `predux_any` fast path.
- Tails, unaligned access, and approximation-region boundaries are covered.

### Decompositions, solvers, matrix functions

- Tests check backward error and invariants (normalized reconstruction, residual, orthogonality, structure), not one
  forward answer. A forward comparison needs a conditioning estimate.
- Coverage scales with the change and includes the structures the algorithm is sensitive to: ill-conditioned,
  near-singular, singular, rank-deficient, clustered or repeated spectra, defective, extreme scaling, and the matrix
  families listed in `numerics.md`. Status and error reporting are tested as well as successful results.
- No weakening of pivoting, of a stopping or deflation test, or of backward stability relative to the LAPACK
  counterpart without stated justification. Do not require identical pivots, signs, phases, or roundoff-level output.
- A scaling threshold or overflow budget stated in a comment is re-derived when the operation it bounds widens; early
  exits and length-one shortcuts meet the general path's invariant and have boundary tests.

### Test assertions

- `VERIFY_IS_APPROX` and `test_precision` alone are not evidence of accuracy for a new kernel.
- Tolerances are `factor * NumTraits<RealScalar>::epsilon()` at the site, with `factor` explained by an error model; a
  bare decimal literal is a finding.
- Flag comparisons that accept everything: a bound computed by the code under test, and `err <= tol` or
  `if (err > tol)` forms that pass on NaN or inf. Failure should be tested as `!(err <= tol)`.
- A widened tolerance needs evidence that the computation is already as accurate as its conditioning allows, measured
  against quad or MPFR, not against the next wider type.

### Build modes

- Under `-ffast-math` / `-ffinite-math-only`, compilers fold NaN/inf predicates and finiteness comparisons to
  constants, and clang turns NaN/inf constants into poison. New NaN/inf constants need
  `EIGEN_FAST_MATH_CONSTANT_BARRIER`; a passing fast-math test may have had its checks folded away. A runtime FP-mode
  probe must compare against an exactly representable expected value, not test finiteness.
- FTZ/DAZ handling goes through `test/fp_control.h` and stays narrow. That header ends with
  `#undef EIGEN_TEST_HAS_X86_FTZ`, so a test that gates DAZ on it after the include never enables DAZ.
- A subnormal result carries `log2(x / denorm_min)` significant bits. A tolerance near underflow extends the relative
  bound `p * |x|` with the absolute bound `p * min`; "both below `min`" is not agreement.
- A test exclusion for an emulator or compiler defect is scoped to that configuration (the CI job, the compiler
  version), not to a whole architecture; otherwise native hardware loses the coverage too.
- A workaround gated on a compiler version needs evidence that versions outside the gate are correct, and a test that
  fails at the parent commit in the gated configuration.

### Provenance

A new algorithm or constant set cites its source inline (author, year, algorithm or routine). Flag code that appears
adapted from a source of unknown or incompatible license.

## Known reference pitfalls

- libc++ `std::log(std::complex)` computes the real part as `log(abs(z))`, so it is not a boundary-accurate reference
  near `|z| = 1`.
- Reading a packet lane back as `float` through `numext::bit_cast` can quiet an sNaN (GCC 14 `-O3` on ppc64le). Bit
  tests must transport bytes with `memcpy` into integers.

## Output

Write the report in these sections. Every finding gives `file:line`, the rule it rests on (guide section or standard
clause), and a concrete input or matrix that exposes it. Mark each finding **confirmed** (visible in the code) or
**needs a run** (state the run). Omit empty sections.

```
## Numerical-accuracy audit

### Scope
Functions, kernels, and tests in the diff.

### Findings
1. [confirmed|needs a run] file:line — defect. Rule: ... Exposing input: ...

### Special-value table (per changed function)
| Case | Contract | Diff behavior | Test coverage |

### Accuracy evidence
ULP / backward-error evidence present or missing; trades surfaced for a human decision.

### Coverage gaps
Missing structures, special values, lanes, build modes.

### Recommended runs
Exact targets: e.g. `packetmath`, `packetmath_fastmath`, `special_packetmath`,
`ulp_accuracy --func=... --ref=mpfr`, the decomposition's test with fixed seeds.
```

Be terse. Do not explain textbook numerical analysis, flag formatting, or propose refactors unrelated to accuracy.
