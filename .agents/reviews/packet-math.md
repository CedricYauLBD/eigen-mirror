# Packet-Math Review

Use this checklist to review a diff that touches the packet layer: `Eigen/src/Core/arch/`,
`Eigen/src/Core/GenericPacketMath.h`, `packet_traits` or `unpacket_traits`, or code that calls `internal::p*`
operations from paths every backend instantiates. Run the review in a sub-agent or a fresh session when the harness
has one, so that it is not anchored on the author's reasoning. The reviewer reports findings and does not edit files.

## First: load the standard

Read these from the checkout before looking at the diff. They are the review standard; this checklist only adds
procedure and lessons the guides do not record.

- [`simd-gpu.md`](../simd-gpu.md) — packet layer, device-callable code, validation.
- [`numerics.md`](../numerics.md#packet-accuracy), section "Packet Accuracy" — special values and lane rules for
  math functions.
- [`conventions.md`](../conventions.md#hot-paths), section "Hot paths".
- [`AGENTS.md`](../../AGENTS.md) rules 5 (annotations) and 6 (enable a fast path only when its precondition holds).

The checked-out tree wins over any guide or over this checklist. If they disagree, follow the tree and say so in the
report.

## Layout (verify against the tree)

- Generic scalar fallbacks and default traits: `Eigen/src/Core/GenericPacketMath.h`. Scalar math:
  `Eigen/src/Core/MathFunctions.h`.
- `Eigen/src/Core/arch/Default/`: shared **vector** implementations (`GenericPacketMathFunctions.h`, `Half.h`,
  `BFloat16.h`, ...), not scalar fallbacks.
- Backends: list `Eigen/src/Core/arch/` rather than trusting a remembered inventory.
- Feature macros: `Eigen/src/Core/util/ConfigureVectorization.h`.

## Getting the diff

Use read-only commands: `git diff`, `git show`, `git log`, `git merge-base`, `rg`, `ls`. Review what the caller names:
a single commit against its parent (`git show <sha>`), or a range or base. With nothing named, review
`git diff $(git merge-base upstream/master HEAD)` (try `origin/master` if `upstream` does not exist) plus uncommitted
changes. Use `rg -n '<op>\b' Eigen/src/Core/arch/` to enumerate specializations, and read narrow ranges.
Do not build, run tests, run benchmarks, or write files. Recommend the runs instead.

## What to check

1. **Changed operations.** List every `internal::p*` operation, trait, and packet type the diff adds, changes, or
   newly calls, and the scalar code the generic fallbacks reach (`cast_impl` and the functions in `MathFunctions.h`,
   `Half.h`, `BFloat16.h`). Scalar and packet paths must agree.
2. **Silent fallbacks.** This is the main hazard. A missing specialization is not always a compile error: some
   generic fallbacks are the identity or act on one lane (the generic `preverse` returns its argument). For every op
   the diff newly calls from code that all backends instantiate, confirm that each backend reaching the call
   implements it correctly, or that a correct scalar path is kept. SYCL is the usual gap.
3. **Backend coverage, not completeness.** Do not demand the op in every backend because one backend gained an
   intrinsic. Flag a missing specialization only when it is a silent-wrong fallback, when a capability flag claims it,
   or when the diff's own goal (for example a stated speedup on that ISA) needs it.
4. **Divergent semantics.** For each changed op, compare sibling implementations on NaN propagation, signed zero,
   saturation, rounding mode, and subnormal flushing. Plain `pmin`/`pmax` are implementation-defined on NaN
   (`pminmax_propagates_nan`); code that needs a NaN policy must request `PropagateNaN` or `PropagateNumbers`.
5. **Capability flags and traits.** `Has*` flags, half-packet types, `masked_load_available`/`masked_store_available`,
   casts, alignment, and cost metadata match what is implemented. A flag must not advertise a missing or semantically
   different op. New packet types update `packet_traits` and `unpacket_traits` together.
6. **Feature guards.** Each intrinsic is guarded by the macro that enables it, even inside its backend directory (AVX2
   or FMA in `arch/AVX/`, AVX512DQ/VL/FP16/BF16 in `arch/AVX512/`), with a fallback for narrower configurations.
7. **Masks.** `pselect`/`pblend`/`pcmp_*` results follow the backend's established mask layout (all-bits-set lanes).
   Integer-encoded fast paths gated on `packet_has_integer_packet` also admit the FP16 packets
   (`Packet16h`/`Packet32h`), whose integer packets lack `pcmp_eq`; gate on `float`/`double` explicitly.
8. **Lane isolation and tails.** One lane's NaN, inf, or subnormal must not change another lane. Tails, unaligned
   access, and scalar-remainder behavior are preserved. Special-value results match the scalar contract.
9. **Fast-math and compiler folding.** NaN/inf constants are wrapped in `EIGEN_FAST_MATH_CONSTANT_BARRIER`. clang ≥ 19
   folds `pcmp_eq(preinterpret<PacketI>(pabs(a)), 0)` back into an FP compare, so DAZ reads subnormals as zero; a
   classification must stay on the integer packet (`pand` off the sign bit, then compare).
10. **Hot-path cost.** A `predux_any` early exit is justified only for inputs that are certainly rare (NaN, inf,
    subnormal, near overflow/underflow, huge trig arguments), never for value-range splits or zero. Moving a cold path
    out of line (`EIGEN_DONT_INLINE`) can make GCC spill loop-invariant vectors, because a call clobbers every vector
    register; ask for codegen of the enclosing loop, not just a wrapper.
11. **Annotations.** No `EIGEN_DEVICE_FUNC` or `EIGEN_STRONG_INLINE` dropped from existing code. New code reachable
    from CUDA/HIP/SYCL device code carries `EIGEN_DEVICE_FUNC`. Host SIMD backends are disabled under CUDA/HIP, so
    their specializations do not need it.
12. **Tests.** Each changed op and packet type is exercised in `test/packetmath.cpp` (or its helpers) with special
    values placed alone among ordinary lanes; packet bit patterns are checked through `memcpy` to integers, not by
    reading a `float` lane through `numext::bit_cast` (ppc64le GCC quiets sNaN that way). `packetmath`'s `test_cast`
    compares with `areApprox`, so an exact cast or rounding contract needs bit-exact checks elsewhere (for example in
    `array_cwise`). A test exclusion for an emulator or compiler defect is scoped to that CI job or compiler version,
    not to a whole architecture. The configuration a workaround exists for (ISA, compiler version) is tested, or
    named as untested.

## Output

Every finding gives `file:line`, the rule it rests on, and the backend and packet type affected. Mark each finding
**confirmed** (visible in the code) or **needs a run** (state the configuration). Omit empty sections.

```
## Packet-math audit

### Changed operations
- `op<PacketType>` (backend): what changed

### Findings
1. [confirmed|needs a run] file:line — defect. Backend/type: ... Rule: ...

### Sibling backends
| Op | Backend | Status (ok / silent fallback / divergent / not applicable) | Note |

### Traits and flags
(none) | issues

### Recommended runs
Exact targets and configurations: `packetmath`, `packetmath_gather`, `packetmath_generic_{16,32,64}`,
`packetmath_fastmath*`, `special_packetmath`, plus the tests that reach changed scalar paths (e.g. `array_cwise_*`,
`half_float`, `bfloat16_float`), per affected ISA flag set; name backends that need a cross or emulated build.
```

Be terse. List facts; do not explain what packet math is, flag formatting, or propose unrelated refactors.
