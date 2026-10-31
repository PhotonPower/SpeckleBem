---
name: ilut-kernel-findings
description: WP-SC5 ILUT row-update speed-up - cause was a GCC store-forwarding stall on the std::complex multiplier, not __muldc3; bitwise-neutral fix, golden test design, measurement pitfalls
metadata:
  type: project
---

WP-SC5 (2026-10-31, branch wp/sc5-ilut-kernel).

- Profile: the WP-SC3 loop `w[j] -= l * f.val[e]` ran at ~33-44 cycles per complex MAC. Not
  __muldc3 (GCC inlines the naive product + `ucomisd/jp` NaN check; the call is the cold path).
  GCC 16 spilled the multiplier l to the stack as two 8-byte stores and reloaded it as one
  16-byte `movupd` every iteration -> failed store-to-load forwarding. The same loop moved into a
  functor (l by value) was already ~4x faster; explicit doubles (lr, li, raw pointers) ~6-7x at
  U (8e8 MAC/s idle-ish), ~4x at 16 um (4.4e8 MAC/s, memory-streaming of the U rows).
- MSYS2 UCRT64 GCC defaults to -march=nocona (SSE3: addsubpd), no FMA; CI Linux x86-64 SSE2.
- SSE2 intrinsics (interleaved [lr*ur, lr*ui] + [-li*ui, li*ur]) are bitwise equal but only
  ~5 % faster; prefetching the heap front's U row: no clear gain (noise +-15 % under load).
  Kept the scalar explicit version (no platform branch).
- Bitwise traps: a fill-in must be `w = -(l u)`, not `0 - l u` (sign of zero); exact zeros in
  w are never stored (mag > thr fails) but a zero real part next to a nonzero imag part is.
- Golden test: std::normal_distribution differs between libstdc++/libc++ -> portable inputs
  from mt19937_64 bits; GCC vs Clang factor bits differ (__divdc3 of libgcc vs compiler-rt,
  hypot) so the exact-bits table is per toolchain (_WIN32 + compiler), elsewhere 1e-14 only.
- Verification method: dump the ILUT sources from the driver once (scratch patch), then a
  scratch harness compares all factor arrays byte-wise old vs new library (U and 16 um).

**Why:** follow-ups (parallel A elimination, apply speed) start from these numbers.
**How to apply:** check generated asm of hot std::complex loops for stack round trips before
blaming __muldc3; see [[ilut-preconditioner-findings]].
