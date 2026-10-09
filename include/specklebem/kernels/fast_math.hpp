#pragma once
/// @file fast_math.hpp
/// Branch-free elementary functions for the plain-kernel hot loop of kernels::element_blocks
/// (WP-P2): sin / cos with one shared argument reduction and exp of a non-positive argument.
///
/// Why: near and far pairs evaluate exp(-jkR)/R at 36 to 5 329 point pairs per call; with the
/// C library (UCRT64: one sincos and one exp call per point) these calls were ~30 to 40 ns per
/// kernel evaluation, the dominant cost of dense assembly. The functions below are plain
/// arithmetic without branches or library calls, so the compiler can inline and vectorise the
/// point loop (with -fno-math-errno for std::sqrt, set for src/kernels/operators.cpp).
///
/// Accuracy (tests/unit/test_fast_math.cpp, against the C library, 4e4 random arguments per
/// range): fast_sincos absolute error <= 1.1e-16 for |x| <= 1e3 and <= 2.2e-16 up to
/// kFastSincosMax, relative <= 2 ulp on |x| <= 0.78; fast_exp_nonpositive relative error
/// <= 2.2e-16 on [-708, 0]. Near/far element blocks (icosphere, vacuum / glass / Si / Ag) agree
/// with the C-library arithmetic to 2.7e-15 relative, far below any quadrature error.
///
/// Requirements: IEEE double arithmetic in round-to-nearest mode without reassociation (no
/// -ffast-math / -fassociative-math): round_to_int relies on (x + 1.5 2^52) - 1.5 2^52 being
/// evaluated as written. Fused multiply-add contraction is harmless. Both requirements are
/// checked at compile time below (-ffast-math defines __FAST_MATH__; x87 excess precision
/// gives FLT_EVAL_METHOD != 0).
#include "specklebem/core/types.hpp"

#include <bit>
#include <cfloat>
#include <cstdint>

#if defined(__FAST_MATH__)
#error "kernels/fast_math.hpp: round_to_int needs IEEE evaluation; do not build with -ffast-math"
#endif
static_assert(FLT_EVAL_METHOD == 0,
              "kernels/fast_math.hpp: round_to_int needs double evaluated in double precision "
              "(no x87 excess precision; use SSE2 arithmetic)");

namespace specklebem::kernels::fastmath {

/// 1.5 * 2^52: for |x| < 2^51, (x + kRoundMagic) - kRoundMagic is x rounded to an integer
/// (nearest, ties to even).
inline constexpr Real kRoundMagic = 6755399441055744.0;

/// x rounded to the nearest integer (ties to even), for |x| < 2^51.
inline Real round_to_int(Real x) {
    return (x + kRoundMagic) - kRoundMagic;
}

/// Largest |x| for fast_sincos: the quadrant index q = round(2 x / pi) stays below 2^20, so the
/// products of q with the 33-bit parts of pi / 2 are exact (Cody-Waite reduction, fdlibm
/// constants).
inline constexpr Real kFastSincosMax = 1.0e6;

/// sin(x) and cos(x) for |x| <= kFastSincosMax (outside: undefined result, not checked). Argument
/// reduction x = q pi / 2 + r with |r| <= pi / 4 in three steps (pi / 2 = kPio2_1 + kPio2_2 +
/// kPio2_3 to ~1e-31), minimax polynomials of fdlibm's __kernel_sin / __kernel_cos on |r| <= pi /
/// 4 (error < 1 ulp each), quadrant selection by arithmetic masks.
inline void fast_sincos(Real x, Real& s, Real& c) {
    constexpr Real kTwoOverPi = 6.36619772367581382433e-01;
    constexpr Real kPio2_1 = 1.57079632673412561417e+00;  // first 33 bits of pi / 2
    constexpr Real kPio2_2 = 6.07710050630396597660e-11;  // next 33 bits
    constexpr Real kPio2_3 = 2.02226624871116645580e-21;  // next 33 bits
    constexpr Real kS1 = -1.66666666666666324348e-01;
    constexpr Real kS2 = 8.33333333332248946124e-03;
    constexpr Real kS3 = -1.98412698298579493134e-04;
    constexpr Real kS4 = 2.75573137070700676789e-06;
    constexpr Real kS5 = -2.50507602534068634195e-08;
    constexpr Real kS6 = 1.58969099521155010221e-10;
    constexpr Real kC1 = 4.16666666666666019037e-02;
    constexpr Real kC2 = -1.38888888888741095749e-03;
    constexpr Real kC3 = 2.48015872894767294178e-05;
    constexpr Real kC4 = -2.75573143513906633035e-07;
    constexpr Real kC5 = 2.08757232129817482790e-09;
    constexpr Real kC6 = -1.13596475577881948265e-11;
    const Real q = round_to_int(x * kTwoOverPi);
    // x - q kPio2_1 is exact (Sterbenz; q kPio2_1 is exact), q kPio2_2 is exact.
    const Real r = ((x - q * kPio2_1) - q * kPio2_2) - q * kPio2_3;
    const Real z = r * r;
    const Real sr = r + (r * z) * (kS1 + z * (kS2 + z * (kS3 + z * (kS4 + z * (kS5 + z * kS6)))));
    const Real cr =
        (1.0 - 0.5 * z) + (z * z) * (kC1 + z * (kC2 + z * (kC3 + z * (kC4 + z * (kC5 + z * kC6)))));
    // Quadrant j = q mod 4 in 0..3: floor(q / 4) = round(q / 4 - 3 / 8) for integer q.
    const Real j = q - 4.0 * round_to_int(0.25 * q - 0.375);
    const bool swap = j == 1.0 || j == 3.0;
    const Real sv = swap ? cr : sr;
    const Real cv = swap ? sr : cr;
    s = j >= 2.0 ? -sv : sv;
    c = (j == 1.0 || j == 2.0) ? -cv : cv;
}

/// Arguments below this give exactly 0 in fast_exp_nonpositive (exp(-708) = 3.3e-308; the true
/// values below are subnormal or zero).
inline constexpr Real kFastExpMin = -708.0;

/// exp(x) for x <= 0 (x > 0: undefined result, not checked; NaN propagates; -inf and x <
/// kFastExpMin give 0). x = n ln 2 + r with |r| <= ln 2 / 2 (Cody-Waite, fdlibm constants),
/// exp(r) by its Taylor polynomial through r^13 (truncation < 4e-18 relative), times 2^n built
/// from the exponent bits.
inline Real fast_exp_nonpositive(Real x) {
    constexpr Real kLog2E = 1.44269504088896338700e+00;
    constexpr Real kLn2Hi = 6.93147180369123816490e-01;  // 32 significant bits of ln 2
    constexpr Real kLn2Lo = 1.90821492927058770002e-10;  // ln 2 - kLn2Hi
    constexpr Real kTwo52 = 4503599627370496.0;
    const Real xc = x < kFastExpMin ? kFastExpMin : x;
    const Real n = round_to_int(xc * kLog2E);  // -1021 <= n <= 0
    const Real r = (xc - n * kLn2Hi) - n * kLn2Lo;
    // Taylor coefficients 1 / m!, m = 13 .. 2.
    constexpr Real kF13 = 1.0 / 6227020800.0;
    constexpr Real kF12 = 1.0 / 479001600.0;
    constexpr Real kF11 = 1.0 / 39916800.0;
    constexpr Real kF10 = 1.0 / 3628800.0;
    constexpr Real kF9 = 1.0 / 362880.0;
    constexpr Real kF8 = 1.0 / 40320.0;
    constexpr Real kF7 = 1.0 / 5040.0;
    constexpr Real kF6 = 1.0 / 720.0;
    constexpr Real kF5 = 1.0 / 120.0;
    constexpr Real kF4 = 1.0 / 24.0;
    constexpr Real kF3 = 1.0 / 6.0;
    constexpr Real kF2 = 0.5;
    const Real p =
        kF2 +
        r * (kF3 +
             r * (kF4 +
                  r * (kF5 +
                       r * (kF6 +
                            r * (kF7 +
                                 r * (kF8 +
                                      r * (kF9 +
                                           r * (kF10 + r * (kF11 + r * (kF12 + r * kF13))))))))));
    const Real e = 1.0 + r * (1.0 + r * p);
    // 2^n: the low mantissa bits of n + 1023 + 2^52 are n + 1023 (1 <= n + 1023 <= 1023);
    // shifted into the exponent field.
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(n + (1023.0 + kTwo52)) << 52;
    const Real scale = std::bit_cast<Real>(bits);
    return x < kFastExpMin ? 0.0 : e * scale;
}

}  // namespace specklebem::kernels::fastmath
