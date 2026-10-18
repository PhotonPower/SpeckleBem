// Validation 7h of the PML route at the 7c size (WP-PML5a; ADR 0008 amendment 2026-10-18 decision
// 6, docs/05 "PML route" row 7h): the fast path (compression "mlfmm" on the PML route: MLFMM on
// the real functions plus the exact PML ring, GMRES with the recommended diagonal preconditioner,
// tolerance 1e-7) against the dense PML matrix and its LU solve on the rough Ag U case of WP-PML4b
// (pml_rough_ag::u_case(): sigma = 50 nm, Lc = 500 nm, seed 1, L = 3.75 um, h = lambda / 4,
// w0 = lambda, PML defaults; 2N = 22 816, 45 % of the functions stretched). Criteria (docs/05,
// not loosened): matvec < 1e-3 at d0 = 3, R within 1e-4 relative, windowed far field L2 (ADR 0006
// metric, pml_rough_ag::compare) < 1e-3. Memory and time of both paths are reported (WARN ROW
// line; record benchmarks/results/pml_ring.md). A second case checks the d0 = 5 matvec gate
// (< 1e-5) on the same case without a solve (WARN MATVEC line).
//
// Label `validation-large` (release only, RUN_SERIAL; manual:
// `ctest --preset win-release -L validation-large -R "ring|7h"`). SKIPs when the dense matrix
// (8.3 GB, LU in place; the fast run is released before) does not fit into 60 % of the available
// memory.
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "pml_ring_support.hpp"

TEST_CASE("PML 7h: pml_ring fast path vs dense on rough Ag at the 7c size (2N = 22 816)",
          "[pml_7h]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    using namespace pml_ring;
    const RoughCase c = pml_rough_ag::u_case();
    const pml_rough_ag::Size s = pml_rough_ag::size_of(c);
    const Real avail = system_memory::available_memory_bytes();
    // The fast run (operator <~ 0.6 x dense) is released before the dense one: peak ~ dense.
    if (!(0.6 * avail > 1.2 * s.dense_bytes))
        SKIP("dense 2N = " << s.unknowns << " needs " << s.dense_bytes / 1e9
                           << " GB (+ the fast operator), available " << avail / 1e9 << " GB");
    const FastSettings f;  // d0 = 3, GMRES tolerance 1e-7
    const FastRun fr = run_fast(c, f);
    WARN(fr.report);
    const DenseRun dr = run_dense(c, fr);
    WARN(row_line(c, f, fr, dr));
    const pml_rough_ag::Metrics m = pml_rough_ag::compare(fr.far, dr.far);
    WARN(pml_rough_ag::compare_line(fr.far, dr.far, m));
    CHECK(fr.converged);
    CHECK(dr.residual < 1e-8);
    CHECK(fr.unknowns == s.unknowns);
    CHECK(fr.levels >= 3);
    CHECK(fr.ring_far_pairs > 0);
    // Validation 7h (docs/05; not loosened).
    CHECK(max_of(dr.matvec_errors) < 1e-3);
    CHECK(std::abs(m.dR) < 1e-4);
    CHECK(m.l2 < 1e-3);
#endif
}

TEST_CASE("PML 7h: pml_ring matvec at d0 = 5 vs dense on rough Ag at the 7c size (2N = 22 816)",
          "[pml_7h]") {
#ifndef NDEBUG
    SKIP("validation-large cases run in optimised builds only");
#else
    // The d0 = 5 gate of 7h at the 7c size, matvec only (the solve gates are the d0 = 3 case
    // above): the leaf rule gives lambda / 2 floors (r_max / a <= 0.3), so the near ring is
    // ~3.6x the d0 = 3 one; the fast operator (~5.5 GB) is released before the dense matrix.
    using namespace pml_ring;
    const RoughCase c = pml_rough_ag::u_case();
    const pml_rough_ag::Size s = pml_rough_ag::size_of(c);
    const Real avail = system_memory::available_memory_bytes();
    if (!(0.6 * avail > 1.2 * s.dense_bytes))
        SKIP("dense 2N = " << s.unknowns << " needs " << s.dense_bytes / 1e9
                           << " GB (+ the fast operator), available " << avail / 1e9 << " GB");
    FastSettings f;
    f.digits = 5.0;
    const MatvecRun r = run_matvec(c, f);
    WARN(matvec_line(c, f, r));
    CHECK(r.unknowns == s.unknowns);
    CHECK(r.levels >= 3);
    // Validation 7h (docs/05; not loosened).
    CHECK(max_of(r.errors) < 1e-5);
#endif
}
