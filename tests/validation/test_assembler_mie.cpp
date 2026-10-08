// Validation of the dense assembly (WP9) against the Mie series, sphere of radius 0.5 um,
// lambda = 500 nm, plane wave k = +z, E along x (docs/05, docs/06):
//  * residual of the exact Mie currents projected onto RWG functions (fields_test_support.hpp),
//    max(|(Z x - b)_E| / |b_E|, |(Z x - b)_H| / |b_H|) (assembler_test_support.hpp; the rows
//    are normalised separately because |b_E| / |b_H| ~ eta0), decreases from icosphere n = 2
//    to n = 3 and is < 0.5 at n = 3, for PMCHWT and ICTF, n = 1.5 and Ag (a sign error would
//    leave an O(1) residual);
//  * jump-term signs of each region for Ag (single-region systems, icosphere n = 2);
//  * first dense solves of the project: bistatic RCS error eps_rr (docs/05, E_ref = sqrt(sigma),
//    37 angles in the xz-plane) at n = 3 below 0.1 and below the n = 2 value for the n = 1.5
//    sphere (PMCHWT, ICTF), below 0.2 for Ag with PMCHWT;
//  * assembly time at n = 3 (2N = 3840) on 3 threads, < 30 s in optimised builds.
// Unoptimised sanitizer build: an n = 3 assembly takes ~2.3 min and the dense LU of 2N = 3840
// ~25 min (Eigen without BLAS), so everything at n = 3 is release-only there (SKIP where nothing
// else runs); the n = 2 parts run in both builds (residual for the first combination, the Ag
// jump signs, the PMCHWT n = 1.5 solve).
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstddef>
#include <vector>

#include "assembler_test_support.hpp"

#ifdef SPECKLEBEM_HAVE_OPENMP
#include <omp.h>
#endif

using namespace assembler_test;
using formulation::Kind;

namespace {

Real solve_eps_rr(const material::Material& mat, int subdivisions, Kind kind) {
    SphereCase c(mat, subdivisions, kind);
    const SolveResult res = solve_and_compare(c);
    WARN(c.form->name() << " eps_r = " << mat.eps_r << ", icosphere " << subdivisions << " (2N = "
                        << 2 * c.size() << "): eps_rr = " << res.eps_rr << ", LU residual "
                        << res.residual << ", exact-current residual " << res.exact_residual);
    CHECK(res.residual < 1e-10);
    CHECK(res.exact_residual < 0.5);
    return res.eps_rr;
}

/// Residual of the exact Mie currents for one case; the n = 3 assembly of the first
/// combination is timed on 3 threads.
Real mie_residual(const material::Material& mat, int n, Kind kind, bool timed) {
    SphereCase c(mat, n, kind);
#ifdef SPECKLEBEM_HAVE_OPENMP
    const int saved = omp_get_max_threads();
    if (timed)
        omp_set_num_threads(3);
#endif
    const auto t0 = std::chrono::steady_clock::now();
    const MatrixXc Z = assemble(c.problem());
    const Real seconds = std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
#ifdef SPECKLEBEM_HAVE_OPENMP
    omp_set_num_threads(saved);
#endif
    if (timed) {
        WARN("dense assembly, icosphere n = " << n << " (2N = " << Z.rows()
                                              << "), 3 threads: " << seconds << " s");
#ifdef NDEBUG
        CHECK(seconds < 30.0);
#endif
    }
    const VectorXc b = op::assemble_rhs(c.problem());
    return residual(Z, c.exact(), b);
}

}  // namespace

TEST_CASE("assembler validation: residual of the exact Mie currents, icosphere n = 2, 3",
          "[validation][operator]") {
    // Measured (WP9), row-wise residual r(n = 2) -> r(n = 3): n = 1.5 0.192 -> 0.057, Ag
    // 0.135 -> 0.034 (about O(h^2), like the projection error of the currents), identical for
    // PMCHWT and ICTF (row scalings of each other, assembler.cpp). The PMCHWT n = 1.5 assembly at n
    // = 3 runs on 3 threads and is timed (< 30 s in optimised builds; measured 2.6 s release, 137 s
    // in the sanitizer build, where n = 3 is skipped).
    struct Combo {
        material::Material mat;
        Kind kind;
    };
    const std::vector<Combo> combos = {{lossless_n15(), Kind::PMCHWT},
                                       {lossless_n15(), Kind::ICTF},
                                       {material::silver_500nm(), Kind::PMCHWT},
                                       {material::silver_500nm(), Kind::ICTF}};
#ifdef NDEBUG
    for (std::size_t i = 0; i < combos.size(); ++i) {
        const Combo& cb = combos[i];
        const Real r2 = mie_residual(cb.mat, 2, cb.kind, false);
        const Real r3 = mie_residual(cb.mat, 3, cb.kind, i == 0);
        WARN(formulation::make_formulation(cb.kind)->name()
             << " eps_r = " << cb.mat.eps_r << ": residual n = 2: " << r2 << ", n = 3: " << r3);
        CHECK(r3 < r2);
        CHECK(r3 < 0.5);
    }
#else
    // Sanitizer build: first combination at n = 2 only (an n = 3 assembly takes ~2.3 min).
    const Real r2 = mie_residual(combos[0].mat, 2, combos[0].kind, false);
    WARN("PMCHWT n = 1.5: residual n = 2: " << r2 << " (n = 3 runs in optimised builds only)");
    CHECK(r2 < 0.4);
#endif
}

TEST_CASE("assembler validation: jump-term signs of each region for Ag", "[validation][operator]") {
    // Single-region systems (assembler_test_support.hpp), Ag sphere, icosphere n = 2 at 500 nm.
    // The n = 1.5 case runs as a unit test (n = 1 at 1 um). Measured (WP9): region 1 E/H rows
    // 0.100 / 0.176 (flipped 0.523 / 1.62), region 2 0.044 / 0.178 (flipped 0.541 / 1.66).
    SphereCase c(material::silver_500nm(), 2, Kind::PMCHWT);
    const JumpSignResiduals r = jump_sign_residuals(c);
    WARN("Ag, n = 2: region 1 E/H rows "
         << r.ok1.e << " / " << r.ok1.h << " (flipped " << r.flipped1.e << " / " << r.flipped1.h
         << "), region 2 " << r.ok2.e << " / " << r.ok2.h << " (flipped " << r.flipped2.e << " / "
         << r.flipped2.h << ")");
    for (const RowResiduals& ok : {r.ok1, r.ok2}) {
        CHECK(ok.e < 0.3);
        CHECK(ok.h < 0.3);
    }
    for (const RowResiduals& bad : {r.flipped1, r.flipped2}) {
        CHECK(bad.e > 0.45);
        CHECK(bad.h > 0.45);
    }
}

TEST_CASE("assembler validation: dense solve of the n = 1.5 sphere, PMCHWT and ICTF",
          "[validation][operator]") {
    // Measured (WP9): eps_rr n = 2 -> 3: PMCHWT 0.0089 -> 0.0028, ICTF identical. With the
    // jump terms cancelled, Z_ICTF = diag(2/(eta1+eta2) I, (eta1+eta2)/2 I) Z_PMCHWT and b scales
    // alike, so both give the same currents up to rounding; the formulations differ only in
    // the conditioning seen by an iterative solver (docs/03).
#ifdef NDEBUG
    for (const Kind kind : {Kind::PMCHWT, Kind::ICTF}) {
        const Real e2 = solve_eps_rr(lossless_n15(), 2, kind);
        const Real e3 = solve_eps_rr(lossless_n15(), 3, kind);
        CHECK(e2 < 0.3);
        CHECK(e3 < 0.1);
        CHECK(e3 < e2);
    }
#else
    // Sanitizer build: PMCHWT at n = 2 only (the n = 3 LU would take ~25 min).
    CHECK(solve_eps_rr(lossless_n15(), 2, Kind::PMCHWT) < 0.3);
#endif
}

TEST_CASE("assembler validation: dense PMCHWT solve of the Ag sphere", "[validation][operator]") {
#ifdef NDEBUG
    // Measured (WP9): eps_rr n = 2: 0.020, n = 3: 0.0036.
    const Real e2 = solve_eps_rr(material::silver_500nm(), 2, Kind::PMCHWT);
    const Real e3 = solve_eps_rr(material::silver_500nm(), 3, Kind::PMCHWT);
    CHECK(e3 < 0.2);
    CHECK(e3 < e2);
#else
    SKIP("dense LU of 2N = 3840 is too slow in the unoptimised build");
#endif
}
