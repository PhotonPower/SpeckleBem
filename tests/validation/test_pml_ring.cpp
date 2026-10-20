// Validation 7h of the PML route at CI size (WP-PML5a, through the ACA far ring of WP-PML5b; ADR
// 0008 amendment 2026-10-18 decision 6, docs/05 "PML route" row 7h): the fast path (compression
// "mlfmm" on the PML route: MLFMM on the real functions plus the PML ring, near exact, far ACA)
// against the dense PML matrix on flat Ag:
//  * matvec at d0 = 3 and 5 on a 2N = 2816 surface (< 1e-3 / < 1e-5);
//  * the GMRES solve against the dense LU solve on a 7a-type case (pml_ring::flat_ag_ci_case():
//    normal incidence, w0 = lambda, h = lambda / 4, 2N = 4592): matvec < 1e-3 at d0 = 3, R within
//    1e-4 relative, windowed far field L2 (ADR 0006 metric) < 1e-3.
// Criteria of docs/05, not loosened. The 7c-size case (rough Ag, 2N = 22 816) is
// tests/validation_large/test_pml_ring_large.cpp. Release only (the stretched dense assembly takes
// minutes under the sanitizers).
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

#include "pml_ring_support.hpp"

TEST_CASE("pml_ring 7h: flat Ag fast path vs dense LU (CI size)", "[pml_ring_7h]") {
#ifndef NDEBUG
    SKIP("dense stretched assembly of 2N = 4592: release only");
#else
    using namespace pml_ring;
    const RoughCase c = flat_ag_ci_case();
    const FastSettings f;  // d0 = 3, GMRES tolerance 1e-7
    const FastRun fr = run_fast(c, f);
    const DenseRun dr = run_dense(c, fr);
    WARN(row_line(c, f, fr, dr));
    const pml_rough_ag::Metrics m = pml_rough_ag::compare(fr.far, dr.far);
    CHECK(fr.converged);
    CHECK(dr.residual < 1e-8);
    CHECK(fr.stretched > 0);
    CHECK(fr.levels >= 3);  // the MLFMM has a far part
    CHECK(fr.ring_far_pairs > 0);
    // WP-PML5b: through the ACA far ring (default).
    CHECK(fr.far_stats.aca);
    CHECK(fr.far_stats.aca_blocks > 0);
    // Validation 7h (docs/05; not loosened).
    CHECK(max_of(dr.matvec_errors) < 1e-3);
    CHECK(std::abs(m.dR) < 1e-4);
    CHECK(m.l2 < 1e-3);
#endif
}

TEST_CASE("pml_ring 7h: matvec vs the dense PML matrix at d0 = 3 and 5 (flat Ag, 2N = 2816)",
          "[pml_ring_7h]") {
#ifndef NDEBUG
    SKIP("dense stretched assembly of 2N = 2816: release only");
#else
    // Flat open surface, h = 125 nm: patch 1, taper 0.25, gap 0.25, PML 0.375 um (22 x 22 cells,
    // 968 triangles, 2N = 2816), vacuum over Ag, sigma_max 1 (a cheaper stretched touching rule
    // than the default 4; the comparison does not depend on the PML quality), m 2. Accurate
    // kernels (target 1e-7) so that the dense real-real far entries are an oracle for the
    // expansion at d0 = 5; the ring is exact, the difference is the real-real MLFMM far part.
    using namespace specklebem;
    geometry::OpenSurfaceParams op;
    op.patch_size = 1.0e-6;
    op.taper_width = 0.25e-6;
    op.gap_width = 0.25e-6;
    op.pml_width = 0.375e-6;
    const geometry::OpenSurfaceMesh om = geometry::make_open_surface_mesh(
        fresnel_flat::flat_height_map(op.patch_size + 2.0 * op.taper_width, 125e-9), op);
    const basis::RwgSpace space(om.mesh);
    const geometry::CoordinateStretch pml = geometry::make_pml_stretch(om, 1.0);
    const material::Material ag = material::silver_500nm();
    const auto form = formulation::make_formulation(formulation::recommend(ag.eps_r).kind);
    op::Problem p;
    p.space = &space;
    p.exterior = material::vacuum();
    p.object = ag;
    p.formulation = form.get();
    p.omega = 2.0 * constants::pi * constants::c0 / pml_rough_ag::kLambda;
    p.kernel_options.target_accuracy = 1e-7;
    p.stretch = &pml;
    const Index N = space.size();
    REQUIRE(2 * N == 2816);
    auto t0 = std::chrono::steady_clock::now();
    const auto D = op::DenseStrategy().build(p);
    const Real dense_s = pml_ring::seconds(t0);
    std::vector<VectorXc> xs, yd;
    for (const std::uint64_t seed : pml_ring::kSeeds) {
        xs.push_back(pml_ring::random_vector(2 * N, seed));
        yd.push_back(*D * xs.back());
    }
    for (const Real d0 : {3.0, 5.0}) {
        mlfmm::MlfmmParams mp;
        mp.accuracy_digits = d0;
        t0 = std::chrono::steady_clock::now();
        const mlfmm::MlfmmOperator Z(p, mp);
        const Real fast_s = pml_ring::seconds(t0);
        REQUIRE(Z.octree().levels() >= 3);
        REQUIRE(Z.ring_operator() != nullptr);
        Real err = 0;
        for (std::size_t i = 0; i < xs.size(); ++i)
            err = std::max(err, (Z * xs[i] - yd[i]).norm() / yd[i].norm());
        const mlfmm::PmlRingOperator& ring = *Z.ring_operator();
        WARN("pml_ring 7h flat Ag 2N = "
             << 2 * N << ", d0 = " << d0 << ": matvec error " << err << ", " << Z.octree().levels()
             << " levels, stretched " << ring.stretched_count() << " of " << N << ", ring near "
             << static_cast<Real>(ring.near_bytes()) / 1e6 << " MB, far "
             << static_cast<Real>(ring.far_bytes()) / 1e6 << " MB, total "
             << static_cast<Real>(Z.memory_bytes()) / 1e6 << " MB vs dense "
             << 64.0 * static_cast<Real>(N) * static_cast<Real>(N) / 1e6 << " MB; setup " << fast_s
             << " s (ring near " << ring.near_seconds() << " s, far " << ring.far_seconds()
             << " s) vs dense " << dense_s << " s; " << pml_ring::far_segment(ring.far_stats()));
        // WP-PML5b: through the ACA far ring (default).
        CHECK(ring.far_stats().aca);
        CHECK(ring.far_stats().aca_blocks > 0);
        // Validation 7h (docs/05; not loosened).
        CHECK(err < (d0 == 3.0 ? 1e-3 : 1e-5));
    }
#endif
}
