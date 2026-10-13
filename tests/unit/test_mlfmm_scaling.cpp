// Smoke test of the MLFMM scaling study's case code (WP22b1, benchmarks/mlfmm_scaling.cpp,
// tests/support/mlfmm_scaling_support.hpp) at tiny patch sizes: geometry (box parameters from
// rough_surface_box_params, the waist rule), estimates, the describe() parser and one complete
// Ag run (setup, timed matvecs, GMRES) through Simulation with compression "mlfmm". The Si
// estimate (order searches up to L ~ 100) and the end-to-end run are checked in optimised builds
// only, so that every case stays below ~20 s under the sanitizers.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlfmm_scaling_support.hpp"

using namespace mlfmm_scaling;

namespace {

/// A file name in the temp directory unique to this process and call (ctest -j runs test
/// processes concurrently, possibly from several build trees).
std::filesystem::path unique_temp_path(const std::string& stem) {
    std::random_device rd;
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           (stem + "_" + std::to_string(rd()) + "_" + std::to_string(ticks) + ".txt");
}

}  // namespace

TEST_CASE("mlfmm_scaling: geometry follows the ADR 0006 box parameters", "[mlfmm_scaling]") {
    Case si;
    si.material = "si";
    si.L = 0.6e-6;
    const Geometry gs = make_geometry(si);
    CHECK(gs.mesh.signed_volume() > 0.0);
    REQUIRE(gs.fine_depth.has_value());  // Si: fine band 3 delta + 3 sigma
    CHECK_THAT(*gs.fine_depth,
               Catch::Matchers::WithinRel(
                   default_box_fine_depth(material::silicon_500nm(), kLambda, kSigma), 1e-12));
    CHECK_THAT(gs.depth, Catch::Matchers::WithinRel(5.6455e-6, 1e-3));
    CHECK(gs.grading.target_spacing == kBoxMeshSize);
    CHECK(gs.top_triangles == 2 * 12 * 12);
    CHECK(uses_fine_band(si));
    const RoughBoxParams bs = box_params(si);
    CHECK(bs.box_mesh_size == kBoxMeshSize);  // explicit override of the automatic rule
    CHECK_THAT(bs.exterior_wavelength, Catch::Matchers::WithinRel(kLambda, 1e-12));

    Case ag = si;
    ag.material = "ag";
    const Geometry ga = make_geometry(ag);
    CHECK_FALSE(ga.fine_depth.has_value());  // Ag: no fine band
    CHECK(ga.depth == kMinBoxDepth);
    CHECK(ga.grading.levels == 1);  // 100 nm = 2 x 50 nm
    CHECK_FALSE(uses_fine_band(ag));
    ag.fine_band = true;
    CHECK(uses_fine_band(ag));
    ag.fine_band.reset();

    // Automatic rule capped at lambda1 / 5 (rough_surface_box_params) and the uncapped rule of
    // the record's "auto" series.
    ag.box_mesh_size.reset();
    CHECK(make_geometry(ag).grading.target_spacing <= kBoxMeshSize * (1.0 + 1e-9));
    CHECK(box_option(ag) == "auto");
    ag.uncapped_box = true;
    CHECK(box_option(ag) == "uncapped");
    CHECK(make_geometry(ag).grading.target_spacing > 0.0);
    ag.box_mesh_size = kBoxMeshSize;  // uncapped needs the automatic rule
    CHECK_THROWS_AS(make_geometry(ag), std::invalid_argument);

    Case bad = si;
    bad.material = "gold";
    CHECK_THROWS_AS(make_geometry(bad), std::invalid_argument);
    bad = si;
    bad.seed = 0;
    CHECK_THROWS_AS(make_geometry(bad), std::invalid_argument);
}

TEST_CASE("mlfmm_scaling: sigma sets the surface, the fine band and the labels",
          "[mlfmm_scaling]") {
    Case c;
    c.material = "si";
    c.L = 0.6e-6;
    const Geometry g50 = make_geometry(c);
    c.sigma = 100e-9;  // the Phase 4 Si 30 x 30 um target (WP22d)
    const Geometry g100 = make_geometry(c);
    REQUIRE(g100.fine_depth.has_value());
    // Fine band 3 delta + 3 sigma: 150 nm deeper for sigma 100 nm than for 50 nm.
    CHECK_THAT(*g100.fine_depth - *g50.fine_depth, Catch::Matchers::WithinRel(150e-9, 1e-9));
    CHECK_THAT(*g100.fine_depth,
               Catch::Matchers::WithinRel(
                   default_box_fine_depth(material::silicon_500nm(), kLambda, 100e-9), 1e-12));
    // Same grid, rougher top face: same triangle count, larger height spread.
    CHECK(g100.top_triangles == g50.top_triangles);
    const auto rms_top = [](const Geometry& g) {
        // Top-face triangles: normal out of R2 (z > xi) into R1, i.e. n_z < 0 (the walls have
        // n_z ~ 0, the bottom n_z > 0). RMS of their centroid heights.
        Real sum = 0.0;
        Index count = 0;
        for (Index t = 0; t < g.mesh.num_triangles(); ++t) {
            if (g.mesh.normal(t).z() > -0.5)
                continue;
            Real z = 0.0;
            for (int k = 0; k < 3; ++k) z += g.mesh.vertices()(g.mesh.triangles()(t, k), 2) / 3.0;
            sum += z * z;
            ++count;
        }
        return std::sqrt(sum / static_cast<Real>(std::max<Index>(count, 1)));
    };
    CHECK(rms_top(g100) > 1.5 * rms_top(g50));
    CHECK(label(c).find("sigma = 100 nm") != std::string::npos);
    c.sigma = 0.0;
    CHECK_THROWS_AS(make_geometry(c), std::invalid_argument);
    c.sigma = 100e-9;
    c.box_depth = 0.0;  // a set box depth must be positive
    CHECK_THROWS_AS(make_geometry(c), std::invalid_argument);
}

TEST_CASE("mlfmm_scaling: the power summary needs the angular-spectrum beam", "[mlfmm_scaling]") {
    Case c;
    c.material = "ag";
    c.L = 0.4e-6;
    c.far_field_dtheta_deg = 5.0;  // beam paraxial: refused before any assembly
    const Geometry geo = make_geometry(c);
    CHECK_THROWS_AS(run(geo, c, Estimate{}), std::invalid_argument);
}

TEST_CASE("mlfmm_scaling: power-summary options are validated before the assembly",
          "[mlfmm_scaling]") {
    Case c;
    c.material = "ag";
    c.L = 0.4e-6;
    c.beam = Beam::angular_spectrum;
    CHECK_NOTHROW(validate_power_summary(c));  // 0 = off
    for (const Real d : {1.0, 6.0, 25.0}) {
        // 25 deg: half-node grid n_theta = round(1.8) = 2, n_phi = round(3.6) = 4.
        c.far_field_dtheta_deg = d;
        CHECK_NOTHROW(validate_power_summary(c));
    }
    // Too coarse for the half-node grid (n_phi = round(3.46) = 3) or for both; negative,
    // non-finite or absurdly fine spacings.
    for (const Real d : {26.0, 60.0, 200.0, -1.0, 1e-4, std::numeric_limits<Real>::quiet_NaN(),
                         std::numeric_limits<Real>::infinity()}) {
        c.far_field_dtheta_deg = d;
        INFO("dtheta = " << d);
        CHECK_THROWS_AS(validate_power_summary(c), std::invalid_argument);
    }
    c.far_field_dtheta_deg = 1.0;
    for (const int degree : {1, 2, 4, 6, 8, 10}) {
        c.far_field_degree = degree;
        CHECK_NOTHROW(validate_power_summary(c));
    }
    // Outside 1..20, or negative weights / points outside the triangle (3, 7, 11).
    for (const int degree : {0, -2, 21, 3, 7, 11}) {
        c.far_field_degree = degree;
        INFO("degree = " << degree);
        CHECK_THROWS_AS(validate_power_summary(c), std::invalid_argument);
    }
    c.far_field_degree = 6;
    c.beam = Beam::paraxial;  // no exact P_inc
    CHECK_THROWS_AS(validate_power_summary(c), std::invalid_argument);
    c.far_field_dtheta_deg = 0.0;
    CHECK_NOTHROW(validate_power_summary(c));

    // run() checks the options before building the Simulation (no assembly: immediate).
    c.beam = Beam::angular_spectrum;
    c.far_field_dtheta_deg = 30.0;
    const Geometry geo = make_geometry(c);
    CHECK_THROWS_AS(run(geo, c, Estimate{}), std::invalid_argument);
    c.far_field_dtheta_deg = 1.0;
    c.far_field_degree = 3;
    CHECK_THROWS_AS(run(geo, c, Estimate{}), std::invalid_argument);
}

TEST_CASE("mlfmm_scaling: a failing power summary keeps the solve result", "[mlfmm_scaling]") {
    Case c;
    c.material = "si";
    c.beam = Beam::angular_spectrum;
    c.far_field_dtheta_deg = 1.0;
    Result r;
    r.est.triangles = 300;
    r.est.top_triangles = 200;
    r.est.unknowns = 420;
    r.iterations = 7;
    r.converged = true;
    r.solve_s = 12.5;

    run_power_summary(
        [&] {
            r.p_inc = 2.0;
            r.p_refl = 0.7;
            r.p_refl_coarse = 0.7;
            r.p_abs = 1.2;
            r.post_s = 3.0;
        },
        r);
    CHECK(r.post_error.empty());
    CHECK(r.p_inc == 2.0);
    std::string line = row(c, r);
    CHECK(line.find(" R=0.35 ") != std::string::npos);
    CHECK(line.find(" A=0.6 ") != std::string::npos);
    CHECK(line.find("post_error") == std::string::npos);

    // A throw after a partial result: the partial power fields are discarded, the solve fields
    // stay, and the ROW line is still produced, without R / A and with post_error=1.
    run_power_summary(
        [&] {
            r.p_inc = 2.0;
            r.p_refl = 0.5;
            throw std::runtime_error("far field failed");
        },
        r);
    CHECK(r.post_error == "far field failed");
    CHECK(r.p_inc == 0.0);
    CHECK(r.p_refl == 0.0);
    CHECK(r.p_abs == 0.0);
    CHECK(r.post_s == 0.0);
    CHECK(r.iterations == 7);
    CHECK(r.solve_s == 12.5);
    line = row(c, r);
    CHECK(line.rfind("ROW material=si", 0) == 0);
    CHECK(line.find(" it=7 ") != std::string::npos);
    CHECK(line.find(" conv=1 ") != std::string::npos);
    CHECK(line.find(" R=") == std::string::npos);
    CHECK(line.find(" A=") == std::string::npos);
    CHECK(line.size() >= 12);
    CHECK(line.compare(line.size() - 12, 12, "post_error=1") == 0);
    CHECK(summary(r).find("post-processing failed after the solve: far field failed") !=
          std::string::npos);

    Result r2;
    run_power_summary([] { throw 42; }, r2);  // not a std::exception
    CHECK(r2.post_error == "non-standard exception");
}

TEST_CASE("mlfmm_scaling: the waist rule (L/4 Ag, L/6 Si) is enforced", "[mlfmm_scaling]") {
    Case c;
    c.material = "ag";
    c.L = 0.4e-6;
    c.waist_factor = 3.0;  // w0 = L/3 > L/4
    CHECK_THROWS_AS(make_geometry(c), std::invalid_argument);
    c.allow_wide_beam = true;  // deliberate edge-effect study: allowed (logged)
    CHECK_NOTHROW(make_geometry(c));
    c.allow_wide_beam = false;
    c.waist_factor = 4.0;
    CHECK_NOTHROW(make_geometry(c));
    c.waist_factor.reset();  // default: 1 / beam_waist_fraction = 4 for Ag
    CHECK(waist_factor_of(c) == 4.0);
    CHECK(label(c).find("w0 = L/4,") != std::string::npos);

    // Si (weakly absorbing): w0 <= L / 6 since the ADR 0006 amendment 2026-10-12b.
    c.material = "si";
    c.L = 0.6e-6;
    CHECK(waist_fraction(c) == 1.0 / 6.0);
    CHECK(waist_factor_of(c) == 6.0);
    CHECK(label(c).find("w0 = L/6,") != std::string::npos);
    CHECK_NOTHROW(make_geometry(c));
    c.waist_factor = 4.0;  // the recorded WP22b/WP22d Si rows: needs allow_wide_beam now
    CHECK_THROWS_AS(make_geometry(c), std::invalid_argument);
    c.allow_wide_beam = true;
    CHECK_NOTHROW(make_geometry(c));
    c.waist_factor = 0.0;
    CHECK_THROWS_AS(make_geometry(c), std::invalid_argument);
}

TEST_CASE("mlfmm_scaling: the angular-spectrum beam covers the mesh", "[mlfmm_scaling]") {
    Case c;
    c.material = "ag";
    c.L = 0.4e-6;
    c.beam = Beam::angular_spectrum;
    const Geometry geo = make_geometry(c);
    const auto beam = make_beam(c, geo);
    const Real rmax = geo.mesh.vertices().rowwise().norm().maxCoeff();
    CHECK(beam->controlled_radius() >= rmax);
    CHECK(beam_name(c) == "angular_spectrum");
    // E(focus) . x = 1 V/m, E along x at normal incidence.
    const Vec3c e0 = beam->electric_field(Vec3::Zero());
    CHECK_THAT(e0.x().real(), Catch::Matchers::WithinAbs(1.0, 1e-8));
    CHECK(std::abs(e0.y()) < 1e-8);
    c.beam = Beam::paraxial;
    CHECK(std::isinf(make_beam(c, geo)->controlled_radius()));
}

TEST_CASE("mlfmm_scaling: estimate respects the ADR 0008 leaf rule", "[mlfmm_scaling]") {
    std::vector<std::string> materials = {"ag"};
#ifdef NDEBUG
    materials.emplace_back("si");  // the Si order searches (L up to ~110) are slow unoptimised
    const Real patch = 0.8e-6;
#else
    const Real patch = 0.4e-6;  // sanitizer build: smaller patch, same octree depth
#endif
    for (const std::string& m : materials) {
        Case c;
        c.material = m;
        c.L = patch;
        const Geometry geo = make_geometry(c);
        const Estimate e = estimate(geo, c);
        CHECK(e.unknowns == 2 * geo.mesh.num_edges());
        CHECK(e.leaf_edge >= (1.0 - mlfmm::kMinBoxSizeTolerance) * 0.25 * kLambda);
        CHECK(e.max_support_radius / e.leaf_edge <= mlfmm::kLeafMaxSupportRatioD3 + 1e-12);
        CHECK(e.near_bytes > 0);
        CHECK(e.exact_budget >= e.near_bytes);
        // R1 always expands at the leaf; the far tables come with the patterns.
        CHECK(e.leaf_order[0] > 0);
        CHECK(e.expansion_levels[0] >= 1);
        CHECK(e.pattern_bytes > 0);
        CHECK(e.far_table_bytes > 0);
        CHECK(e.peak_setup_bytes > kPeakNearFactor * static_cast<Real>(e.near_bytes) +
                                       static_cast<Real>(e.pattern_bytes + e.far_table_bytes));
        if (m == "si") {
            CHECK(e.exact_pairs == 0);  // Si: x* / alpha = 13 um, no exact fallback estimate
            CHECK(e.leaf_order[1] > e.leaf_order[0]);  // the Si interior expands (k2 = 4.3 k1)
        } else {
            CHECK(e.leaf_order[1] == 0);  // Ag interior: no expansion (exact / truncated)
            CHECK(e.expansion_levels[1] == 0);
        }
    }
}

TEST_CASE("mlfmm_scaling: exact-pair estimate counts pairs within the decay reach",
          "[mlfmm_scaling]") {
    // A small rough patch; a strongly decaying k (short reach x*/alpha) gives fewer pairs than a
    // weaker one, and no decay gives none.
    Case c;
    c.material = "ag";
    c.L = 0.3e-6;
    const Geometry geo = make_geometry(c);
    const basis::RwgSpace space(geo.mesh);
    const Real k0 = 2.0 * constants::pi / kLambda;
    CHECK(estimate_exact_pairs(space, Complex(k0, 0.0), 3.0, 0) == 0);
    const Index tight = estimate_exact_pairs(space, Complex(k0, -200e6), 3.0, 0);  // 59 nm
    const Index wide = estimate_exact_pairs(space, Complex(k0, -80e6), 3.0, 0);    // 147 nm
    CHECK(tight >= space.size());  // every basis with itself (gap 0)
    CHECK(wide > tight);
    CHECK(estimate_exact_pairs(space, Complex(k0, -80e6), 3.0, wide) == 0);
}

TEST_CASE("mlfmm_scaling: number_after parses describe() and refuses missing markers",
          "[mlfmm_scaling]") {
    const std::string text = "octree built in 0.25 s\n  far: 3.5 s setup\n  near assembled in x";
    CHECK(number_after(text, "built in ") == 0.25);
    CHECK(number_after(text, "\n  far: ") == 3.5);
    CHECK_THROWS_AS(number_after(text, "missing marker "), std::runtime_error);
    CHECK_THROWS_AS(number_after(text, "assembled in "), std::runtime_error);  // no number
}

TEST_CASE("mlfmm_scaling: one tiny Ag case runs end to end", "[mlfmm_scaling]") {
#ifndef NDEBUG
    SKIP("unoptimised build: the end-to-end MLFMM run is checked in release");
#else
    Case c;
    c.material = "ag";
    c.L = 0.6e-6;
    c.matvecs = 3;
    c.tolerance = 1e-1;  // smoke test: the study itself uses 1e-3
    c.max_iter = 200;
    const Geometry geo = make_geometry(c);
    const Result r = run(geo, c, estimate(geo, c));
    CHECK(r.formulation == "ICTF");
    CHECK(r.jacobi);
    CHECK(r.converged);
    CHECK(r.true_residual < 0.2);
    CHECK(r.est.levels >= 3);
    CHECK(r.matvec_s > 0.0);
    CHECK(r.near_matvec_s > 0.0);
    CHECK(r.far_matvec_s > 0.0);
    CHECK(r.near_s > 0.0);
    CHECK(r.far_s > 0.0);
    CHECK(r.assembly_s >= r.near_s);
    CHECK(r.near_bytes > 0);
    CHECK(r.far_bytes > 0);
    CHECK(r.krylov_bytes > 0.0);
    CHECK(r.peak_rss >= r.peak_setup_rss);
    // The far-memory model (patterns + tables) against the operator (block check not
    // modelled: the model may only be larger).
    CHECK(static_cast<Real>(r.est.pattern_bytes + r.est.far_table_bytes) >=
          0.9 * static_cast<Real>(r.far_bytes));
    const std::string line = row(c, r);
    CHECK(line.rfind("ROW material=ag", 0) == 0);
    CHECK(line.find(" N2=" + std::to_string(r.est.unknowns) + " ") != std::string::npos);
    CHECK(line.find(" beam=paraxial ") != std::string::npos);
    CHECK(line.find(" sigma_nm=50 ") != std::string::npos);
    CHECK(line.find(" form=ictf jacobi=1") != std::string::npos);  // formulation::recommend
    CHECK(line.find(" R=") == std::string::npos);                  // no power summary requested
    CHECK(r.p_inc == 0.0);
#endif
}

TEST_CASE("mlfmm_scaling: iteration log and power summary of a tiny Ag case", "[mlfmm_scaling]") {
#ifndef NDEBUG
    SKIP("unoptimised build: the end-to-end MLFMM run is checked in release");
#else
    // Ag (the Si box is 5.65 um deep), rigorous beam, power summary on a coarse hemisphere grid.
    // Kept small for a fast smoke test of the code path (672 unknowns, ~1 s; the far operator is
    // exercised by the case above): 100 nm mesh with the uniform 100 nm box, box depth 0.5 um
    // (20 decay lengths of Ag, instead of the 2 um default: the near-field assembly of the walls
    // dominated the cost), waist L / 2 (allow_wide_beam) and tolerance 1e-1. The beam lights
    // the walls, so only physical bounds are checked, not the reflectance of a large patch
    // (benchmarks/results/si_large_area.md).
    Case c;
    c.material = "ag";
    c.L = 0.4e-6;
    c.mesh_size = 100e-9;
    c.box_depth = 0.5e-6;
    c.waist_factor = 2.0;
    c.allow_wide_beam = true;
    c.matvecs = 1;
    c.tolerance = 1e-1;
    c.max_iter = 300;
    c.beam = Beam::angular_spectrum;
    c.far_field_dtheta_deg = 15.0;
    c.far_field_degree = 2;
    const Geometry geo = make_geometry(c);
    CHECK_THAT(geo.depth, Catch::Matchers::WithinRel(0.5e-6, 1e-12));
    std::vector<int> its;
    std::vector<Real> res;
    std::vector<Real> times;
    const Result r = run(geo, c, estimate(geo, c), [&](int k, Real rr, Real t) {
        its.push_back(k);
        res.push_back(rr);
        times.push_back(t);
    });
    REQUIRE(r.converged);
    REQUIRE(static_cast<int>(its.size()) == r.iterations);
    // The kept residual history (WP22b2 history records) is r_0 = 1 plus the logged residuals.
    REQUIRE(r.residual_history.size() == its.size() + 1);
    CHECK_THAT(r.residual_history.front(), Catch::Matchers::WithinRel(1.0, 1e-12));
    for (std::size_t i = 0; i < its.size(); ++i) {
        CHECK(r.residual_history[i + 1] == res[i]);
        CHECK(its[i] == static_cast<int>(i) + 1);
        if (i > 0)
            CHECK(times[i] >= times[i - 1]);
    }
    CHECK(res.back() <= c.tolerance);
    CHECK(times.back() <= r.solve_s + 1e-3);
    CHECK(r.p_inc > 0.0);
    const Real R = r.p_refl / r.p_inc;
    const Real A = r.p_abs / r.p_inc;
    INFO("R = " << R << ", A = " << A << ", R_coarse = " << r.p_refl_coarse / r.p_inc);
    // Flat Ag at normal incidence: R = |(1 - n) / (1 + n)|^2 = 0.98; the rough tiny box scatters
    // part of the beam sideways (walls, forward hemisphere). Bounds only.
    CHECK(R > 0.4);
    CHECK(R < 1.05);
    CHECK(A > 0.0);
    CHECK(A < 0.3);
    CHECK(R + A < 1.1);
    CHECK_THAT(r.p_refl_coarse, Catch::Matchers::WithinRel(r.p_refl, 0.1));
    CHECK(r.post_s > 0.0);
    const std::string line = row(c, r);
    CHECK(line.find(" R=") != std::string::npos);
    CHECK(line.find(" A=") != std::string::npos);
    CHECK(line.find(" beam=angular_spectrum ") != std::string::npos);
    CHECK(line.find(" depth_um=0.5 ") != std::string::npos);  // non-default depth recorded
    CHECK(line.find("post_error") == std::string::npos);
    CHECK(label(c).find(", depth 0.5") != std::string::npos);
#endif
}

TEST_CASE("mlfmm_scaling: option defaults (recorded box and solver, material waist rule)",
          "[mlfmm_scaling]") {
    const Options o = parse_options({});
    CHECK(o.c.material == "si");
    // Waist: the material's limit (WP-B3: Si L/6, Ag L/4), not the recorded series' L/4.
    CHECK_FALSE(o.c.waist_factor.has_value());
    CHECK_FALSE(o.c.allow_wide_beam);
    CHECK(waist_factor_of(o.c) == 6.0);
    REQUIRE(o.c.box_mesh_size.has_value());
    CHECK(*o.c.box_mesh_size == kBoxMeshSize);
    CHECK(o.c.beam == Beam::paraxial);
    CHECK(o.c.digits == 3.0);
    CHECK(o.c.tolerance == 1e-3);
    CHECK(o.c.max_iter == 3000);
    CHECK(o.c.restart == 0);
    CHECK_FALSE(o.c.formulation.has_value());  // formulation::recommend
    CHECK_FALSE(o.c.jacobi.has_value());
    CHECK(o.c.solve);
    CHECK_FALSE(o.estimate_only);
    CHECK_FALSE(o.force);
    CHECK_FALSE(o.history);
    CHECK(o.history_file.empty());
    CHECK(o.history_full_file.empty());
    CHECK(o.history_stride == 10);
    CHECK(o.history_digits == 4);
}

TEST_CASE("mlfmm_scaling: options of the WP22b2 formulation sweep", "[mlfmm_scaling]") {
    const Options o = parse_options({"--material",
                                     "ag",
                                     "--L",
                                     "3.2e-6",
                                     "--box-mesh-size",
                                     "50e-9",
                                     "--beam",
                                     "angular-spectrum",
                                     "--formulation",
                                     "pmchwt",
                                     "--jacobi",
                                     "0",
                                     "--max-iter",
                                     "4000",
                                     "--history-file",
                                     "h.txt",
                                     "--history-stride",
                                     "5",
                                     "--history-digits",
                                     "6",
                                     "--history-full-file",
                                     "full.txt"});
    CHECK(o.c.material == "ag");
    CHECK(o.c.L == 3.2e-6);
    REQUIRE(o.c.box_mesh_size.has_value());
    CHECK(*o.c.box_mesh_size == 50e-9);
    CHECK(o.c.beam == Beam::angular_spectrum);
    REQUIRE(o.c.formulation.has_value());
    CHECK(*o.c.formulation == formulation::Kind::PMCHWT);
    REQUIRE(o.c.jacobi.has_value());
    CHECK_FALSE(*o.c.jacobi);
    CHECK(o.c.max_iter == 4000);
    CHECK(o.history_file == "h.txt");
    CHECK(o.history_stride == 5);
    CHECK(o.history_digits == 6);
    CHECK(o.history_full_file == "full.txt");

    const Options m = parse_options({"--formulation", "MCTF", "--jacobi", "1"});
    REQUIRE(m.c.formulation.has_value());
    CHECK(*m.c.formulation == formulation::Kind::MCTF);
    REQUIRE(m.c.jacobi.has_value());
    CHECK(*m.c.jacobi);
    const Options i = parse_options({"--formulation", "ictf"});
    REQUIRE(i.c.formulation.has_value());
    CHECK(*i.c.formulation == formulation::Kind::ICTF);
    // Box options of the record (unchanged).
    CHECK_FALSE(parse_options({"--box-mesh-size", "auto"}).c.box_mesh_size.has_value());
    CHECK(parse_options({"--box-mesh-size", "uncapped"}).c.uncapped_box);
}

TEST_CASE("mlfmm_scaling: the WP22b2 Si waist pin parses (w0 = L/4 past the WP-B3 default)",
          "[mlfmm_scaling]") {
    const Options o =
        parse_options({"--material", "si", "--waist-factor", "4", "--allow-wide-beam"});
    REQUIRE(o.c.waist_factor.has_value());
    CHECK(waist_factor_of(o.c) == 4.0);
    CHECK(o.c.allow_wide_beam);
    CHECK(row(o.c, Result{}).find(" waist=4 ") != std::string::npos);
    // Without --allow-wide-beam the pinned Si waist parses but the case is refused by the waist
    // rule at make_geometry (before the assembly).
    const Options narrow = parse_options({"--material", "si", "--waist-factor", "4"});
    CHECK_FALSE(narrow.c.allow_wide_beam);
    CHECK_THROWS_AS(make_geometry(narrow.c), std::invalid_argument);
}

TEST_CASE("mlfmm_scaling: invalid options are refused", "[mlfmm_scaling]") {
    using V = std::vector<std::string>;
    const std::vector<V> bad = {
        V{"--formulation", "efie"}, V{"--jacobi", "2"}, V{"--jacobi", "yes"},
        V{"--history-stride", "0"}, V{"--history-digits", "0"}, V{"--history-digits", "18"},
        V{"--no-solve", "--history-file", "h.txt"}, V{"--no-solve", "--history-full-file", "h.txt"},
        V{"--matvecs", "0"}, V{"--L"}, V{"--history-file"}, V{"--unknown"}, V{"--beam", "plane"},
        V{"--exact-budget-gb", "0"}, V{"--far-field", "1"},  // needs the angular-spectrum beam
        // Numeric values are consumed whole (no stoi/stod prefix).
        V{"--history-stride", "5x"}, V{"--history-digits", "4.5"}, V{"--max-iter", "4000 "},
        V{"--L", "3.2e-6m"}, V{"--waist-factor", "4x"}, V{"--seed", "-1"}, V{"--seed", "1.5"},
        V{"--fine-band", "1x"}, V{"--tol", ""}, V{"--tol", "abc"},
        // Leading whitespace and non-finite values.
        V{"--max-iter", " 5"}, V{"--L", " 3e-6"}, V{"--exact-budget-gb", "inf"}, V{"--tol", "nan"},
        // Out of range: std::out_of_range becomes std::invalid_argument.
        V{"--max-iter", "99999999999"}, V{"--L", "1e999"}, V{"--seed", "99999999999999999999999"},
        // History records need a solve.
        V{"--estimate-only", "--history-file", "h.txt"},
        V{"--estimate-only", "--history-full-file", "h.txt"},
        V{"--history-file", "h.txt", "--estimate-only"}};
    for (const V& args : bad) {
        std::string joined;
        for (const std::string& t : args) joined += "'" + t + "' ";
        INFO(joined);
        CHECK_THROWS_AS(parse_options(args), std::invalid_argument);
    }
}

TEST_CASE("mlfmm_scaling: history indices every stride plus the last", "[mlfmm_scaling]") {
    CHECK(history_indices(25, 10) == std::vector<int>{0, 10, 20, 25});
    CHECK(history_indices(20, 10) == std::vector<int>{0, 10, 20});
    CHECK(history_indices(3, 10) == std::vector<int>{0, 3});
    CHECK(history_indices(0, 10) == std::vector<int>{0});
    CHECK(history_indices(4, 1) == std::vector<int>{0, 1, 2, 3, 4});
    CHECK(history_indices(-1, 10).empty());
    CHECK_THROWS_AS(history_indices(5, 0), std::invalid_argument);
}

TEST_CASE("mlfmm_scaling: history record and appended history file", "[mlfmm_scaling]") {
    Case c;
    c.material = "ag";
    c.L = 3.2e-6;
    c.box_mesh_size = 50e-9;
    c.beam = Beam::angular_spectrum;
    Result r;
    r.formulation = "PMCHWT";
    r.jacobi = false;
    r.iterations = 12;
    r.converged = false;
    r.est.unknowns = 1234;
    for (int k = 0; k <= r.iterations; ++k)
        r.residual_history.push_back(std::ldexp(1.0, -k) / 3.0);  // no rounding ties
    CHECK(formulation_key(r) == "pmchwt");
    const std::string h = history_record(c, r, 5, 3);
    CHECK(h.rfind("HISTORY material=ag L_um=3.2 box=", 0) == 0);
    // Ag default waist L/4 (waist_factor_of) and the iteration cap (review round keys).
    CHECK(h.find(" fine=0 waist=4 beam=angular_spectrum ") != std::string::npos);
    CHECK(h.find(" tol=0.001 max_iter=3000 seed=1 ") != std::string::npos);
    CHECK(h.find(" N2=1234 form=pmchwt jacobi=0 it=12 conv=0 stride=5 ") != std::string::npos);
    // k = 0, 5, 10, 12: 2^-k / 3 with 3 significant digits.
    CHECK(h.substr(h.find(" res=")) == " res=3.33e-01,1.04e-02,3.26e-04,8.14e-05");
    // 17 digits: every iteration, and each value reads back exactly.
    const std::string full = history_record(c, r, 1, 17);
    std::stringstream values(full.substr(full.find(" res=") + 5));
    std::string token;
    std::size_t k = 0;
    while (std::getline(values, token, ',')) {
        REQUIRE(k < r.residual_history.size());
        CHECK(std::stod(token) == r.residual_history[k]);
        ++k;
    }
    CHECK(k == r.residual_history.size());
    CHECK(row(c, r).find(" form=pmchwt jacobi=0") != std::string::npos);
    CHECK(formulation_key(Result{}) == "unknown");

    CHECK_THROWS_AS(history_record(c, r, 0, 3), std::invalid_argument);
    CHECK_THROWS_AS(history_record(c, r, 5, 0), std::invalid_argument);
    CHECK_THROWS_AS(history_record(c, r, 5, 18), std::invalid_argument);
    Result short_history = r;
    short_history.residual_history.pop_back();
    CHECK_THROWS_AS(history_record(c, short_history, 5, 3), std::invalid_argument);

    Case pinned = c;  // a Si-style explicit waist is recorded as given
    pinned.waist_factor = 6.0;
    pinned.max_iter = 450;
    const std::string hp = history_record(pinned, r, 5, 3);
    CHECK(hp.find(" waist=6 ") != std::string::npos);
    CHECK(hp.find(" max_iter=450 ") != std::string::npos);

    const std::filesystem::path p = unique_temp_path("specklebem_test_mlfmm_scaling_history");
    std::filesystem::remove(p);
    append_line(core::path_to_utf8(p), h);
    append_line(core::path_to_utf8(p), full);
    {
        std::ifstream in(p, std::ios::binary);
        std::string l1;
        std::string l2;
        std::string l3;
        REQUIRE(std::getline(in, l1));
        REQUIRE(std::getline(in, l2));
        CHECK(l1 == h);
        CHECK(l2 == full);
        CHECK_FALSE(std::getline(in, l3));
    }
    std::filesystem::remove(p);
    CHECK_THROWS_AS(append_line(core::path_to_utf8(p / "no_such_dir" / "x.txt"), h),
                    std::runtime_error);
}

TEST_CASE("mlfmm_scaling: history files are probed for appending before the solve",
          "[mlfmm_scaling]") {
    Options o;
    CHECK_NOTHROW(probe_history_files(o));  // no history files: nothing to probe
    const std::filesystem::path p = unique_temp_path("specklebem_test_mlfmm_scaling_probe");
    std::filesystem::remove(p);
    o.history_file = core::path_to_utf8(p);
    CHECK_NOTHROW(probe_history_files(o));
    CHECK(std::filesystem::exists(p));  // created empty, appended to after the solve
    CHECK(std::filesystem::file_size(p) == 0);
    append_line(o.history_file, "HISTORY x");
    CHECK_NOTHROW(probe_history_files(o));  // an existing file is kept, not truncated
    CHECK(std::filesystem::file_size(p) == std::string("HISTORY x\n").size());
    std::filesystem::remove(p);

    Options bad;
    bad.history_full_file = core::path_to_utf8(p / "no_such_dir" / "full.txt");
    CHECK_THROWS_AS(probe_history_files(bad), std::invalid_argument);
    bad.history_full_file.clear();
    bad.history_file = core::path_to_utf8(p / "no_such_dir" / "h.txt");
    CHECK_THROWS_AS(probe_history_files(bad), std::invalid_argument);
    CHECK_FALSE(std::filesystem::exists(p));
}
