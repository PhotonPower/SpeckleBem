// Unit tests of the support pieces of validation 7e (WP-PML4c, tests/support/
// pml_si_crosscheck_support.hpp): the subsampled height map, the bitwise height identity of the
// open mesh's patch and the box top face on the common grid and of their triangles (oriented
// vertex triples; also at the three configurations of the record solved on both routes, meshing
// only), the comparison metric on synthetic far fields (fixed seeds), the 7e verdict, the memory
// guard and the stored-run round trip. The solves themselves are the validation-large test
// (test_pml_si_crosscheck_large.cpp).
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include "pml_si_crosscheck_support.hpp"

using namespace pml_si_crosscheck;
using Catch::Approx;

namespace {

long long process_id() {
#ifdef _WIN32
    return static_cast<long long>(_getpid());
#else
    return static_cast<long long>(getpid());
#endif
}

/// A small case: 2.5 um map generated at lambda / 12, meshed at lambda / 4, L = 1 um.
Case small_case() {
    Case c;
    c.map_size = 2.5e-6;
    c.gen_spacing = kLambda / 12.0;
    c.mesh_size = kLambda / 4.0;
    c.L = 1.0e-6;
    c.waist = c.L / 6.0;
    c.taper = 0.5e-6;
    c.gap = 0.25e-6;
    c.pml_width = 0.25e-6;
    return c;
}

/// Random far field on the box-validity grids (positive intensities), fixed seed.
box_validity::FarField random_far_field(std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<Real> u(0.1, 2.0);
    box_validity::FarField f;
    f.refl.resize(box_validity::kHemiTheta, box_validity::kHemiPhi);
    f.cut_xz.resize(box_validity::kCut, 1);
    f.cut_yz.resize(box_validity::kCut, 1);
    for (Index i = 0; i < f.refl.size(); ++i) f.refl.data()[i] = u(rng);
    for (Index i = 0; i < f.cut_xz.size(); ++i) f.cut_xz.data()[i] = u(rng);
    for (Index i = 0; i < f.cut_yz.size(); ++i) f.cut_yz.data()[i] = u(rng);
    return f;
}

}  // namespace

TEST_CASE("pml_si_crosscheck: subsampling keeps the fine nodes on the same centred grid",
          "[pml_si_crosscheck]") {
    const Case c = small_case();
    const geometry::HeightMap fine = generated_map(c);
    REQUIRE(fine.z.rows() == 61);  // 2.5 um / (lambda / 12) = 60 cells
    const Index k = subsample_factor(fine, c.mesh_size);
    CHECK(k == 3);
    const geometry::HeightMap s = subsample(fine, k);
    REQUIRE(s.z.rows() == 21);
    REQUIRE(s.z.cols() == 21);
    CHECK(s.dx == 3.0 * fine.dx);
    CHECK(s.dy == s.dx);
    for (Index i = 0; i < s.z.rows(); ++i)
        for (Index j = 0; j < s.z.cols(); ++j) CHECK(s.z(i, j) == fine.z(3 * i, 3 * j));
    // Node i of the subsampled grid is node 3 i of the fine grid (x_i = -(n - 1) dx / 2 + i dx).
    const Real xs = -0.5 * static_cast<Real>(s.z.rows() - 1) * s.dx + 7.0 * s.dx;
    const Real xf = -0.5 * static_cast<Real>(fine.z.rows() - 1) * fine.dx + 21.0 * fine.dx;
    CHECK(std::abs(xs - xf) < 1e-15);
    CHECK(mesh_map(c).z == s.z);  // generation is deterministic for a fixed seed
    // h must be an integer multiple of the generation spacing; n - 1 divisible by k.
    CHECK_THROWS_AS(subsample_factor(fine, 100e-9), std::invalid_argument);
    CHECK_THROWS_AS(subsample(fine, 7), std::invalid_argument);
    CHECK_THROWS_AS(subsample(fine, 0), std::invalid_argument);
    // Central crop on the subsampled grid.
    const geometry::HeightMap top = crop(s, c.L);
    REQUIRE(top.z.rows() == 9);
    CHECK(top.z(0, 0) == s.z(6, 6));
    CHECK(top.z(8, 8) == s.z(14, 14));
    CHECK_THROWS_AS(crop(s, 1.1e-6), std::invalid_argument);    // not a sub-grid
    CHECK_THROWS_AS(crop(s, 1.125e-6), std::invalid_argument);  // odd margin
}

TEST_CASE("pml_si_crosscheck: open-mesh patch heights are bitwise the box top face",
          "[pml_si_crosscheck]") {
    const Case c = small_case();
    const geometry::HeightMap map = mesh_map(c);
    const geometry::OpenSurfaceMesh om = make_open_mesh(map, c);
    CHECK(om.patch_half == Approx(0.5e-6).epsilon(1e-12));
    CHECK(om.taper_end == Approx(1.0e-6).epsilon(1e-12));
    const BoxGeometry box = make_box(map, c);
    REQUIRE(box.top.z.rows() == 9);
    CHECK(box.top_triangles == 128);
    const HeightIdentity id = height_identity(box.top, om.mesh, box.mesh);
    CHECK(id.grid_nodes == 81);
    CHECK(id.open_nodes == 81);
    CHECK(id.box_nodes == 81);
    CHECK(id.open_vs_map == 0);
    CHECK(id.box_vs_map == 0);
    CHECK(id.open_vs_box == 0);
    CHECK(id.max_xy_offset < 1e-18);
    CHECK(id.bitwise());
    // The surface is rough (not a trivial flat identity).
    CHECK(box.top.z.cwiseAbs().maxCoeff() > 1e-9);
    // The patch triangles are the box top-face triangles, as oriented vertex triples.
    CHECK(count_patch_triangles(om) == 128);
    const TriangleIdentity tri = patch_triangle_identity(box.top, om, box.mesh);
    CHECK(tri.open_patch == 128);
    CHECK(tri.box_top == 128);
    CHECK(tri.matched == 128);
    CHECK(tri.agree());
    // A changed height on the reference grid is detected (both meshes against the map).
    geometry::HeightMap other = box.top;
    other.z(3, 4) += 1e-12;
    const HeightIdentity bad = height_identity(other, om.mesh, box.mesh);
    CHECK(bad.open_vs_map == 1);
    CHECK(bad.box_vs_map == 1);
    CHECK(bad.open_vs_box == 0);
    CHECK_FALSE(bad.bitwise());
    // A changed height in the open mesh is detected against the map and against the box.
    const Real half = 0.5 * static_cast<Real>(box.top.z.rows() - 1) * box.top.dx;
    const Real x34 = -half + 3.0 * box.top.dx;
    const Real y34 = -half + 4.0 * box.top.dx;
    Vertices v = om.mesh.vertices();
    Index hit = -1;
    for (Index k = 0; k < v.rows(); ++k)
        if (std::abs(v(k, 0) - x34) < 1e-6 * box.top.dx &&
            std::abs(v(k, 1) - y34) < 1e-6 * box.top.dx)
            hit = k;
    REQUIRE(hit >= 0);
    v(hit, 2) += 1e-12;
    const geometry::TriangleMesh raised(v, om.mesh.triangles());
    const HeightIdentity bad_open = height_identity(box.top, raised, box.mesh);
    CHECK(bad_open.open_nodes == 81);
    CHECK(bad_open.open_vs_map == 1);
    CHECK(bad_open.box_vs_map == 0);
    CHECK(bad_open.open_vs_box == 1);
    CHECK_FALSE(bad_open.bitwise());
    // A misaligned grid (open mesh shifted by a quarter cell) finds no common node.
    Vertices s = om.mesh.vertices();
    s.col(0).array() += 0.25 * box.top.dx;
    s.col(1).array() += 0.25 * box.top.dx;
    geometry::OpenSurfaceMesh shifted = om;
    shifted.mesh = geometry::TriangleMesh(s, om.mesh.triangles());
    const HeightIdentity bad_grid = height_identity(box.top, shifted.mesh, box.mesh);
    CHECK(bad_grid.open_nodes == 0);
    CHECK(bad_grid.box_nodes == 81);
    CHECK_FALSE(bad_grid.bitwise());
    const TriangleIdentity bad_tri = patch_triangle_identity(box.top, shifted, box.mesh);
    CHECK(bad_tri.open_patch == 128);
    CHECK(bad_tri.matched == 0);
    CHECK_FALSE(bad_tri.agree());
    // The reversed open mesh (same vertex sets, opposite orientation) matches no triple. (A
    // single flipped triangle would be repaired by the TriangleMesh constructor.)
    Triangles flipped = om.mesh.triangles();
    flipped.col(1).swap(flipped.col(2));
    geometry::OpenSurfaceMesh reversed = om;
    reversed.mesh = geometry::TriangleMesh(om.mesh.vertices(), flipped);
    const TriangleIdentity rev = patch_triangle_identity(box.top, reversed, box.mesh);
    CHECK(rev.open_patch == 128);
    CHECK(rev.matched == 0);
    CHECK_FALSE(rev.agree());
    // The box rule: w0 <= L / 6 for Si.
    Case wide = c;
    wide.waist = c.L / 5.0;
    CHECK_THROWS_AS(make_box(map, wide), std::invalid_argument);
}

TEST_CASE("pml_si_crosscheck: the PML runs of the record share heights and triangles with the box",
          "[pml_si_crosscheck]") {
    // The three configurations solved on both routes (record: 7e at L = 4.5 um, lambda / 4; the
    // cross-check at L = 3.5 um, lambda / 4 and lambda / 6). Meshing only, no solve.
    struct Run {
        Real L;
        Real h;
        Real w0;  // the waists of the runs (L / 6)
        Index nodes;
        Index triangles;
    };
    const Case base;
    const geometry::HeightMap fine = generated_map(base);
    for (const Run run : {Run{4.5e-6, kLambda / 4.0, 0.75e-6, 1369, 2592},
                          Run{3.5e-6, kLambda / 4.0, 0.5833333333333333e-6, 841, 1568},
                          Run{3.5e-6, kLambda / 6.0, 0.5833333333333333e-6, 1849, 3528}}) {
        Case c = base;
        c.L = run.L;
        c.waist = run.w0;
        c.mesh_size = run.h;
        INFO("L " << run.L << " m, h " << run.h << " m");
        const geometry::HeightMap map = subsample(fine, subsample_factor(fine, c.mesh_size));
        const geometry::OpenSurfaceMesh om = make_open_mesh(map, c);
        const BoxGeometry box = make_box(map, c);
        const HeightIdentity id = height_identity(box.top, om.mesh, box.mesh);
        CHECK(id.grid_nodes == run.nodes);
        CHECK(id.bitwise());
        const TriangleIdentity tri = patch_triangle_identity(box.top, om, box.mesh);
        CHECK(box.top_triangles == run.triangles);
        CHECK(tri.open_patch == run.triangles);
        CHECK(tri.box_top == run.triangles);
        CHECK(tri.matched == run.triangles);
    }
}

TEST_CASE("pml_si_crosscheck: metric on synthetic far fields", "[pml_si_crosscheck]") {
    const box_validity::FarField r = random_far_field(11);
    SECTION("identical far fields") {
        const Metrics m = compare(r, r, 0.3, 0.3);
        CHECK(m.l2_refl == 0.0);
        CHECK(m.l2_off30 == 0.0);
        CHECK(m.dP_refl == 0.0);
        CHECK(m.eps_xz == 0.0);
        CHECK(m.eps_yz == 0.0);
        CHECK(m.dR == 0.0);
        CHECK(verdict(m, kTolerances).all());
    }
    SECTION("uniform scaling by 1 + e") {
        const Real e = 2.5e-3;
        box_validity::FarField t = r;
        t.refl *= 1.0 + e;
        t.cut_xz *= 1.0 + e;
        t.cut_yz *= 1.0 + e;
        const Metrics m = compare(t, r, 0.3 * (1.0 + e), 0.3);
        CHECK(m.l2_refl == Approx(e).epsilon(1e-9));
        CHECK(m.l2_off30 == Approx(e).epsilon(1e-9));
        CHECK(m.dP_refl == Approx(e).epsilon(1e-9));
        CHECK(m.dR == Approx(e).epsilon(1e-9));
        const Real rms = std::sqrt(r.cut_xz.squaredNorm() / static_cast<Real>(r.cut_xz.size()));
        CHECK(m.eps_xz == Approx(e * rms / r.cut_xz.maxCoeff()).epsilon(1e-9));
        // e = 0.25 % lies above every 7e tolerance (l2 0.229 %; eps_rr here e rms / max > 0.058 %).
        const Verdict v = verdict(m, kTolerances);
        CHECK_FALSE(v.l2);
        CHECK_FALSE(v.eps);
        CHECK_FALSE(v.R);
        // A scaling of 0.2 % passes l2 (<= 0.229 %) but not R (> 4.1e-5).
        box_validity::FarField t2 = r;
        t2.refl *= 1.002;
        const Verdict v2 = verdict(compare(t2, r, 0.3 * 1.002, 0.3), kTolerances);
        CHECK(v2.l2);
        CHECK(v2.eps);
        CHECK_FALSE(v2.R);
    }
    SECTION("random perturbation against the direct formula") {
        const box_validity::FarField d = random_far_field(12);
        box_validity::FarField t = r;
        t.refl += 1e-3 * d.refl;
        const Metrics m = compare(t, r, 0.3, 0.3);
        Real num = 0, den = 0, num30 = 0, den30 = 0;
        for (Index i = 0; i < r.refl.rows(); ++i) {
            const Real th = (static_cast<Real>(i) + 0.5) * box_validity::kDeg;
            for (Index j = 0; j < r.refl.cols(); ++j) {
                const Real a = std::sin(th) * std::pow(1e-3 * d.refl(i, j), 2);
                const Real b = std::sin(th) * std::pow(r.refl(i, j), 2);
                num += a;
                den += b;
                if (static_cast<Real>(i) + 0.5 >= 30.0) {
                    num30 += a;
                    den30 += b;
                }
            }
        }
        CHECK(m.l2_refl == Approx(std::sqrt(num / den)).epsilon(1e-9));
        CHECK(m.l2_off30 == Approx(std::sqrt(num30 / den30)).epsilon(1e-9));
        CHECK(m.eps_xz == 0.0);
        CHECK(m.dR == 0.0);
    }
    SECTION("invalid input") {
        box_validity::FarField t = r;
        t.refl.resize(10, 10);
        CHECK_THROWS_AS(compare(t, r, 0.3, 0.3), std::invalid_argument);
        CHECK_THROWS_AS(compare(r, r, 0.3, 0.0), std::invalid_argument);
    }
}

TEST_CASE("pml_si_crosscheck: verdict at the tolerance boundaries", "[pml_si_crosscheck]") {
    const Tolerances t{1e-3, 2e-4, 3e-5};
    Metrics m;
    m.l2_refl = 1e-3;
    m.eps_xz = 1e-4;
    m.eps_yz = 2e-4;
    m.dR = -3e-5;
    CHECK(verdict(m, t).all());
    m.eps_yz = 2.1e-4;
    CHECK_FALSE(verdict(m, t).eps);
    CHECK(verdict(m, t).l2);
    m.dR = 3.1e-5;
    CHECK_FALSE(verdict(m, t).R);
    m.l2_refl = 1.1e-3;
    CHECK_FALSE(verdict(m, t).l2);
    // The 7e tolerances are positive and ordered as the box sensitivities (l2 > eps_rr > dR).
    CHECK(kTolerances.l2_refl > kTolerances.eps_rr);
    CHECK(kTolerances.eps_rr > kTolerances.dR);
    CHECK(kTolerances.dR > 0.0);
}

TEST_CASE("pml_si_crosscheck: memory guard", "[pml_si_crosscheck]") {
    CHECK_NOTHROW(check_memory(20.0, 100.0, 50.0, false));
    CHECK_THROWS_AS(check_memory(51.0, 100.0, 50.0, false), std::runtime_error);
    CHECK_THROWS_AS(check_memory(20.0, 23.0, 50.0, false), std::runtime_error);  // 4 GB margin
    CHECK_NOTHROW(check_memory(20.0, 0.0, 50.0, false));  // available memory unknown
    CHECK_NOTHROW(check_memory(80.0, 10.0, 50.0, true));
    CHECK_THROWS_AS(check_memory(-1.0, 100.0, 50.0, true), std::invalid_argument);
    CHECK_THROWS_AS(check_memory(1.0, 100.0, 0.0, true), std::invalid_argument);
    // PML estimate: LU doubles the dense matrix, GMRES adds the Krylov basis.
    const Case c = small_case();
    const geometry::OpenSurfaceMesh om = make_open_mesh(mesh_map(c), c);
    const PmlEstimate lu = estimate_pml(c, om);
    CHECK(lu.unknowns == 2 * om.mesh.num_edges());
    CHECK(lu.dense_gb == Approx(16.0 * std::pow(static_cast<Real>(lu.unknowns), 2) / 1e9));
    CHECK(lu.peak_gb == Approx(2.0 * lu.dense_gb));
    Case g = c;
    g.pml_solver = PmlSolver::Gmres;
    const PmlEstimate ge = estimate_pml(g, om);
    CHECK(ge.peak_gb ==
          Approx(ge.dense_gb + box_validity::krylov_basis_gb(g.pml_max_iter, ge.unknowns)));
}

TEST_CASE("pml_si_crosscheck: stored runs round trip and shared keys", "[pml_si_crosscheck]") {
    // Per-process directory: the debug and release suites may run this case at the same time.
    const std::filesystem::path base =
        std::filesystem::temp_directory_path() /
        ("specklebem_pml_si_crosscheck_unit_" + std::to_string(process_id()));
    std::filesystem::remove_all(base);
    const Case c = small_case();
    RouteResult r;
    r.route = "box";
    r.converged = true;
    r.ff = random_far_field(21);
    r.p_inc = 2.0;
    r.p_refl = 0.6;
    const std::string dir_a = core::path_to_utf8(base / "a");
    store(dir_a, c, r, "a");
    RouteResult t = r;
    t.route = "pml";
    t.ff.refl *= 1.001;
    t.p_refl = 0.6006;
    const std::string dir_b = core::path_to_utf8(base / "b");
    Case cb = c;
    cb.L = 1.25e-6;  // L and route may differ
    store(dir_b, cb, t, "b");
    const StoredRun a = load(dir_a);
    const StoredRun b = load(dir_b);
    CHECK(a.tag == "a");
    CHECK(a.reflectance == Approx(0.3).epsilon(1e-15));
    CHECK(a.ff.refl == r.ff.refl);
    const Metrics m = compare_stored(b, a);
    CHECK(m.l2_refl == Approx(1e-3).epsilon(1e-9));
    CHECK(m.dR == Approx(1e-3).epsilon(1e-9));
    // A different surface (seed) is refused.
    Case cs = c;
    cs.seed = 2;
    const std::string dir_c = core::path_to_utf8(base / "c");
    store(dir_c, cs, r, "c");
    CHECK_THROWS_AS(compare_stored(load(dir_c), a), std::runtime_error);
    // An unconverged run is refused.
    RouteResult u = r;
    u.converged = false;
    const std::string dir_u = core::path_to_utf8(base / "u");
    store(dir_u, c, u, "u");
    CHECK_THROWS_AS(load(dir_u), std::runtime_error);
    std::filesystem::remove_all(base);
}
