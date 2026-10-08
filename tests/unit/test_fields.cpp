#include "specklebem/kernels/green.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "fields_test_support.hpp"

using namespace fields_test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("fields: projected RWG currents have unit normal component on their edge", "[post]") {
    const geometry::TriangleMesh mesh = geometry::make_icosphere(1.0, 1);
    const basis::RwgSpace space(mesh);
    for (Index n = 0; n < space.size(); ++n) {
        const Vec3 va = mesh.vertices().row(mesh.edges()(n, 0)).transpose();
        const Vec3 vb = mesh.vertices().row(mesh.edges()(n, 1)).transpose();
        const Index tp = space.plus_triangle(n);
        const Vec3 nu = (vb - va).cross(mesh.normal(tp)).normalized();
        const Vec3 mid = 0.5 * (va + vb);
        REQUIRE_THAT(space.value(n, tp, mid).dot(nu), WithinAbs(1.0, 1e-12));
    }
}

TEST_CASE("fields: Stratton-Chu signs from projected Mie currents, dielectric n = 1.5", "[post]") {
    // Exterior points must reproduce the Mie scattered field, interior points the Mie internal
    // field; this fixes the signs of the representation in R1 and R2 (fields.cpp). A wrong sign
    // leaves an O(1) error that does not decrease with refinement. Icosphere n = 2 and 3 here
    // (debug-build time limit); n = 3 -> 4 with the 5 % bound at n = 4, and the Ag sphere, are
    // the validation test tests/validation/test_post_mie_projection.cpp.
    const material::Material mat = lossless_n15();
    const MieSetup s2(mat, 2);
    const MieSetup s3(mat, 3);
    const ProjectionErrors p2 = near_field_errors(s2);
    const ProjectionErrors p3 = near_field_errors(s3);
    const RcsErrors r2 = rcs_errors(s2);
    const RcsErrors r3 = rcs_errors(s3);
    print_errors("n=1.5", 2, p2, r2);
    print_errors("n=1.5", 3, p3, r3);

    CHECK(p3.e_ext < p2.e_ext);
    CHECK(p3.h_ext < p2.h_ext);
    CHECK(p3.e_int < p2.e_int);
    CHECK(p3.h_int < p2.h_int);
    CHECK(r3.rms < r2.rms);
    // Measured at n = 3: 6.0 %, 4.8 %, 6.5 %, 8.2 % (E/H exterior/interior), RCS 1.8 %.
    CHECK(p3.e_ext < 0.1);
    CHECK(p3.h_ext < 0.1);
    CHECK(p3.e_int < 0.1);
    CHECK(p3.h_int < 0.1);
    CHECK(r3.rms < 0.05);
}

TEST_CASE("fields: far field is the r -> infinity limit of the scattered field", "[post]") {
    // Same discrete currents in both evaluations, so the mesh resolution does not matter (n = 2
    // keeps the test fast; ctest runs every test case in its own process, so a shared n = 3
    // setup would be rebuilt per test case anyway).
    const MieSetup s(lossless_n15(), 2);
    const std::vector<Vec3> dirs = {direction(0.0, 0.0), direction(kPi / 3, 0.0),
                                    direction(kPi, 0.0), direction(1.0, 1.0), direction(2.2, 4.0)};
    const auto nd = static_cast<Index>(dirs.size());
    Vertices khat(nd, 3);
    for (Index i = 0; i < nd; ++i) khat.row(i) = dirs[static_cast<std::size_t>(i)].transpose();
    FieldMatrix F;
    post::far_field(s.solution, khat, F);

    const Real k = k_vacuum();
    const auto scaled_near = [&](Real r) {
        FieldMatrix E;
        FieldMatrix H;
        post::scattered_field(s.solution, r * khat, E, H);
        FieldMatrix g(nd, 3);
        for (Index i = 0; i < nd; ++i) {
            g.row(i) = r * std::exp(kJ * k * r) * E.row(i);
        }
        return g;
    };
    const Real r1 = 1e4 * kLambda;
    const FieldMatrix g1 = scaled_near(r1);
    const FieldMatrix g2 = scaled_near(2 * r1);
    const FieldMatrix richardson = 2.0 * g2 - g1;  // removes the O(1/r) term
    const Real scale = F.cwiseAbs().maxCoeff();
    const Real err_raw = (g2 - F).cwiseAbs().maxCoeff() / scale;
    const Real err = (richardson - F).cwiseAbs().maxCoeff() / scale;
    WARN("far field: raw error at 2e4 lambda " << err_raw << ", Richardson " << err);
    CHECK(err < 1e-6);
    CHECK(err < err_raw);

    // Far field is transverse.
    for (Index i = 0; i < nd; ++i) {
        const Vec3c Fi = F.row(i).transpose();
        CHECK(std::abs(dot_cr(Fi, dirs[static_cast<std::size_t>(i)])) < 1e-12 * scale);
    }
}

TEST_CASE("fields: total field adds the incident field in R1 only", "[post]") {
    MieSetup s(lossless_n15(), 2);  // coarse: projection error ~ 25 %
    const TestPlaneWave wave(omega_of(kLambda));
    s.problem.excitation = &wave;

    Vertices pts(2, 3);
    pts.row(0) = (2.0 * kRadius * direction(1.0, 0.5)).transpose();
    pts.row(1) = (0.5 * kRadius * direction(2.0, 1.5)).transpose();
    FieldMatrix Es;
    FieldMatrix Hs;
    FieldMatrix Et;
    FieldMatrix Ht;
    post::scattered_field(s.solution, pts, Es, Hs);
    post::total_field(s.solution, pts, Et, Ht);
    const Vec3 r0 = pts.row(0).transpose();
    CHECK((Et.row(0) - Es.row(0) - incident_E(r0).transpose()).norm() < 1e-14);
    CHECK((Ht.row(0) - Hs.row(0) - incident_H(r0).transpose()).norm() < 1e-14 / constants::eta0);
    CHECK((Et.row(1) - Es.row(1)).norm() == 0.0);
    CHECK((Ht.row(1) - Hs.row(1)).norm() == 0.0);
    // Total field against the Mie total field (outside) and internal field (inside); a missing or
    // misplaced incident field would leave an error of ~ 1 V/m.
    const Vec3c mie_total = incident_E(r0) + s.mie.scattered_E(r0);
    CHECK((Et.row(0).transpose() - mie_total).cwiseAbs().maxCoeff() < 0.4);
    CHECK((Et.row(1).transpose() - s.mie.internal_E(pts.row(1).transpose())).cwiseAbs().maxCoeff() <
          0.4);

    // The frequency is taken from the excitation when SurfaceSolution::omega is 0.
    post::SurfaceSolution s_exc = s.solution;
    s_exc.omega = 0.0;
    FieldMatrix E2;
    FieldMatrix H2;
    post::total_field(s_exc, pts, E2, H2);
    CHECK((E2 - Et).norm() == 0.0);

    // |E_inc| at the origin is 1 V/m, so the RCS is unchanged by attaching the excitation.
    VectorXr angles(3);
    angles << 0.0, 1.0, kPi;
    const VectorXr with_exc = post::bistatic_rcs(s.solution, Vec3::UnitY(), angles);
    s.problem.excitation = nullptr;
    const VectorXr without = post::bistatic_rcs(s.solution, Vec3::UnitY(), angles);
    for (Index i = 0; i < 3; ++i) CHECK_THAT(with_exc(i), WithinRel(without(i), 1e-14));

    // Mismatched frequency and missing excitation throw.
    s.problem.excitation = &wave;
    post::SurfaceSolution s_bad = s.solution;
    s_bad.omega = 1.01 * omega_of(kLambda);
    CHECK_THROWS_AS(post::scattered_field(s_bad, pts, E2, H2), std::invalid_argument);
    s.problem.excitation = nullptr;
    CHECK_THROWS_AS(post::total_field(s.solution, pts, E2, H2), std::invalid_argument);
}

TEST_CASE("fields: bistatic RCS plane mapping", "[post]") {
    // plane_normal = -x scans the yz-plane (phi = pi/2): theta -> (0, sin, cos).
    const MieSetup s(lossless_n15(), 2);
    VectorXr angles(4);
    angles << 0.0, 0.7, 1.9, kPi;
    const VectorXr yz = post::bistatic_rcs(s.solution, -Vec3::UnitX(), angles);
    Vertices dirs(4, 3);
    for (Index i = 0; i < 4; ++i) dirs.row(i) = direction(angles(i), kPi / 2).transpose();
    FieldMatrix F;
    post::far_field(s.solution, dirs, F);
    for (Index i = 0; i < 4; ++i) {
        CHECK_THAT(yz(i), WithinRel(4 * kPi * F.row(i).squaredNorm(), 1e-12));
    }
    // A scattering plane must contain z.
    CHECK_THROWS_AS(post::bistatic_rcs(s.solution, Vec3::UnitZ(), angles), std::invalid_argument);
    CHECK_THROWS_AS(post::bistatic_rcs(s.solution, Vec3::Zero(), angles), std::invalid_argument);
}

TEST_CASE("fields: region detection by the generalised winding number", "[post]") {
    const geometry::TriangleMesh mesh = geometry::make_icosphere(1.0, 2);
    std::mt19937 rng(20261008);
    std::normal_distribution<Real> gauss(0.0, 1.0);
    Vertices pts(40, 3);
    for (Index i = 0; i < 40; ++i) {
        const Vec3 d = Vec3(gauss(rng), gauss(rng), gauss(rng)).normalized();
        const Real r = i < 20 ? 0.9 * static_cast<Real>(i) / 20.0 : 1.2 + static_cast<Real>(i);
        pts.row(i) = (r * d).transpose();
    }
    for (Index i = 0; i < 40; ++i) {
        const Real w = post::winding_number(mesh, pts.row(i).transpose());
        CHECK_THAT(w, WithinAbs(i < 20 ? 1.0 : 0.0, 1e-10));
    }
    const auto region = post::observation_regions(mesh, pts);
    for (Index i = 0; i < 40; ++i) CHECK(region(i) == (i < 20 ? 2 : 1));

    // Close to a face: rejected within 1e-3 h, accepted (and classified) at 1e-2 h.
    const Vec3 c = mesh.centroid(0);
    const Vec3 n = mesh.normal(0);
    const Real h = std::max({mesh.edge_length(0), mesh.edge_length(1)});  // order of magnitude
    Vertices near(2, 3);
    near.row(0) = (c + 2e-2 * h * n).transpose();
    near.row(1) = (c - 2e-2 * h * n).transpose();
    const auto near_region = post::observation_regions(mesh, near);
    CHECK(near_region(0) == 1);
    CHECK(near_region(1) == 2);
    Vertices on(1, 3);
    on.row(0) = (c + 1e-5 * h * n).transpose();
    CHECK_THROWS_AS(post::observation_regions(mesh, on), std::invalid_argument);
    on.row(0) = mesh.vertices().row(0);  // a vertex lies on the surface
    CHECK_THROWS_AS(post::observation_regions(mesh, on), std::invalid_argument);
    on.row(0) = c.transpose();
    CHECK_THROWS_AS(post::observation_regions(mesh, on), std::invalid_argument);
    on(0, 0) = std::nan("");
    CHECK_THROWS_AS(post::observation_regions(mesh, on), std::invalid_argument);
    CHECK_THROWS_AS(post::observation_regions(mesh, near, 0.0), std::invalid_argument);

    // scattered_field rejects a surface point as well.
    const basis::RwgSpace space(mesh);
    op::Problem problem;
    problem.space = &space;
    const post::SurfaceSolution sol{&problem, VectorXc::Zero(2 * space.size()), omega_of(kLambda)};
    FieldMatrix E;
    FieldMatrix H;
    on.row(0) = mesh.vertices().row(3);
    CHECK_THROWS_AS(post::scattered_field(sol, on, E, H), std::invalid_argument);

    // Open patch in z = 0 with normal -z (rough-surface convention: R1 at z < 0, R2 at z > 0).
    Vertices pv(4, 3);
    pv << -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0;
    Triangles pt(2, 3);
    pt << 0, 2, 1, 0, 3, 2;
    const geometry::TriangleMesh patch(pv, pt);
    REQUIRE(patch.normal(0).z() < 0);
    Vertices pp(2, 3);
    pp << 0.1, 0.2, 0.3, 0.1, 0.2, -0.3;
    const auto patch_region = post::observation_regions(patch, pp);
    CHECK(patch_region(0) == 2);
    CHECK(patch_region(1) == 1);
}

TEST_CASE("fields: plane and cylinder grids", "[post]") {
    const Vec3 o(1, 2, 3);
    const Vec3 u(2, 0, 0);
    const Vec3 v(0, 0, 4);
    const Vertices g = post::plane_grid(o, u, v, 3, 2);
    REQUIRE(g.rows() == 6);
    CHECK((g.row(0).transpose() - o).norm() < 1e-15);
    CHECK((g.row(1).transpose() - (o + v)).norm() < 1e-15);        // i = 0, j = 1
    CHECK((g.row(2).transpose() - (o + 0.5 * u)).norm() < 1e-15);  // i = 1, j = 0
    CHECK((g.row(5).transpose() - (o + u + v)).norm() < 1e-15);    // last corner
    const Vertices g1 = post::plane_grid(o, u, v, 1, 1);
    REQUIRE(g1.rows() == 1);
    CHECK((g1.row(0).transpose() - o).norm() == 0.0);
    CHECK_THROWS_AS(post::plane_grid(o, u, v, 0, 2), std::invalid_argument);
    CHECK_THROWS_AS(post::plane_grid(o, u, v, 2, -1), std::invalid_argument);

    const Real radius = 2.0;
    const Vertices c = post::cylinder_grid(radius, -0.5, 1.0, 4, -1.0, 1.0, 3);
    REQUIRE(c.rows() == 12);
    for (Index k = 0; k < 4; ++k) {
        const Real theta = -0.5 + 1.5 * static_cast<Real>(k) / 3.0;
        for (Index l = 0; l < 3; ++l) {
            const Vec3 p = c.row(k * 3 + l).transpose();
            CHECK_THAT(std::hypot(p.x(), p.z()), WithinRel(radius, 1e-14));
            CHECK_THAT(std::atan2(p.x(), -p.z()), WithinAbs(theta, 1e-14));  // from -z
            CHECK_THAT(p.y(), WithinAbs(-1.0 + static_cast<Real>(l), 1e-14));
        }
    }
    // theta = 0 is the -z direction (reflection geometry of docs/06).
    const Vertices c0 = post::cylinder_grid(1.0, 0.0, 0.0, 1, 0.5, 0.5, 1);
    CHECK((c0.row(0).transpose() - Vec3(0, 0.5, -1)).norm() < 1e-15);
    CHECK_THROWS_AS(post::cylinder_grid(0.0, 0, 1, 2, 0, 1, 2), std::invalid_argument);
    CHECK_THROWS_AS(post::cylinder_grid(1.0, 0, 1, 0, 0, 1, 2), std::invalid_argument);
    CHECK_THROWS_AS(post::cylinder_grid(1.0, 0, 1, 2, 0, 1, 0), std::invalid_argument);
}

TEST_CASE("fields: polarised intensities", "[post]") {
    std::mt19937 rng(42);
    std::uniform_real_distribution<Real> uni(-1.0, 1.0);
    FieldMatrix E(10, 3);
    for (Index i = 0; i < 10; ++i) {
        for (Index c = 0; c < 3; ++c) E(i, c) = Complex(uni(rng), uni(rng));
    }
    const post::PolarizedIntensity pi = post::polarized_intensity(E, Vec3(1, 2, -0.5));
    for (Index i = 0; i < 10; ++i) {
        CHECK_THAT(pi.co(i) + pi.cross(i), WithinRel(E.row(i).squaredNorm(), 1e-14));
        CHECK(pi.co(i) >= 0.0);
        CHECK(pi.cross(i) >= 0.0);
    }
    // A field along e_co is 100 % co-polarised (direction normalised internally).
    FieldMatrix Ec(2, 3);
    Ec.row(0) << Complex(0.3, -0.7), 0, 0;
    Ec.row(1) << Complex(1, 1), Complex(1, 1), 0;
    const post::PolarizedIntensity px = post::polarized_intensity(Ec.topRows(1), Vec3(2, 0, 0));
    CHECK_THAT(px.co(0), WithinRel(Ec.row(0).squaredNorm(), 1e-14));
    CHECK_THAT(px.cross(0), WithinAbs(0.0, 1e-15));
    const post::PolarizedIntensity pd = post::polarized_intensity(Ec.bottomRows(1), Vec3(1, 1, 0));
    CHECK_THAT(pd.co(0), WithinRel(4.0, 1e-14));
    CHECK_THAT(pd.cross(0), WithinAbs(0.0, 1e-14));
    CHECK_THROWS_AS(post::polarized_intensity(E, Vec3::Zero()), std::invalid_argument);
}

TEST_CASE("fields: differential reflection coefficient averages |E|^2 along the axis", "[post]") {
    const int nt = 3;
    const int ny = 4;
    FieldMatrix E(nt * ny, 3);
    for (Index k = 0; k < nt; ++k) {
        for (Index l = 0; l < ny; ++l) {
            const auto a = static_cast<Real>((k + 1) * (l + 1));
            E.row(k * ny + l) << a * Complex(1, 1), 0, Complex(0, 0);
        }
    }
    const VectorXr drc = post::differential_reflection_coefficient(E, nt, ny);
    REQUIRE(drc.size() == nt);
    for (Index k = 0; k < nt; ++k) {
        // mean over l of 2 (k+1)^2 (l+1)^2 = 2 (k+1)^2 (1 + 4 + 9 + 16) / 4
        CHECK_THAT(drc(k), WithinRel(15.0 * static_cast<Real>((k + 1) * (k + 1)), 1e-14));
    }
    CHECK_THROWS_AS(post::differential_reflection_coefficient(E, nt, ny + 1),
                    std::invalid_argument);
    CHECK_THROWS_AS(post::differential_reflection_coefficient(E, 0, ny), std::invalid_argument);
}

TEST_CASE("fields: invalid input throws", "[post]") {
    const geometry::TriangleMesh mesh = geometry::make_icosphere(kRadius, 1);
    const basis::RwgSpace space(mesh);
    op::Problem problem;
    problem.space = &space;
    Vertices pts(1, 3);
    pts << 0, 0, 3 * kRadius;
    FieldMatrix E;
    FieldMatrix H;
    VectorXr angles(1);
    angles << 0.0;
    const Real omega = omega_of(kLambda);

    // Wrong currents length.
    const post::SurfaceSolution bad_len{&problem, VectorXc::Zero(2 * space.size() - 1), omega};
    CHECK_THROWS_AS(post::scattered_field(bad_len, pts, E, H), std::invalid_argument);
    CHECK_THROWS_AS(post::far_field(bad_len, pts, E), std::invalid_argument);
    CHECK_THROWS_AS(post::bistatic_rcs(bad_len, Vec3::UnitY(), angles), std::invalid_argument);

    // Null problem / space.
    const post::SurfaceSolution no_problem{nullptr, VectorXc::Zero(2 * space.size()), omega};
    CHECK_THROWS_AS(post::scattered_field(no_problem, pts, E, H), std::invalid_argument);
    CHECK_THROWS_AS(post::total_field(no_problem, pts, E, H), std::invalid_argument);
    CHECK_THROWS_AS(post::far_field(no_problem, pts, E), std::invalid_argument);
    CHECK_THROWS_AS(post::bistatic_rcs(no_problem, Vec3::UnitY(), angles), std::invalid_argument);
    op::Problem no_space;
    const post::SurfaceSolution no_space_sol{&no_space, VectorXc::Zero(0), omega};
    CHECK_THROWS_AS(post::scattered_field(no_space_sol, pts, E, H), std::invalid_argument);

    // Unknown frequency, negative frequency.
    const post::SurfaceSolution no_omega{&problem, VectorXc::Zero(2 * space.size()), 0.0};
    CHECK_THROWS_AS(post::scattered_field(no_omega, pts, E, H), std::invalid_argument);
    CHECK_THROWS_AS(post::far_field(no_omega, pts, E), std::invalid_argument);
    const post::SurfaceSolution neg_omega{&problem, VectorXc::Zero(2 * space.size()), -omega};
    CHECK_THROWS_AS(post::scattered_field(neg_omega, pts, E, H), std::invalid_argument);

    // Quadrature degree must be positive-interior; zero direction.
    const post::SurfaceSolution ok{&problem, VectorXc::Ones(2 * space.size()), omega};
    post::FieldOptions opt;
    opt.quad_degree = 3;  // negative weight
    CHECK_THROWS_AS(post::scattered_field(ok, pts, E, H, opt), std::invalid_argument);
    opt.quad_degree = 21;
    CHECK_THROWS_AS(post::far_field(ok, pts, E, opt), std::invalid_argument);
    Vertices zero_dir = Vertices::Zero(1, 3);
    CHECK_THROWS_AS(post::far_field(ok, zero_dir, E), std::invalid_argument);

    // Problem::omega takes precedence (WP9): it alone defines the frequency (same field as
    // SurfaceSolution::omega), a different SurfaceSolution::omega throws, a negative one throws.
    {
        FieldMatrix E_ref;
        FieldMatrix H_ref;
        post::scattered_field(ok, pts, E_ref, H_ref);
        problem.omega = omega;
        const post::SurfaceSolution from_problem{&problem, ok.currents, 0.0};
        FieldMatrix Ep;
        FieldMatrix Hp;
        post::scattered_field(from_problem, pts, Ep, Hp);
        CHECK((Ep - E_ref).norm() == 0.0);
        const post::SurfaceSolution mismatch{&problem, ok.currents, 1.01 * omega};
        CHECK_THROWS_AS(post::scattered_field(mismatch, pts, Ep, Hp), std::invalid_argument);
        problem.omega = -omega;
        CHECK_THROWS_AS(post::scattered_field(from_problem, pts, Ep, Hp), std::invalid_argument);
        problem.omega = 0.0;
    }

    // Zero currents radiate nothing.
    const post::SurfaceSolution zero{&problem, VectorXc::Zero(2 * space.size()), omega};
    post::scattered_field(zero, pts, E, H);
    CHECK(E.norm() == 0.0);
    CHECK(H.norm() == 0.0);
}

TEST_CASE("fields: inline kernel factors agree with kernels::green / grad_green", "[post]") {
    const Real k0 = k_vacuum();
    const Vec3 rp(0.1e-6, -0.2e-6, 0.05e-6);
    const Vec3 dir = Vec3(0.3, -0.5, 0.8).normalized();
    // vacuum, lossless dielectric, Ag (lossy, Im k < 0)
    const Complex ks[] = {Complex(k0, 0), k0 * lossless_n15().refractive_index(),
                          k0 * material::silver_500nm().refractive_index()};
    for (const Complex k : ks) {
        REQUIRE(k.imag() <= 0.0);
        for (const Real R : {1e-9, 5e-8, 0.4e-6, 3e-6}) {
            const Vec3 r = rp + R * dir;
            const post::detail::GreenFactors gf =
                post::detail::green_factors(k.real(), k.imag(), R);
            const Complex G = kernels::green(r, rp, k);
            const Vec3c gradG = kernels::grad_green(r, rp, k);
            const Complex f(gf.f_re, gf.f_im);
            CHECK(std::abs(Complex(gf.g_re, gf.g_im) - G) <= 1e-13 * std::abs(G));
            const Vec3c fd = f * (r - rp).cast<Complex>();
            CHECK((fd - gradG).norm() <= 1e-13 * gradG.norm());
        }
    }
}

TEST_CASE("fields: open-mesh region rule beyond the patch edge", "[post]") {
    // Patch [-1, 1]^2 in z = 0 with normal -z (R1 at z < 0, R2 at z > 0).
    Vertices pv(4, 3);
    pv << -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0;
    Triangles pt(2, 3);
    pt << 0, 2, 1, 0, 3, 2;
    const geometry::TriangleMesh patch(pv, pt);
    REQUIRE(!patch.is_closed());

    // Laterally beyond the edge the winding number only tells on which side of the plane of the
    // patch the point lies (|w| small but well above 1e-6 here): this is the documented failure
    // mode of the open-mesh rule, which cannot know where the true interface continues.
    Vertices lateral(2, 3);
    lateral << 3.0, 0.0, 0.5, 3.0, 0.0, -0.5;
    const Real w_up = post::winding_number(patch, lateral.row(0).transpose());
    CHECK(w_up > 1e-6);
    CHECK(w_up < 0.05);
    const auto reg = post::observation_regions(patch, lateral);
    CHECK(reg(0) == 2);
    CHECK(reg(1) == 1);

    // In (or very close to) the plane of the patch beyond its edge, w -> 0: ambiguous, throws.
    Vertices in_plane(1, 3);
    in_plane << 3.0, 0.0, 0.0;
    CHECK(post::winding_number(patch, in_plane.row(0).transpose()) == 0.0);
    CHECK_THROWS_AS(post::observation_regions(patch, in_plane), std::invalid_argument);
    in_plane << 3.0, 0.0, 1e-9;
    CHECK_THROWS_AS(post::observation_regions(patch, in_plane), std::invalid_argument);
    // A closed mesh never throws for this reason (w = 0 far outside is unambiguous).
    const geometry::TriangleMesh sphere = geometry::make_icosphere(1.0, 1);
    Vertices far(1, 3);
    far << 100.0, 0.0, 0.0;
    CHECK(post::observation_regions(sphere, far)(0) == 1);
}
