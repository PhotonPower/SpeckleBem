/// @file test_huygens.cpp
/// Unit tests of the Huygens-plane far field (post::huygens_*, ADR 0010, WP-PML6):
///  * the anchored plane grid and its window weights (sum rule of the raised-cosine square
///    window, symmetry, reweighting to a smaller window bitwise);
///  * the equivalent-current sign convention J_H = n x H, M_H = -n x E on analytic fields: the
///    plane radiation of electric and magnetic point-dipole fields reproduces the dipole far
///    field in the half space beyond the plane, and the plane currents with the reversed normal
///    radiate no field into the source half space (null field of Love's equivalence);
///  * the stretched-source near field (post::exterior_scattered_field): on a flat open surface
///    with a PML, RWG currents interpolated from the analytic continuation of a dipole's traces
///    reproduce the dipole field below the surface, also next to the PML start, where the PML
///    sources matter; without a stretch it is bitwise scattered_field();
///  * side checks of the directions, option validation, thread-count independence.
#include "specklebem/basis/rwg.hpp"
#include "specklebem/geometry/coordinate_stretch.hpp"
#include "specklebem/geometry/open_surface.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/postprocessing/fields.hpp"
#include "specklebem/postprocessing/huygens.hpp"

#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef SPECKLEBEM_HAVE_OPENMP
#include <omp.h>
#endif

using namespace specklebem;

namespace {

using FieldMatrix = Eigen::Matrix<Complex, Eigen::Dynamic, 3>;
constexpr Complex kJ{0.0, 1.0};
constexpr Real kLam = 1e-6;  ///< vacuum wavelength of the tests [m]

/// a x b without conjugation (Eigen's cross() conjugates complex results).
Vec3c cross_cc(const Vec3c& a, const Vec3c& b) {
    return {a(1) * b(2) - a(2) * b(1), a(2) * b(0) - a(0) * b(2), a(0) * b(1) - a(1) * b(0)};
}

Real omega_of(Real lambda) {
    return 2.0 * constants::pi * constants::c0 / lambda;
}

/// Fields at the (possibly complex) point r of an electric point current p and a magnetic point
/// current m at r0 in vacuum (J = p delta(r - r0), M = m delta(r - r0); the representation of
/// fields.cpp): with grad G = f d, d = r - r0, f = -(1 + jkR) G / R^2, f' = -(k^2 G + 3 f) / R,
///   E = -j w mu G p + (1 / j w eps) [f p + (f' / R)(d . p) d] - f d x m,
///   H = f d x p - j w eps G m + (1 / j w mu) [f m + (f' / R)(d . m) d].
/// R = sqrt(d . d) unconjugated on the principal branch (the analytic continuation).
void dipole_fields(const Vec3c& r, const Vec3& r0, const Vec3c& p, const Vec3c& m, Real lambda,
                   Vec3c& E, Vec3c& H) {
    const Real k = 2.0 * constants::pi / lambda;
    const Real omega = omega_of(lambda);
    const Vec3c d = r - r0.cast<Complex>();
    const Complex R = std::sqrt((d.array() * d.array()).sum());  // unconjugated d . d
    const Complex G = std::exp(-kJ * k * R) / (4.0 * constants::pi * R);
    const Complex f = -(1.0 + kJ * k * R) * G / (R * R);
    const Complex fp = -(k * k * G + 3.0 * f) / R;
    const Complex jwmu = kJ * omega * constants::mu0;
    const Complex jweps = kJ * omega * constants::eps0;
    const auto udot = [](const Vec3c& a, const Vec3c& b) {
        return a(0) * b(0) + a(1) * b(1) + a(2) * b(2);
    };
    const Vec3c gp = f * p + (fp / R) * udot(d, p) * d;
    const Vec3c gm = f * m + (fp / R) * udot(d, m) * d;
    E = -jwmu * G * p + gp / jweps - f * cross_cc(d, m);
    H = f * cross_cc(d, p) - jweps * G * m + gm / jwmu;
}

/// Far-field amplitude of the point currents (F = (j k / 4 pi) k x [eta (k x p) + m] e^{jk k.r0}).
Vec3c dipole_far(const Vec3& kh, const Vec3& r0, const Vec3c& p, const Vec3c& m, Real lambda) {
    const Real k = 2.0 * constants::pi / lambda;
    const Real eta = constants::eta0;
    const Vec3c khc = kh.cast<Complex>();
    const Complex ph = std::exp(kJ * k * kh.dot(r0));
    return (kJ * k / (4.0 * constants::pi)) * ph * cross_cc(khc, eta * cross_cc(khc, p) + m);
}

/// Directions at polar angles theta (from the pole pole_z z) and azimuths phi.
Vertices cone_directions(Real pole_z, const std::vector<Real>& theta_deg, int n_phi) {
    Vertices d(static_cast<Index>(theta_deg.size()) * n_phi, 3);
    Index r = 0;
    for (const Real t : theta_deg) {
        const Real th = t * constants::pi / 180.0;
        for (int j = 0; j < n_phi; ++j) {
            const Real ph = 2.0 * constants::pi * j / n_phi;
            d.row(r++) = Vec3(std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph),
                              pole_z * std::cos(th))
                             .transpose();
        }
    }
    return d;
}

/// Plane fields of the dipoles on a HuygensPlane.
void sample_dipoles(const post::HuygensPlane& pl, const Vec3& r0, const Vec3c& p, const Vec3c& m,
                    FieldMatrix& E, FieldMatrix& H) {
    E.resize(pl.points.rows(), 3);
    H.resize(pl.points.rows(), 3);
    for (Index i = 0; i < pl.points.rows(); ++i) {
        Vec3c e, h;
        dipole_fields(pl.points.row(i).transpose().cast<Complex>(), r0, p, m, kLam, e, h);
        E.row(i) = e.transpose();
        H.row(i) = h.transpose();
    }
}

/// Flat open surface (vacuum over Ag at lambda = 1 um): patch 1 um, taper / gap 0.25 um, PML
/// 1 um (pml_start 1 um, outer 2 um), sigma_max 4, order 2; mesh spacing h.
struct FlatPml {
    geometry::OpenSurfaceMesh om;
    std::unique_ptr<basis::RwgSpace> space;
    geometry::CoordinateStretch stretch;
    op::Problem problem;

    explicit FlatPml(Real h) : om(make_mesh(h)), stretch(geometry::make_pml_stretch(om, 4.0, 2.0)) {
        space = std::make_unique<basis::RwgSpace>(om.mesh);
        problem.space = space.get();
        problem.exterior = material::vacuum();
        problem.object = material::silver_500nm();
        problem.omega = omega_of(kLam);
        problem.stretch = &stretch;
    }

    static geometry::OpenSurfaceMesh make_mesh(Real h) {
        geometry::OpenSurfaceParams op;
        op.patch_size = 1e-6;
        op.taper_width = 0.25e-6;
        op.gap_width = 0.25e-6;
        op.pml_width = 1e-6;
        const Real L = op.patch_size + 2.0 * op.taper_width;
        const auto cells = static_cast<Index>(std::llround(L / h));
        geometry::HeightMap hm;
        hm.dx = L / static_cast<Real>(cells);
        hm.dy = hm.dx;
        hm.z = MatrixXr::Zero(cells + 1, cells + 1);
        return geometry::make_open_surface_mesh(hm, op);
    }
};

VectorXc random_currents(Index n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<Real> u(-1.0, 1.0);
    VectorXc x(n);
    for (Index i = 0; i < n; ++i) x(i) = Complex(u(rng), u(rng));
    return x;
}

}  // namespace

TEST_CASE("huygens: anchored plane grid, window weights and sum rule", "[huygens]") {
    // int int w(max(|x|, |y|)) dx dy = int_0^b w(t) 8 t dt = 2 (a^2 + b^2) - 8 (b - a)^2 / pi^2.
    const post::CurrentWindow win{1.0, 2.0};
    const Real exact = 2.0 * (1.0 + 4.0) - 8.0 / (constants::pi * constants::pi);
    Real prev_err = 1.0;
    for (const Real h : {0.1, 0.05, 0.025}) {
        const post::HuygensPlane pl = post::huygens_plane(-0.5, -1.0, win, h);
        REQUIRE(pl.points.rows() == pl.weights.size());
        CHECK(pl.spacing == h);
        CHECK(pl.normal_z == -1.0);
        CHECK((pl.points.col(2).array() == -0.5).all());
        CHECK((pl.weights.array() > 0.0).all());
        CHECK(pl.points.cwiseAbs().leftCols(2).maxCoeff() < win.outer_half);
        // Nodes are integer multiples of h (anchored at 0) and the set is symmetric.
        Real max_off = 0.0, sum_x = 0.0, sum_y = 0.0;
        for (Index i = 0; i < pl.points.rows(); ++i) {
            for (int c = 0; c < 2; ++c) {
                const Real q = pl.points(i, c) / h;
                max_off = std::max(max_off, std::abs(q - std::round(q)));
            }
            sum_x += pl.points(i, 0) * pl.weights(i);
            sum_y += pl.points(i, 1) * pl.weights(i);
        }
        CHECK(max_off < 1e-9);
        CHECK(std::abs(sum_x) < 1e-12);
        CHECK(std::abs(sum_y) < 1e-12);
        const Real err = std::abs(pl.weights.sum() / exact - 1.0);
        CHECK(err < 2e-3);
        CHECK(err < prev_err);  // converges with the spacing
        prev_err = err;
        // Weight = h^2 w(t).
        const Index mid = pl.points.rows() / 2;
        const Real t = std::max(std::abs(pl.points(mid, 0)), std::abs(pl.points(mid, 1)));
        CHECK(pl.weights(mid) == h * h * post::current_window_value(win, t));
    }
}

TEST_CASE("huygens: the window error of a point source falls like 1 / (k Delta)^2", "[huygens]") {
    // int int exp(-jkR) / R dA over the plane at distance d from a point = 2 pi exp(-jkd) / (jk)
    // (rho drho = R dR: the integrand does not decay in R). The windowed trapezoid sum differs by
    // the window-end terms ~ pi^2 / (2 (k Delta)^2) of the C^1 raised cosine (the measured
    // property behind the Huygens-plane window rule, ADR 0010 result): doubling the transition
    // width Delta divides the error by ~4.
    const Real k = 2.0 * constants::pi / kLam;
    const Complex ref = 2.0 * constants::pi * std::exp(-kJ * k * kLam) / (kJ * k);
    std::vector<Real> err;
    for (const Real delta : {2.0, 4.0, 8.0}) {
        const post::HuygensPlane pl = post::huygens_plane(
            -kLam, -1.0, post::CurrentWindow{kLam, (1.0 + delta) * kLam}, kLam / 8.0);
        Complex s = 0.0;
        for (Index i = 0; i < pl.points.rows(); ++i) {
            const Real R = pl.points.row(i).norm();
            s += pl.weights(i) * std::exp(-kJ * k * R) / R;
        }
        err.push_back(std::abs(s / ref - 1.0));
        const Real model = constants::pi * constants::pi / (2.0 * std::pow(k * delta * kLam, 2));
        WARN("window transition " << delta << " lambda: error " << err.back() << ", model per end "
                                  << model);
        CHECK(err.back() < 3.0 * model);
    }
    CHECK(err[0] / err[1] > 2.5);
    CHECK(err[1] / err[2] > 2.5);
}

TEST_CASE("huygens: reweighting a plane field equals the smaller plane bitwise", "[huygens]") {
    const Real h = 0.125;
    const post::CurrentWindow big{1.5, 3.0};
    const post::CurrentWindow small{1.0, 2.2};
    post::HuygensPlaneField f;
    f.plane = post::huygens_plane(2.0, 1.0, big, h);
    const Index n = f.plane.points.rows();
    f.E.resize(n, 3);
    f.H.resize(n, 3);
    const VectorXc a = random_currents(6 * n, 7);
    for (Index i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c) {
            f.E(i, c) = a(6 * i + c);
            f.H(i, c) = a(6 * i + 3 + c);
        }
    f.k = Complex(3.0, 0.0);
    f.eta = Complex(377.0, 0.0);
    const post::HuygensPlaneField g = post::huygens_reweight(f, small);
    const post::HuygensPlane ref = post::huygens_plane(2.0, 1.0, small, h);
    REQUIRE(g.plane.points.rows() == ref.points.rows());
    CHECK((g.plane.points.array() == ref.points.array()).all());
    CHECK((g.plane.weights.array() == ref.weights.array()).all());
    // The fields follow their points.
    for (Index r = 0; r < g.plane.points.rows(); ++r) {
        Index i = 0;
        while (!(f.plane.points.row(i).array() == g.plane.points.row(r).array()).all()) ++i;
        CHECK((f.E.row(i).array() == g.E.row(r).array()).all());
        CHECK((f.H.row(i).array() == g.H.row(r).array()).all());
    }
    CHECK_THROWS_AS(post::huygens_reweight(f, post::CurrentWindow{1.0, 3.5}),
                    std::invalid_argument);
    CHECK_THROWS_AS(post::huygens_reweight(f, post::CurrentWindow{2.0, 1.0}),
                    std::invalid_argument);
}

TEST_CASE("huygens: plane radiation of dipole fields reproduces their far field", "[huygens]") {
    // Electric and magnetic point currents above the plane z = -lambda (normal -z: the far field
    // is evaluated in the lower half space); spacing lambda / 6, window inner 2 lambda, outer 8
    // and 11 lambda; directions up to 40 deg from -z (stationary points within 0.84 lambda of the
    // axis). The error is the window's: a point source's plane field does not decay in the
    // R-integral (rho drho = R dR), so the C^1 raised cosine leaves ~ pi^2 / (2 (k Delta)^2) per
    // window end (Delta = outer - inner; test case "the window error of a point source"):
    // 3.5e-3 / 1.5e-3 at Delta = 6 / 9 lambda. The bound is c times that model with c = 5.5: the
    // model is the scalar per-end term of one point source, while the check is the max-norm
    // vector far-field error over a 40 deg cone (two window ends, electric and magnetic parts);
    // measured 2.4-4.6 x the model (electric 9.9e-3 / 3.9e-3, magnetic 8.3e-3 / 3.3e-3, both
    // 1.58e-2 / 6.2e-3 at outer 8 / 11 lambda), so c = 5.5 (1.9e-2 / 8.5e-3) keeps a margin of
    // >= 1.2 and is tighter than the earlier fixed bounds 2e-2 / 1e-2.
    const Vec3 r0(0.13 * kLam, -0.07 * kLam, 0.0);
    const Real eta = constants::eta0;
    const Vec3c p(Complex(1.0, 0.2), Complex(-0.4, 0.1), Complex(0.7, -0.3));
    const Vec3c m = eta * Vec3c(Complex(0.3, -0.5), Complex(0.9, 0.0), Complex(-0.2, 0.4));
    const Vertices dirs = cone_directions(-1.0, {0.0, 10.0, 20.0, 30.0, 40.0}, 8);
    constexpr Real kInner = 2.0;  // inner window edge [lambda]
    struct Source {
        Vec3c p, m;
        const char* name;
    };
    for (const Source& src : {Source{p, Vec3c::Zero(), "electric"},
                              Source{Vec3c::Zero(), m, "magnetic"}, Source{p, m, "both"}}) {
        Real prev = 1.0;
        for (const Real outer : {8.0, 11.0}) {
            const post::HuygensPlane pl = post::huygens_plane(
                -kLam, -1.0, post::CurrentWindow{kInner * kLam, outer * kLam}, kLam / 6.0);
            FieldMatrix E, H, F;
            sample_dipoles(pl, r0, src.p, src.m, E, H);
            post::huygens_radiate(pl, E, H, Complex(2.0 * constants::pi / kLam, 0.0),
                                  Complex(eta, 0.0), dirs, F);
            Real err = 0.0, scale = 0.0;
            for (Index i = 0; i < dirs.rows(); ++i) {
                const Vec3c ref = dipole_far(dirs.row(i).transpose(), r0, src.p, src.m, kLam);
                err = std::max(err, (F.row(i).transpose() - ref).norm());
                scale = std::max(scale, ref.norm());
            }
            const Real rel = err / scale;
            WARN(src.name << " dipole, window outer " << outer
                          << " lambda: max |dF| / max |F| = " << rel);
            const Real k_delta =
                2.0 * constants::pi * (outer - kInner);  // k Delta, Delta in lambda
            const Real model = constants::pi * constants::pi / (2.0 * k_delta * k_delta);
            CHECK(rel < 5.5 * model);
            CHECK(rel < prev);  // decreases with the window (quadrature/truncation controlled)
            prev = rel;
        }
    }
}

TEST_CASE("huygens: reversed plane currents radiate no field into the source half space",
          "[huygens]") {
    // Love's equivalence: J = n x H, M = -n x E with n pointing away from the sources reproduce
    // the field beyond the plane and zero on the source side. With n reversed (+z, towards the
    // dipole) the plane currents are -J, -M of the correct ones, and their radiation into the
    // upper half space (the source side) must vanish; a sign error between J_H and M_H would
    // give twice the dipole field there instead. The bound 1e-2 is therefore discriminating: a
    // sign error gives O(1) (|F| / |F_dipole| ~ 2 in these directions), while the correct
    // currents leave only the window-end error of the truncated plane (the point-source model
    // pi^2 / (2 (k Delta)^2) = 1.5e-3 per end at Delta = 9 lambda; measured 5.8e-4).
    const Vec3 r0(0.1 * kLam, 0.05 * kLam, 0.0);
    const Vec3c p(Complex(1.0, 0.0), Complex(0.5, 0.5), Complex(0.2, 0.0));
    const Vec3c m =
        constants::eta0 * Vec3c(Complex(0.0, 1.0), Complex(0.4, 0.0), Complex(0.0, 0.0));
    const post::HuygensPlane pl =
        post::huygens_plane(-kLam, 1.0, post::CurrentWindow{2.0 * kLam, 11.0 * kLam}, kLam / 6.0);
    FieldMatrix E, H, F;
    sample_dipoles(pl, r0, p, m, E, H);
    const Vertices up = cone_directions(1.0, {0.0, 15.0, 30.0}, 6);
    post::huygens_radiate(pl, E, H, Complex(2.0 * constants::pi / kLam, 0.0),
                          Complex(constants::eta0, 0.0), up, F);
    Real scale = 0.0;
    for (Index i = 0; i < up.rows(); ++i)
        scale = std::max(scale, dipole_far(up.row(i).transpose(), r0, p, m, kLam).norm());
    const Real rel = F.rowwise().norm().maxCoeff() / scale;
    WARN("null field on the source side: max |F| / max |F_dipole| = " << rel);
    CHECK(rel < 1e-2);
}

TEST_CASE("huygens: directions must lie beyond the plane", "[huygens]") {
    const post::HuygensPlane pl =
        post::huygens_plane(0.0, -1.0, post::CurrentWindow{0.5, 1.0}, 0.25);
    const FieldMatrix E = FieldMatrix::Ones(pl.points.rows(), 3);
    const FieldMatrix H = FieldMatrix::Ones(pl.points.rows(), 3);
    FieldMatrix F;
    Vertices ok(3, 3);
    ok << 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.3, 0.2, 1e-13;  // below, grazing, grazing + rounding
    CHECK_NOTHROW(post::huygens_radiate(pl, E, H, Complex(1.0, 0.0), Complex(1.0, 0.0), ok, F));
    CHECK(F.rows() == 3);
    Vertices bad(1, 3);
    bad << 0.3, 0.0, 1e-6;  // k_z > 0: the surface side of a plane with normal -z
    CHECK_THROWS_AS(post::huygens_radiate(pl, E, H, Complex(1.0, 0.0), Complex(1.0, 0.0), bad, F),
                    std::invalid_argument);
    bad << 0.0, 0.0, 0.0;
    CHECK_THROWS_AS(post::huygens_radiate(pl, E, H, Complex(1.0, 0.0), Complex(1.0, 0.0), bad, F),
                    std::invalid_argument);
    const post::HuygensPlane up =
        post::huygens_plane(0.0, 1.0, post::CurrentWindow{0.5, 1.0}, 0.25);
    Vertices down(1, 3);
    down << 0.0, 0.0, -1.0;
    CHECK_THROWS_AS(post::huygens_radiate(up, E, H, Complex(1.0, 0.0), Complex(1.0, 0.0), down, F),
                    std::invalid_argument);
    // E / H of the wrong size.
    const FieldMatrix E2 = FieldMatrix::Ones(pl.points.rows() + 1, 3);
    CHECK_THROWS_AS(post::huygens_radiate(pl, E2, H, Complex(1.0, 0.0), Complex(1.0, 0.0), ok, F),
                    std::invalid_argument);
}

TEST_CASE("huygens: option validation", "[huygens]") {
    const post::CurrentWindow w{0.5, 1.0};
    CHECK_THROWS_AS(post::huygens_plane(0.0, 0.0, w, 0.1), std::invalid_argument);
    CHECK_THROWS_AS(post::huygens_plane(0.0, 0.5, w, 0.1), std::invalid_argument);
    CHECK_THROWS_AS(post::huygens_plane(std::nan(""), -1.0, w, 0.1), std::invalid_argument);
    CHECK_THROWS_AS(post::huygens_plane(0.0, -1.0, w, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(post::huygens_plane(0.0, -1.0, w, -0.1), std::invalid_argument);
    CHECK_THROWS_AS(post::huygens_plane(0.0, -1.0, post::CurrentWindow{1.0, 0.5}, 0.1),
                    std::invalid_argument);
    CHECK_THROWS_AS(post::huygens_plane(0.0, -1.0, w, 1e-5), std::invalid_argument);  // too many

    const FlatPml c(kLam / 4.0);
    const post::SurfaceSolution s{&c.problem, random_currents(2 * c.space->size(), 11)};
    const post::CurrentWindow cw{c.om.taper_end, c.om.pml_start};
    post::FieldOptions fo;
    fo.window = cw;
    fo.huygens = post::HuygensOptions{};
    // Defaults: z_h = lambda_1, spacing lambda_1 / 4, the currents' window, normal -z.
    const post::HuygensPlane pl = post::huygens_plane(s, fo);
    CHECK(pl.normal_z == -1.0);
    CHECK(std::abs(pl.z - (-kLam)) < 1e-15);
    CHECK(std::abs(pl.spacing / (kLam / 4.0) - 1.0) < 1e-12);
    CHECK(pl.window.inner_half == cw.inner_half);
    CHECK(pl.window.outer_half == cw.outer_half);
    post::FieldOptions bad = fo;
    bad.window.reset();
    CHECK_THROWS_AS(post::huygens_plane(s, bad), std::invalid_argument);  // no window
    bad = fo;
    bad.huygens->z_h = 0.0;
    CHECK_THROWS_AS(post::huygens_plane(s, bad), std::invalid_argument);
    bad = fo;
    bad.huygens->spacing = 0.6 * kLam;  // > lambda_1 / 2
    CHECK_THROWS_AS(post::huygens_plane(s, bad), std::invalid_argument);
    bad = fo;
    bad.huygens->window = post::CurrentWindow{cw.inner_half, 1.01 * c.om.pml_start};
    CHECK_THROWS_AS(post::huygens_plane(s, bad), std::invalid_argument);  // over the PML
    // Explicit z_h / spacing / window are used.
    post::FieldOptions ex = fo;
    ex.huygens->z_h = 0.3 * kLam;
    ex.huygens->spacing = kLam / 6.0;
    ex.huygens->window = post::CurrentWindow{0.2e-6, 0.6e-6};
    const post::HuygensPlane pe = post::huygens_plane(s, ex);
    CHECK(std::abs(pe.z + 0.3 * kLam) < 1e-15);
    CHECK(pe.spacing == kLam / 6.0);
    CHECK(pe.window.outer_half == 0.6e-6);
    // Near-field functions refuse the Huygens options; exterior_scattered_field refuses a window.
    FieldMatrix E, H;
    Vertices pt(1, 3);
    pt << 0.1e-6, 0.0, -0.5e-6;
    post::FieldOptions only_h;
    only_h.huygens = post::HuygensOptions{};
    CHECK_THROWS_AS(post::exterior_scattered_field(s, pt, E, H, only_h), std::invalid_argument);
    post::FieldOptions only_w;
    only_w.window = cw;
    CHECK_THROWS_AS(post::exterior_scattered_field(s, pt, E, H, only_w), std::invalid_argument);
    // Points over the PML (beyond the stretch start on x, on y, on both; just beyond the start
    // line) are refused; a point on the start line itself is a real point and accepted. A point
    // in R2 is refused.
    const Real start = c.om.pml_start;
    for (const Vec3& q :
         {Vec3(1.2e-6, 0.0, -0.5e-6), Vec3(0.0, -1.2e-6, -0.5e-6), Vec3(-1.1e-6, 1.1e-6, -0.5e-6),
          Vec3(start * (1.0 + 1e-12), 0.0, -0.5e-6), Vec3(0.0, -start * (1.0 + 1e-12), -0.5e-6)}) {
        pt.row(0) = q.transpose();
        CHECK_THROWS_AS(post::exterior_scattered_field(s, pt, E, H), std::invalid_argument);
    }
    pt << start, -start, -0.5e-6;
    CHECK_NOTHROW(post::exterior_scattered_field(s, pt, E, H));
    CHECK(E.allFinite());
    pt << 0.1e-6, 0.0, 0.5e-6;
    CHECK_THROWS_AS(post::exterior_scattered_field(s, pt, E, H), std::invalid_argument);
    // A closed mesh has no Huygens plane.
    const geometry::TriangleMesh sphere = geometry::make_icosphere(1e-6, 0);
    const basis::RwgSpace ss(sphere);
    op::Problem sp;
    sp.space = &ss;
    sp.exterior = material::vacuum();
    sp.object = material::silver_500nm();
    sp.omega = omega_of(kLam);
    const post::SurfaceSolution so{&sp, random_currents(2 * ss.size(), 3)};
    post::FieldOptions fs;
    fs.huygens = post::HuygensOptions{};
    fs.huygens->window = post::CurrentWindow{0.5e-6, 1e-6};
    CHECK_THROWS_AS(post::huygens_plane(so, fs), std::invalid_argument);
}

TEST_CASE("huygens: the stretched kernel distance has Im R~ <= 0 and refuses growing kernels",
          "[huygens]") {
    // detail::stretched_distance, the invariant check of the stretched source samples in
    // exterior_scattered_field: for real observation points (|x|, |y| <= start) and source points
    // anywhere in the PML of the monotone stretch Im R~ <= 0 (random pairs, fixed seed); an
    // observation point beyond the stretch start on the source's side (what the observation-point
    // rule refuses) gives Im R~ > 0, a growing kernel, and throws std::logic_error.
    const FlatPml c(kLam / 4.0);
    const geometry::CoordinateStretch& st = c.stretch;
    const Real start = c.om.pml_start;
    const Real outer = c.om.outer_half;
    std::mt19937_64 rng(20261025);
    std::uniform_real_distribution<Real> u(-1.0, 1.0);
    std::uniform_real_distribution<Real> pml(start * (1.0 + 1e-9), outer);
    Real max_im = -1.0;
    Index stretched = 0;
    for (Index i = 0; i < 2000; ++i) {
        const Vec3 r(start * u(rng), start * u(rng), -kLam * (0.05 + std::abs(u(rng))));
        // Source in the PML along x, along y or both (corner), on either side.
        Vec3 q(start * u(rng), start * u(rng), 0.1e-6 * u(rng));
        const Index zone = i % 3;
        if (zone != 1)
            q.x() = std::copysign(pml(rng), u(rng));
        if (zone != 0)
            q.y() = std::copysign(pml(rng), u(rng));
        REQUIRE(st.in_pml(q));
        const Vec3c qt = st.map(q);
        stretched += (qt.x().imag() != 0.0 || qt.y().imag() != 0.0) ? 1 : 0;
        Complex R;
        REQUIRE_NOTHROW(R = post::detail::stretched_distance(r, qt, i));
        const Vec3c d = r.cast<Complex>() - qt;
        CHECK(std::abs(R - std::sqrt(d(0) * d(0) + d(1) * d(1) + d(2) * d(2))) == 0.0);
        CHECK(R.real() > 0.0);
        max_im = std::max(max_im, R.imag() / std::abs(R));
    }
    CHECK(stretched == 2000);
    CHECK(max_im <= 0.0);
    // Observation point beyond the start and beyond the source point on its side: Im R~ > 0.
    const Vec3c qt = st.map(Vec3(1.5e-6, 0.0, 0.0));
    REQUIRE(qt.x().imag() < 0.0);
    CHECK_THROWS_AS(post::detail::stretched_distance(Vec3(1.8e-6, 0.0, -0.5e-6), qt, 7),
                    std::logic_error);
    const Vec3c qy = st.map(Vec3(0.0, -1.5e-6, 0.0));
    CHECK_THROWS_AS(post::detail::stretched_distance(Vec3(0.0, -1.8e-6, -0.5e-6), qy, 8),
                    std::logic_error);
    // The opposite side is fine (the separation keeps its sign along the stretch).
    CHECK(post::detail::stretched_distance(Vec3(-1.8e-6, 0.0, -0.5e-6), qt, 9).imag() <= 0.0);
    // A real (unstretched) source point gives the real distance.
    const Complex Rr = post::detail::stretched_distance(Vec3(0.1e-6, 0.2e-6, -0.5e-6),
                                                        st.map(Vec3(0.4e-6, -0.2e-6, 0.0)), 10);
    CHECK(Rr.imag() == 0.0);
    CHECK(std::abs(Rr.real() - Vec3(0.3e-6, -0.4e-6, 0.5e-6).norm()) < 1e-20);
}

TEST_CASE("huygens: exterior field without a stretch is bitwise scattered_field", "[huygens]") {
    FlatPml c(kLam / 4.0);
    c.problem.stretch = nullptr;
    const post::SurfaceSolution s{&c.problem, random_currents(2 * c.space->size(), 5)};
    Vertices pts(3, 3);
    pts << 0.0, 0.0, -0.4e-6, 0.7e-6, -0.3e-6, -1e-6, 1.5e-6, 1.8e-6, -0.2e-6;
    FieldMatrix E1, H1, E2, H2;
    post::scattered_field(s, pts, E1, H1);
    post::exterior_scattered_field(s, pts, E2, H2);
    CHECK((E1.array() == E2.array()).all());
    CHECK((H1.array() == H2.array()).all());
}

TEST_CASE("huygens: stretched sources continue a dipole field through the PML", "[huygens]") {
    // RWG coefficients interpolated from the analytic continuation of the traces J = n x H,
    // M = -n x E (n = -z) of a dipole pair at height 0.3 lambda above the flat open surface:
    // J~(S(r)) on the stretched surface, carried to the real mesh by the Piola map
    // (J_real = det DS DS^-1 J~). The representation below the surface reproduces the dipole
    // field (source in R2: no incident field in R1). Next to the PML start the PML sources carry
    // a large part of it: dropping them (stretch removed, PML coefficients zero) fails.
    const FlatPml c(kLam / 16.0);
    const basis::RwgSpace& space = *c.space;
    const geometry::TriangleMesh& mesh = space.mesh();
    const Index N = space.size();
    const Vec3 r0(0.75e-6, 0.2e-6, 0.3e-6);
    const Vec3c p(Complex(1.0, 0.3), Complex(-0.5, 0.0), Complex(0.4, 0.2));
    const Vec3c m =
        constants::eta0 * Vec3c(Complex(0.2, 0.0), Complex(0.6, -0.1), Complex(0.0, 0.5));
    VectorXc x(2 * N);
    VectorXc x_trunc = VectorXc::Zero(2 * N);
    const Vec3c nz(0.0, 0.0, -1.0);
    for (Index n = 0; n < N; ++n) {
        const Vec3 a = mesh.vertices().row(mesh.edges()(n, 0)).transpose();
        const Vec3 b = mesh.vertices().row(mesh.edges()(n, 1)).transpose();
        const Vec3 mid = 0.5 * (a + b);
        const Vec3 f = space.value(n, mesh.edge_triangles()(n, 0), mid);
        const Vec3 e = (b - a).normalized();
        const Vec3 nu = f - f.dot(e) * e;  // unit normal of the edge (f . nu = 1)
        Vec3c E, H;
        dipole_fields(c.stretch.map(mid), r0, p, m, kLam, E, H);
        const Vec3c Jt = cross_cc(nz, H);
        const Vec3c Mt = -cross_cc(nz, E);
        const Vec3c ds = c.stretch.jacobian_diag(mid);
        const Vec3c piola(ds(1), ds(0), ds(0) * ds(1));  // det DS DS^-1
        const Vec3c Jr = piola.cwiseProduct(Jt);
        const Vec3c Mr = piola.cwiseProduct(Mt);
        const Vec3c nuc = nu.cast<Complex>();
        x(n) = Jr(0) * nuc(0) + Jr(1) * nuc(1) + Jr(2) * nuc(2);
        x(N + n) = Mr(0) * nuc(0) + Mr(1) * nuc(1) + Mr(2) * nuc(2);
        if (!c.stretch.in_pml(a) && !c.stretch.in_pml(b)) {
            x_trunc(n) = x(n);
            x_trunc(N + n) = x(N + n);
        }
    }
    Vertices pts(3, 3);
    pts << 0.95e-6, 0.2e-6, -0.25e-6,  // next to the PML start (x_start = 1 um)
        0.4e-6, -0.6e-6, -0.4e-6, 0.0, 0.0, -0.5e-6;
    const post::SurfaceSolution s{&c.problem, x};
    FieldMatrix E, H;
    post::exterior_scattered_field(s, pts, E, H);
    op::Problem plain = c.problem;
    plain.stretch = nullptr;
    const post::SurfaceSolution st{&plain, x_trunc};
    FieldMatrix Et, Ht;
    post::exterior_scattered_field(st, pts, Et, Ht);
    for (Index i = 0; i < pts.rows(); ++i) {
        Vec3c e0, h0;
        dipole_fields(pts.row(i).transpose().cast<Complex>(), r0, p, m, kLam, e0, h0);
        const Real err_e = (E.row(i).transpose() - e0).norm() / e0.norm();
        const Real err_h = (H.row(i).transpose() - h0).norm() / h0.norm();
        const Real trunc_e = (Et.row(i).transpose() - e0).norm() / e0.norm();
        WARN("point " << i << ": |dE|/|E| " << err_e << ", |dH|/|H| " << err_h
                      << "; without the PML sources " << trunc_e);
        CHECK(err_e < 2e-2);
        CHECK(err_h < 2e-2);
        if (i == 0)
            CHECK(trunc_e > 5.0 * err_e);
    }
}

TEST_CASE("huygens: the pml_sources diagnostic drops exactly the stretched source triangles",
          "[huygens]") {
    // pml_sources = false keeps only the real source triangles: the plane field is then
    // independent of the PML profile (bitwise for sigma_max 4 and 1, the PML zone is the same),
    // whereas the default (analytic continuation through the PML) depends on it.
    const FlatPml c(kLam / 4.0);
    op::Problem weak = c.problem;
    const geometry::CoordinateStretch st1 = geometry::make_pml_stretch(c.om, 1.0, 2.0);
    weak.stretch = &st1;
    const VectorXc x = random_currents(2 * c.space->size(), 31);
    const post::SurfaceSolution s4{&c.problem, x};
    const post::SurfaceSolution s1{&weak, x};
    post::FieldOptions fo;
    fo.window = post::CurrentWindow{c.om.taper_end, c.om.pml_start};
    fo.huygens = post::HuygensOptions{};
    fo.huygens->spacing = kLam / 3.0;
    fo.huygens->pml_sources = false;
    const post::HuygensPlaneField t4 = post::huygens_plane_field(s4, fo);
    const post::HuygensPlaneField t1 = post::huygens_plane_field(s1, fo);
    CHECK((t4.E.array() == t1.E.array()).all());
    CHECK((t4.H.array() == t1.H.array()).all());
    FieldMatrix E, H;
    post::exterior_scattered_field(s4, t4.plane.points, E, H, {}, /*pml_sources=*/false);
    CHECK((E.array() == t4.E.array()).all());
    fo.huygens->pml_sources = true;
    const post::HuygensPlaneField f4 = post::huygens_plane_field(s4, fo);
    const post::HuygensPlaneField f1 = post::huygens_plane_field(s1, fo);
    CHECK((f4.E - f1.E).norm() > 1e-6 * f4.E.norm());
    CHECK((f4.E - t4.E).norm() > 1e-3 * f4.E.norm());
}

TEST_CASE("huygens: plane fields and radiation are independent of the thread count", "[huygens]") {
#ifdef SPECKLEBEM_HAVE_OPENMP
    const FlatPml c(kLam / 4.0);
    const post::SurfaceSolution s{&c.problem, random_currents(2 * c.space->size(), 21)};
    post::FieldOptions fo;
    fo.window = post::CurrentWindow{c.om.taper_end, c.om.pml_start};
    fo.huygens = post::HuygensOptions{};
    fo.huygens->spacing = kLam / 3.0;
    const Vertices dirs = cone_directions(-1.0, {0.0, 30.0, 60.0, 89.0}, 6);
    const int saved = omp_get_max_threads();
    omp_set_num_threads(1);
    const post::HuygensPlaneField f1 = post::huygens_plane_field(s, fo);
    FieldMatrix F1;
    post::huygens_radiate(f1, dirs, F1);
    omp_set_num_threads(saved);
    const post::HuygensPlaneField fn = post::huygens_plane_field(s, fo);
    FieldMatrix Fn;
    post::huygens_radiate(fn, dirs, Fn);
    REQUIRE(f1.E.rows() == fn.E.rows());
    CHECK((f1.E.array() == fn.E.array()).all());
    CHECK((f1.H.array() == fn.H.array()).all());
    CHECK((F1.array() == Fn.array()).all());
    // post::far_field with FieldOptions::huygens is huygens_far_field.
    FieldMatrix Ff, Fh;
    post::far_field(s, dirs, Ff, fo);
    post::huygens_far_field(s, dirs, Fh, fo);
    CHECK((Ff.array() == Fn.array()).all());
    CHECK((Fh.array() == Fn.array()).all());
    // The PML sources contribute (a non-trivial stretched part was exercised).
    op::Problem plain = c.problem;
    plain.stretch = nullptr;
    FieldMatrix E0, H0;
    post::exterior_scattered_field(post::SurfaceSolution{&plain, s.currents}, f1.plane.points, E0,
                                   H0);
    CHECK((E0 - f1.E).norm() > 1e-3 * f1.E.norm());
#else
    SUCCEED("built without OpenMP");
#endif
}
