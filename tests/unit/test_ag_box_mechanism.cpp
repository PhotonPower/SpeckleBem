// Unit tests of the WP-V3 diagnostics of the Ag box sensitivity
// (tests/support/ag_box_mechanism_support.hpp, used by benchmarks/box_validity.cpp): face
// classification, sampling of the currents on a box face, matching of two face samplings,
// the radially binned 2D spectrum, the band peak, the SPP wavenumber, the interior probe lines,
// the line spectrum peak, the leading Hessenberg block and the travelling depth-wave fit.
#include "specklebem/basis/rwg.hpp"
#include "specklebem/core/path.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/io/result_writer.hpp"
#include "specklebem/material/material.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "ag_box_mechanism_support.hpp"

using namespace specklebem;
namespace abm = ag_box_mechanism;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr Real kLambda = 500e-9;
const Real kK0 = 2.0 * constants::pi / kLambda;

/// Samples on a square face [-L/2, L/2]^2 (cell centres, spacing h) with J = j_dir exp(-j kj . r)
/// and M = m_dir exp(-j km . r) in the (u, v) plane.
abm::FaceSamples plane_wave_samples(Real L, Real h, const Vec3& kj, const Vec3& km) {
    abm::FaceSamples s;
    const auto n = static_cast<Index>(std::llround(L / h));
    for (Index i = 0; i < n; ++i)
        for (Index j = 0; j < n; ++j) {
            const Real u = -0.5 * L + (static_cast<Real>(i) + 0.5) * h;
            const Real v = -0.5 * L + (static_cast<Real>(j) + 0.5) * h;
            s.points.emplace_back(u, v, 0.0);
            s.u.push_back(u);
            s.v.push_back(v);
            s.area.push_back(h * h);
            s.J.push_back(std::polar(1.0, -(kj.x() * u + kj.y() * v)) * Vec3c(0.0, 1.0, 0.0));
            s.M.push_back(std::polar(2.0, -(km.x() * u + km.y() * v)) * Vec3c(1.0, 0.0, 0.0));
        }
    return s;
}

/// Unique temporary directory, removed with its contents on destruction.
class TempDir {
public:
    TempDir() {
        static std::atomic<int> counter{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("specklebem_ag_box_mechanism_" + std::to_string(stamp) + "_" +
                 std::to_string(counter++));
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

}  // namespace

TEST_CASE("ag_box_mechanism: faces of the closed box by the normal", "[ag_box_mechanism]") {
    CHECK(abm::classify(Vec3(0.1, 0.0, -0.99)) == abm::Face::Top);
    CHECK(abm::classify(Vec3(0.0, 0.0, 1.0)) == abm::Face::Bottom);
    CHECK(abm::classify(Vec3(-1.0, 0.0, 0.0)) == abm::Face::WallXm);
    CHECK(abm::classify(Vec3(0.9, 0.3, 0.2)) == abm::Face::WallXp);
    CHECK(abm::classify(Vec3(0.2, -0.9, 0.1)) == abm::Face::WallYm);
    CHECK(abm::classify(Vec3(0.0, 1.0, 0.0)) == abm::Face::WallYp);
    const Vec3 r(1.0, 2.0, 3.0);
    CHECK(abm::face_coordinates(abm::Face::Top, r) == std::pair<Real, Real>(1.0, 2.0));
    CHECK(abm::face_coordinates(abm::Face::Bottom, r) == std::pair<Real, Real>(1.0, 2.0));
    CHECK(abm::face_coordinates(abm::Face::WallXp, r) == std::pair<Real, Real>(2.0, 3.0));
    CHECK(abm::face_coordinates(abm::Face::WallYm, r) == std::pair<Real, Real>(1.0, 3.0));
    CHECK(std::string(abm::face_name(abm::Face::WallYp)) == "wall+y");
}

TEST_CASE("ag_box_mechanism: currents sampled on the faces of a box mesh", "[ag_box_mechanism]") {
    // Flat 0.4 um patch, 0.1 um cells, depth 0.3 um: uniform box.
    geometry::HeightMap hm;
    hm.dx = hm.dy = 0.1e-6;
    hm.z = MatrixXr::Zero(5, 5);
    const geometry::TriangleMesh mesh = geometry::make_mesh_from_height_map(hm, 0.3e-6, 0.1e-6);
    const basis::RwgSpace space(mesh);
    op::Problem p;
    p.space = &space;
    // Constant tangential coefficients: values are then the RWG sums, which are checked against
    // ag_absorption::currents_at at the centroids.
    std::mt19937_64 rng(20261013);
    std::normal_distribution<Real> g(0.0, 1.0);
    VectorXc x(2 * space.size());
    for (Index i = 0; i < x.size(); ++i) x(i) = Complex(g(rng), g(rng));
    const post::SurfaceSolution s{&p, x};
    std::size_t total = 0;
    for (const abm::Face f : {abm::Face::Top, abm::Face::WallXm, abm::Face::WallXp,
                              abm::Face::WallYm, abm::Face::WallYp, abm::Face::Bottom}) {
        const abm::FaceSamples fs = abm::sample_face(s, f);
        total += fs.size();
        REQUIRE(fs.size() == (f == abm::Face::Top || f == abm::Face::Bottom ? 32u : 24u));
        for (std::size_t i = 0; i < fs.size(); ++i) {
            const auto [u, v] = abm::face_coordinates(f, fs.points[i]);
            CHECK(u == fs.u[i]);
            CHECK(v == fs.v[i]);
            CHECK_THAT(fs.area[i], WithinRel(0.005e-12, 1e-9));
        }
    }
    CHECK(total == static_cast<std::size_t>(mesh.num_triangles()));
    // The generator's top-face count 2 (n_x - 1)(n_y - 1) = 32 is checked when given.
    CHECK(abm::sample_face(s, abm::Face::Top, 32).size() == 32u);
    CHECK_THROWS_AS(abm::sample_face(s, abm::Face::Top, 31), std::runtime_error);
    const abm::FaceSamples top = abm::sample_face(s, abm::Face::Top);
    for (std::size_t i = 0; i < top.size(); ++i) {
        CHECK_THAT(top.points[i].z(), WithinAbs(0.0, 1e-15));
        // Index of that triangle: find by centroid.
        for (Index t = 0; t < mesh.num_triangles(); ++t) {
            if ((mesh.centroid(t) - top.points[i]).norm() > 1e-15)
                continue;
            const auto [J, M] = ag_absorption::currents_at(s, t, top.points[i]);
            CHECK((J - top.J[i]).norm() <= 1e-12 * J.norm());
            CHECK((M - top.M[i]).norm() <= 1e-12 * M.norm());
        }
    }
}

TEST_CASE("ag_box_mechanism: top-face currents transferred between boxes of different depth",
          "[ag_box_mechanism]") {
    geometry::HeightMap hm;
    hm.dx = hm.dy = 0.1e-6;
    hm.z = MatrixXr::Zero(5, 5);
    hm.z(2, 2) = 0.02e-6;  // a bump, so that the top face is not flat
    const geometry::TriangleMesh ma = geometry::make_mesh_from_height_map(hm, 0.3e-6, 0.1e-6);
    const geometry::TriangleMesh mb = geometry::make_mesh_from_height_map(hm, 0.5e-6, 0.1e-6);
    const basis::RwgSpace sa(ma), sb(mb);
    std::mt19937_64 rng(11);
    std::normal_distribution<Real> g(0.0, 1.0);
    VectorXc xa(2 * sa.size());
    for (Index i = 0; i < xa.size(); ++i) xa(i) = Complex(g(rng), g(rng));
    const auto [xb, nb] = abm::transfer_face_currents(sa, xa, sb, abm::Face::Top, 1e-12);
    // Top-face basis functions of a 4 x 4 grid: 16 diagonals + 2 x 12 inner grid edges.
    CHECK(nb == 40);
    const auto [xa_top, na] = abm::transfer_face_currents(sa, xa, sa, abm::Face::Top, 1e-12);
    CHECK(na == 40);
    const auto [back, nback] = abm::transfer_face_currents(sb, xb, sa, abm::Face::Top, 1e-12);
    CHECK(nback == 40);
    CHECK((back - xa_top).norm() == 0.0);
    // The transferred expansion equals the top-only expansion of a on every top triangle (neither
    // carries the rim functions that straddle top and wall), which also checks the signs.
    op::Problem pa, pb;
    pa.space = &sa;
    pb.space = &sb;
    const post::SurfaceSolution s_a{&pa, xa_top}, s_b{&pb, xb};
    int checked = 0;
    for (Index t = 0; t < ma.num_triangles(); ++t) {
        if (abm::classify(ma.normal(t)) != abm::Face::Top)
            continue;
        for (Index u = 0; u < mb.num_triangles(); ++u) {
            if ((mb.centroid(u) - ma.centroid(t)).norm() > 1e-15)
                continue;
            const auto [Ja, Ma] = ag_absorption::currents_at(s_a, t, ma.centroid(t));
            const auto [Jb, Mb] = ag_absorption::currents_at(s_b, u, mb.centroid(u));
            CHECK((Ja - Jb).norm() <= 1e-12 * (1.0 + Ja.norm()));
            CHECK((Ma - Mb).norm() <= 1e-12 * (1.0 + Ma.norm()));
            ++checked;
        }
    }
    CHECK(checked == 32);
    CHECK_THROWS_AS(abm::transfer_face_currents(sa, VectorXc(3), sb, abm::Face::Top, 1e-12),
                    std::invalid_argument);
}

TEST_CASE("ag_box_mechanism: transfer with exchanged plus / minus triangles and a changed mesh",
          "[ag_box_mechanism]") {
    geometry::HeightMap hm;
    hm.dx = hm.dy = 0.1e-6;
    hm.z = MatrixXr::Zero(5, 5);
    hm.z(2, 2) = 0.02e-6;
    const geometry::TriangleMesh ma = geometry::make_mesh_from_height_map(hm, 0.3e-6, 0.1e-6);
    // The same geometry with the vertex numbering reversed: every edge (a, b), a < b, becomes
    // (b', a') with b' < a', so plus and minus triangles are exchanged for every function.
    const Index nv = ma.num_vertices();
    Vertices vr(nv, 3);
    for (Index v = 0; v < nv; ++v) vr.row(nv - 1 - v) = ma.vertices().row(v);
    Triangles tr = ma.triangles();
    for (Index t = 0; t < tr.rows(); ++t)
        for (int c = 0; c < 3; ++c) tr(t, c) = nv - 1 - tr(t, c);
    const geometry::TriangleMesh mr(vr, tr);
    const basis::RwgSpace sa(ma), sr(mr);
    std::mt19937_64 rng(13);
    std::normal_distribution<Real> g(0.0, 1.0);
    VectorXc xa(2 * sa.size());
    for (Index i = 0; i < xa.size(); ++i) xa(i) = Complex(g(rng), g(rng));
    const auto [xa_top, na] = abm::transfer_face_currents(sa, xa, sa, abm::Face::Top, 1e-12);
    const auto [xr, nr] = abm::transfer_face_currents(sa, xa, sr, abm::Face::Top, 1e-12);
    REQUIRE(nr == 40);
    REQUIRE(na == 40);
    // Every matched function has the opposite orientation (coefficient -x): the same current.
    int nonzero = 0;
    for (Index n = 0; n < sr.size(); ++n) nonzero += xr(n) != Complex(0.0, 0.0) ? 1 : 0;
    CHECK(nonzero == 40);
    CHECK_THAT(xr.norm(), WithinRel(xa_top.norm(), 1e-14));  // other summation order
    op::Problem pa, pr;
    pa.space = &sa;
    pr.space = &sr;
    const post::SurfaceSolution s_a{&pa, xa_top}, s_r{&pr, xr};
    int checked = 0;
    for (Index t = 0; t < ma.num_triangles(); ++t) {
        if (abm::classify(ma.normal(t)) != abm::Face::Top)
            continue;
        // Triangle t of ma is triangle t of mr (same rows, renumbered vertices).
        const auto [Ja, Ma] = ag_absorption::currents_at(s_a, t, ma.centroid(t));
        const auto [Jr, Mr] = ag_absorption::currents_at(s_r, t, mr.centroid(t));
        CHECK((Ja - Jr).norm() <= 1e-12 * (1.0 + Ja.norm()));
        CHECK((Ma - Mr).norm() <= 1e-12 * (1.0 + Ma.norm()));
        ++checked;
    }
    CHECK(checked == 32);
    const auto [back, nback] = abm::transfer_face_currents(sr, xr, sa, abm::Face::Top, 1e-12);
    CHECK(nback == 40);
    CHECK((back - xa_top).norm() == 0.0);

    // A mesh that differs at one top vertex: the functions on the edges opposite that vertex
    // keep their midpoints but one of their triangles changes, which must be refused.
    geometry::HeightMap hc = hm;
    hc.z(1, 2) = 0.01e-6;
    const geometry::TriangleMesh mc = geometry::make_mesh_from_height_map(hc, 0.3e-6, 0.1e-6);
    const basis::RwgSpace sc(mc);
    CHECK_THROWS_AS(abm::transfer_face_currents(sa, xa, sc, abm::Face::Top, 1e-12),
                    std::runtime_error);
}

TEST_CASE("ag_box_mechanism: matching and differencing two samplings", "[ag_box_mechanism]") {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<Real> u(-1e-6, 1e-6);
    std::vector<Vec3> a, b;
    for (int i = 0; i < 50; ++i) a.emplace_back(u(rng), u(rng), u(rng));
    // b: every second point of a (jittered below the tolerance, shuffled) plus foreign points.
    for (int i = 0; i < 50; i += 2) b.push_back(a[static_cast<std::size_t>(i)] + Vec3(3e-12, 0, 0));
    for (int i = 0; i < 10; ++i) b.emplace_back(u(rng) + 5e-6, u(rng), u(rng));
    std::shuffle(b.begin(), b.end(), rng);
    const auto pairs = abm::match_points(a, b, 5e-11);
    REQUIRE(pairs.size() == 25);
    for (const auto& [i, j] : pairs) {
        CHECK(i % 2 == 0);
        CHECK((a[i] - b[j]).norm() < 1e-11);
    }
    CHECK(abm::match_points(a, b, 1e-12).empty());
    CHECK_THROWS_AS(abm::match_points(a, b, 0.0), std::invalid_argument);

    abm::FaceSamples sa, sb;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sa.points.push_back(a[i]);
        sa.u.push_back(a[i].x());
        sa.v.push_back(a[i].y());
        sa.area.push_back(1.0);
        sa.J.emplace_back(Complex(static_cast<Real>(i), 0), 0, 0);
        sa.M.emplace_back(0, Complex(0, static_cast<Real>(i)), 0);
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        sb.points.push_back(b[i]);
        sb.u.push_back(b[i].x());
        sb.v.push_back(b[i].y());
        sb.area.push_back(1.0);
        sb.J.emplace_back(Complex(1.0, 0), 0, 0);
        sb.M.emplace_back(0, Complex(0, 1.0), 0);
    }
    const abm::FaceSamples d = abm::difference(sa, sb, pairs);
    const abm::FaceSamples c = abm::subset(sa, pairs);
    REQUIRE(d.size() == 25);
    REQUIRE(c.size() == 25);
    for (std::size_t q = 0; q < d.size(); ++q) {
        const auto i = static_cast<Real>(pairs[q].first);
        CHECK(d.J[q](0) == Complex(i - 1.0, 0));
        CHECK(d.M[q](1) == Complex(0, i - 1.0));
        CHECK(c.J[q](0) == Complex(i, 0));
        CHECK(d.points[q] == a[pairs[q].first]);
    }
}

TEST_CASE("ag_box_mechanism: radial spectrum of plane-wave currents", "[ag_box_mechanism]") {
    const Real L = 4e-6, h = 50e-9;
    SECTION("evanescent J at 1.3 k0, propagating M at 0.5 k0") {
        const Vec3 kj = 1.3 * kK0 * Vec3(std::cos(0.4), std::sin(0.4), 0.0);
        const Vec3 km = 0.5 * kK0 * Vec3(0.0, -1.0, 0.0);
        const abm::FaceSamples s = plane_wave_samples(L, h, kj, km);
        const abm::RadialSpectrum sp = abm::radial_spectrum(s, kK0, 2.5, 1.0 / 32.0, 0.025, true);
        Real sj = 0, sm = 0;
        for (std::size_t b = 0; b < sp.j.size(); ++b) {
            sj += sp.j[b];
            sm += sp.m[b];
        }
        CHECK_THAT(sj, WithinAbs(1.0, 1e-12));
        CHECK_THAT(sm, WithinAbs(1.0, 1e-12));
        CHECK(sp.j_above > 0.99);
        CHECK(sp.m_above < 1e-3);
        const abm::BandPeak pj = abm::band_peak(sp.bin_centre, sp.j, 0.8, 1.8);
        const abm::BandPeak pm = abm::band_peak(sp.bin_centre, sp.m, 0.2, 0.8);
        CHECK_THAT(pj.peak, WithinAbs(1.3, 0.0251));
        CHECK_THAT(pj.centroid, WithinAbs(1.3, 0.02));
        CHECK(pj.fraction > 0.99);
        CHECK_THAT(pm.peak, WithinAbs(0.5, 0.0251));
        CHECK_THAT(sp.rms_j, WithinRel(1.0, 1e-12));
        CHECK_THAT(sp.rms_m, WithinRel(2.0, 1e-12));
        CHECK_THAT(sp.k_max, WithinAbs(2.5, 1e-12));
        CHECK(sp.bin_centre.size() == 142);
    }
    SECTION("the SPP wavenumber is separated from k0 by the ring location") {
        const Real spp = abm::spp_ratio(material::silver_500nm().eps_r).real();
        const abm::FaceSamples s =
            plane_wave_samples(L, h, spp * kK0 * Vec3(1, 0, 0), 0.9 * kK0 * Vec3(0, 1, 0));
        const abm::RadialSpectrum sp = abm::radial_spectrum(s, kK0, 2.5, 1.0 / 32.0, 0.025, true);
        const abm::BandPeak pj = abm::band_peak(sp.bin_centre, sp.j, 0.8, 1.4);
        const abm::BandPeak pm = abm::band_peak(sp.bin_centre, sp.m, 0.6, 1.2);
        CHECK_THAT(pj.centroid, WithinAbs(spp, 0.02));
        CHECK_THAT(pm.centroid, WithinAbs(0.9, 0.02));
        // The Hann main lobe (half width 0.25 k0 at L = 8 lambda) leaks across k0.
        CHECK(sp.j_above > 0.6);
        CHECK(sp.m_above < 0.25);
    }
    SECTION("rectangular window and errors") {
        const abm::FaceSamples s = plane_wave_samples(L, h, Vec3::Zero(), Vec3::Zero());
        const abm::RadialSpectrum sp = abm::radial_spectrum(s, kK0, 1.0, 1.0 / 16.0, 0.05, false);
        // A constant current: the sinc^2 main lobe around k = 0 (zeros at 2 pi / L = k0 / 8).
        CHECK(abm::band_peak(sp.bin_centre, sp.j, 0.0, 0.2).fraction > 0.8);
        CHECK(sp.j_above < 0.01);
        abm::FaceSamples few;
        few.u = {0, 1, 2};
        few.v = {0, 1, 2};
        few.points.resize(3);
        few.area = {1, 1, 1};
        few.J.resize(3);
        few.M.resize(3);
        CHECK_THROWS_AS(abm::radial_spectrum(few, kK0, 1.0, 0.1, 0.1, true), std::invalid_argument);
        CHECK_THROWS_AS(abm::radial_spectrum(s, 0.0, 1.0, 0.1, 0.1, true), std::invalid_argument);
        CHECK_THROWS_AS(abm::radial_spectrum(s, kK0, 1.0, 0.0, 0.1, true), std::invalid_argument);
    }
}

TEST_CASE("ag_box_mechanism: band peak and SPP wavenumber", "[ag_box_mechanism]") {
    const std::vector<Real> c = {0.1, 0.2, 0.3, 0.4};
    const std::vector<Real> p = {0.4, 0.1, 0.3, 0.2};
    const abm::BandPeak b = abm::band_peak(c, p, 0.15, 0.45);
    CHECK(b.peak == 0.3);
    CHECK_THAT(b.fraction, WithinAbs(0.6, 1e-15));
    CHECK_THAT(b.centroid, WithinAbs((0.02 + 0.09 + 0.08) / 0.6, 1e-15));
    CHECK_THROWS_AS(abm::band_peak(c, p, 0.5, 0.6), std::invalid_argument);
    CHECK_THROWS_AS(abm::band_peak(c, {1.0}, 0.0, 1.0), std::invalid_argument);

    // Ag at 500 nm: Re k_spp / k0 = 1.0553, Im < 0 (exp(+jwt), amplitude decay ~ 41 um).
    const Complex r = abm::spp_ratio(Complex(-9.794, -0.313));
    CHECK_THAT(r.real(), WithinAbs(1.0553, 1e-4));
    CHECK(r.imag() < 0.0);
    CHECK_THAT(1.0 / (-r.imag() * kK0), WithinRel(41.5e-6, 0.01));
    // Lossless Drude-like metal: real ratio sqrt(eps / (1 + eps)) > 1.
    CHECK_THAT(abm::spp_ratio(Complex(-4.0, 0.0)).real(), WithinAbs(std::sqrt(4.0 / 3.0), 1e-14));
    CHECK_THROWS_AS(abm::spp_ratio(Complex(-1.0, 0.0)), std::invalid_argument);
}

TEST_CASE("ag_box_mechanism: interior probe lines follow the WP-A2 rule", "[ag_box_mechanism]") {
    const Real L = 4e-6, D = 2e-6, h = 50e-9, step = 20e-9;
    const auto lines = abm::interior_probe_lines(L, D, 0.15e-6, h, step, {0.5e-6, 1.0e-6, 5e-6});
    // x at z_p, z_p + 0.5, z_p + 1.0 (z_p + 5 is below the bottom and dropped), x at the
    // bottom line, y at z_p, vertical.
    REQUIRE(lines.size() == 6);
    const Real zp = 0.15e-6 + 3 * h;
    CHECK(lines[0].name == "x");
    CHECK_THAT(lines[0].level, WithinAbs(zp, 1e-18));
    CHECK_THAT(lines[1].level, WithinAbs(zp + 0.5e-6, 1e-18));
    CHECK_THAT(lines[2].level, WithinAbs(zp + 1.0e-6, 1e-18));
    CHECK_THAT(lines[3].level, WithinAbs(D - 3 * h, 1e-18));
    CHECK(lines[4].name == "y");
    CHECK(lines[5].name == "z");
    for (const auto& l : lines) {
        REQUIRE(static_cast<Index>(l.coordinate.size()) == l.points.rows());
        for (Index i = 0; i < l.points.rows(); ++i) {
            CHECK(std::abs(l.points(i, 0)) <= 0.5 * L - 3 * h + 1e-15);
            CHECK(std::abs(l.points(i, 1)) <= 0.5 * L - 3 * h + 1e-15);
            CHECK(l.points(i, 2) >= zp - 1e-15);
            CHECK(l.points(i, 2) <= D - 3 * h + 1e-15);
        }
    }
    CHECK(lines[0].points.rows() == 186);  // |x| <= 1.85 um at 20 nm
    CHECK_THAT(lines[0].points(0, 0), WithinAbs(-1.85e-6, 1e-15));
    CHECK_THAT(lines[5].coordinate.front(), WithinAbs(zp, 1e-18));
    CHECK(lines[5].coordinate.back() <= D - 3 * h + 1e-15);
    CHECK(lines[5].coordinate.back() > D - 3 * h - step);
    CHECK_THROWS_AS(abm::interior_probe_lines(L, 0.4e-6, 0.15e-6, h, step, {}),
                    std::invalid_argument);
    CHECK_THROWS_AS(abm::interior_probe_lines(0.2e-6, D, 0.0, h, step, {}), std::invalid_argument);
    CHECK_THROWS_AS(abm::interior_probe_lines(L, D, 0.0, h, step, {-1.0}), std::invalid_argument);
    CHECK_THROWS_AS(abm::interior_probe_lines(L, D, 0.0, h, 0.0, {}), std::invalid_argument);
}

TEST_CASE("ag_box_mechanism: line spectrum peak and rms", "[ag_box_mechanism]") {
    std::vector<Real> u;
    Eigen::Matrix<Complex, Eigen::Dynamic, 3> E(201, 3);
    for (Index i = 0; i < 201; ++i) {
        u.push_back(-2e-6 + 20e-9 * static_cast<Real>(i));
        E.row(i) = (std::polar(1.0, -0.7 * kK0 * u.back()) * Vec3c(0, 1, 0)).transpose();
    }
    std::vector<Real> ks;
    for (int i = -100; i <= 100; ++i) ks.push_back(0.02 * i * kK0);
    const auto [k, rel] = abm::line_peak(E, u, ks);
    CHECK_THAT(k / kK0, WithinAbs(0.7, 1e-9));
    CHECK_THAT(rel, WithinAbs(1.0, 1e-12));
    CHECK_THAT(abm::rms(E), WithinAbs(1.0, 1e-14));
    CHECK(abm::rms(Eigen::Matrix<Complex, Eigen::Dynamic, 3>(0, 3)) == 0.0);
    CHECK_THROWS_AS(abm::line_peak(E, {0.0}, ks), std::invalid_argument);
    CHECK_THROWS_AS(abm::line_peak(E, u, {}), std::invalid_argument);
}

TEST_CASE("ag_box_mechanism: leading Hessenberg block", "[ag_box_mechanism]") {
    MatrixXc h = MatrixXc::Zero(6, 5);
    for (Index j = 0; j < 5; ++j)
        for (Index i = 0; i <= j + 1; ++i)
            h(i, j) = Complex(static_cast<Real>(i + 1), static_cast<Real>(j));
    const MatrixXc l = abm::leading_hessenberg(h, 3);
    REQUIRE(l.rows() == 4);
    REQUIRE(l.cols() == 3);
    CHECK(l == h.topLeftCorner(4, 3));
    CHECK(abm::leading_hessenberg(h, 5) == h);
    CHECK_THROWS_AS(abm::leading_hessenberg(h, 0), std::invalid_argument);
    CHECK_THROWS_AS(abm::leading_hessenberg(h, 6), std::invalid_argument);
    CHECK_THROWS_AS(abm::leading_hessenberg(MatrixXc::Zero(5, 5), 2), std::invalid_argument);
}

TEST_CASE("ag_box_mechanism: travelling depth-wave fit", "[ag_box_mechanism]") {
    std::vector<Real> depth;
    for (int i = 0; i < 9; ++i) depth.push_back(2.0e-6 + 0.05e-6 * i);
    const Real beta0 = 2.11 * kK0;
    std::mt19937_64 rng(42);
    std::normal_distribution<Real> g(0.0, 1.0);
    const Index cols = 300;
    VectorXc f_inf(cols), a(cols);
    for (Index c = 0; c < cols; ++c) {
        f_inf(c) = Complex(g(rng), g(rng));
        a(c) = 0.01 * Complex(g(rng), g(rng));
    }
    MatrixXc F(static_cast<Index>(depth.size()), cols);
    for (Index r = 0; r < F.rows(); ++r)
        F.row(r) =
            (f_inf + std::polar(1.0, -beta0 * depth[static_cast<std::size_t>(r)]) * a).transpose();
    std::vector<Real> betas;
    for (int i = 0; i <= 600; ++i) betas.push_back(0.005 * i * kK0);
    SECTION("exact model") {
        const auto [b, res] = abm::fit_depth_wave(F, depth, betas);
        CHECK_THAT(b / kK0, WithinAbs(2.11, 1e-9));
        CHECK(res < 1e-20);
        // A wrong beta leaves most of the depth-dependent part.
        CHECK(abm::depth_wave_residual(F, depth, 1.0 * kK0) > 0.3);
        // beta = 0 is the constant model: residual 1 (the mean is removed exactly).
        CHECK_THAT(abm::depth_wave_residual(F, depth, 0.0), WithinAbs(1.0, 1e-12));
    }
    SECTION("noisy model") {
        MatrixXc N = F;
        for (Index r = 0; r < N.rows(); ++r)
            for (Index c = 0; c < cols; ++c) N(r, c) += 1e-3 * Complex(g(rng), g(rng));
        const auto [b, res] = abm::fit_depth_wave(N, depth, betas);
        CHECK_THAT(b / kK0, WithinAbs(2.11, 0.011));
        CHECK(res < 0.05);
    }
    CHECK_THROWS_AS(abm::depth_wave_residual(F.topRows(2), {2e-6, 2.1e-6}, 1.0),
                    std::invalid_argument);
    CHECK_THROWS_AS(abm::depth_wave_residual(MatrixXc::Ones(3, 4), {1, 2, 3}, 1.0),
                    std::invalid_argument);
    CHECK_THROWS_AS(abm::fit_depth_wave(F, depth, {}), std::invalid_argument);
}

TEST_CASE("ag_box_mechanism: .npy and attributes of a stored run round-trip",
          "[ag_box_mechanism]") {
    TempDir tmp;
    const std::string dir = core::path_to_utf8(tmp.path() / "run");
    std::mt19937_64 rng(20261013);
    std::normal_distribution<Real> g(0.0, 1.0);
    MatrixXr r(7, 3);
    MatrixXc c(5, 4);
    VectorXc v(11);
    for (Index i = 0; i < r.size(); ++i) r(i) = g(rng);
    for (Index i = 0; i < c.size(); ++i) c(i) = Complex(g(rng), g(rng));
    for (Index i = 0; i < v.size(); ++i) v(i) = Complex(g(rng), g(rng));
    {
        auto w = io::open_npy_directory(dir);
        w->write_matrix("real", r);
        w->write_matrix("complex", c);
        w->write_vector("vector", v);
        w->write_attribute("material", "ag");
        w->write_attribute("box_depth", 2.05e-6);
        w->write_attribute("seed", 1.0);
        w->write_attribute("eps_r_imag", -0.313);
    }
    const std::filesystem::path d = core::path_from_utf8(dir);
    // Bit-exact: the reader maps the C-order little-endian data of the writer.
    CHECK(abm::read_npy<Real>(d / "real.npy") == r);
    CHECK(abm::read_npy<Complex>(d / "complex.npy") == c);
    const MatrixXc vr = abm::read_npy<Complex>(d / "vector.npy");
    REQUIRE(vr.cols() == 1);
    CHECK(VectorXc(vr.col(0)) == v);
    CHECK_THROWS_AS(abm::read_npy<Complex>(d / "real.npy"), std::runtime_error);  // dtype
    CHECK_THROWS_AS(abm::read_npy<Real>(d / "missing.npy"), std::runtime_error);
    {
        std::ofstream bad(d / "bad.npy", std::ios::binary);
        bad << "not numpy";
    }
    CHECK_THROWS_AS(abm::read_npy<Real>(d / "bad.npy"), std::runtime_error);

    const abm::Attributes a = abm::read_attributes(d / "attributes.json");
    CHECK(abm::find_attribute(a, "material") == std::optional<std::string>("ag"));
    REQUIRE(abm::find_attribute(a, "box_depth").has_value());
    CHECK(std::stod(*abm::find_attribute(a, "box_depth")) == 2.05e-6);
    CHECK(std::stod(*abm::find_attribute(a, "eps_r_imag")) == -0.313);
    CHECK_FALSE(abm::find_attribute(a, "tag").has_value());
    CHECK_THROWS_AS(abm::read_attributes(d / "missing.json"), std::runtime_error);
}

TEST_CASE("ag_box_mechanism: stored attributes against the configuration", "[ag_box_mechanism]") {
    CHECK(abm::same_attribute_value("1.0", "1"));
    CHECK(abm::same_attribute_value("4e-06", "4.0000000000000001e-06"));
    CHECK(abm::same_attribute_value("2.05e-06", "2.0500000000000002e-06"));
    CHECK_FALSE(abm::same_attribute_value("2.05e-06", "2.1e-06"));
    CHECK_FALSE(abm::same_attribute_value("0", "5e-08"));
    CHECK(abm::same_attribute_value("0.0", "0"));
    CHECK(abm::same_attribute_value("uniform", "uniform"));
    CHECK_FALSE(abm::same_attribute_value("uniform", "auto"));
    CHECK_FALSE(abm::same_attribute_value("1e-6x", "1e-6"));  // not a number: compared as text

    const abm::Attributes stored = {{"material", "ag"},        {"rms_roughness", "0.0"},
                                    {"seed", "1.0"},           {"map_L", "4.8e-06"},
                                    {"waist_radius", "1e-06"}, {"box_mesh_size", "uniform"},
                                    {"eps_r_real", "-9.794"},  {"eps_r_imag", "-0.313"},
                                    {"box_depth", "4e-06"},    {"iterations", "1030.0"}};
    abm::Attributes expected = {{"material", "ag"},
                                {"rms_roughness", "0"},
                                {"seed", "1"},
                                {"map_L", "4.7999999999999998e-06"},
                                {"waist_radius", "1e-06"},
                                {"box_mesh_size", "uniform"},
                                {"eps_r_real", "-9.794"},
                                {"eps_r_imag", "-0.313"},
                                {"box_depth", "4e-06"}};
    CHECK(abm::attribute_mismatches(stored, expected).empty());
    CHECK_NOTHROW(abm::require_attributes(stored, expected, "--load-currents A"));
    // sigma, seed and a missing key each give one message, all in one exception.
    expected[1].second = "5e-08";
    expected[2].second = "2";
    expected.emplace_back("fine_band", "none");
    const std::vector<std::string> m = abm::attribute_mismatches(stored, expected);
    REQUIRE(m.size() == 3);
    CHECK(m[0] == "rms_roughness: stored 0.0, expected 5e-08");
    CHECK(m[1] == "seed: stored 1.0, expected 2");
    CHECK(m[2] == "fine_band: missing, expected none");
    bool thrown = false;
    try {
        abm::require_attributes(stored, expected, "--load-currents A");
    } catch (const std::runtime_error& e) {
        thrown = true;
        const std::string what = e.what();
        CHECK(what.find("--load-currents A") != std::string::npos);
        CHECK(what.find("rms_roughness") != std::string::npos);
        CHECK(what.find("fine_band") != std::string::npos);
    }
    CHECK(thrown);
}

TEST_CASE("ag_box_mechanism: runs of a depth fit differ in the depth only", "[ag_box_mechanism]") {
    const abm::Attributes a = {
        {"material", "ag"},     {"rms_roughness", "5e-08"},   {"seed", "1.0"},
        {"box_depth", "2e-06"}, {"gmres_tolerance", "1e-05"}, {"iterations", "2096.0"},
        {"tag", "ag4_d20"}};
    abm::Attributes b = a, c = a;
    b[3].second = "2.1e-06";
    b[5].second = "1918.0";
    b[6].second = "ag4_d21";
    c[3].second = "4e-06";
    const std::vector<std::string> names = {"A", "B", "C"};
    // Outcome keys (iterations, tag) may differ; the depth is ignored.
    CHECK_NOTHROW(abm::require_same_configuration({a, b, c}, names, {"box_depth"}));
    CHECK_THROWS_AS(abm::require_same_configuration({a, b, c}, names, {}), std::runtime_error);
    abm::Attributes flat = c;
    flat[1].second = "0.0";  // a flat run among rough ones
    CHECK_THROWS_AS(abm::require_same_configuration({a, b, flat}, names, {"box_depth"}),
                    std::runtime_error);
    abm::Attributes no_tol = c;
    no_tol.erase(no_tol.begin() + 4);  // gmres_tolerance missing in one run only
    CHECK_THROWS_AS(abm::require_same_configuration({a, b, no_tol}, names, {"box_depth"}),
                    std::runtime_error);
    // A key absent from every run is not compared.
    abm::Attributes a2 = a, b2 = b;
    a2.erase(a2.begin() + 4);
    b2.erase(b2.begin() + 4);
    CHECK_NOTHROW(abm::require_same_configuration({a2, b2}, {"A", "B"}, {"box_depth"}));
    CHECK_THROWS_AS(abm::require_same_configuration({a}, {"A"}, {}), std::invalid_argument);
    CHECK_THROWS_AS(abm::require_same_configuration({a, b}, {"A"}, {}), std::invalid_argument);
}
