#pragma once
/// @file ag_box_mechanism_support.hpp
/// Diagnostics of the Ag box sensitivity (WP-V3, record benchmarks/results/ag_box_mechanism.md):
/// do surface plasmons (SPPs) or the cavity resonances of the closed box (vacuum inside, Ag walls,
/// WP-A2) make the far field of a truncated rough Ag patch depend on the box depth and on L?
/// Shared by benchmarks/box_validity.cpp and the unit test tests/unit/test_ag_box_mechanism.cpp;
/// no Catch2.
///
/// Conventions (docs/06): exp(+jwt), R1 = vacuum z < xi(x, y), R2 = object, n out of R2 into R1,
/// J = n x H and M = -n x E on the R1 side. Box faces by the unit normal of a triangle: top face
/// n_z < -0.5, bottom plate n_z > 0.5, otherwise the wall whose outward normal is closest to
/// -x, +x, -y or +y (the classification of box_validity's PARTS diagnostic, with the walls split).
///
///  * Spatial spectrum of the currents on a face (experiment (i)): the area-weighted 2D DFT
///    X(k_u, k_v) = sum_t w_t A_t X(c_t) exp(+j (k_u u_t + k_v v_t)) over the triangle centroids
///    c_t, in the in-face coordinates (u, v) of the face (top / bottom: (x, y); x walls: (y, z);
///    y walls: (x, z)), with the separable Hann taper w_t = sin^2(pi (u - u_min) / (u_max - u_min))
///    sin^2(pi (v - v_min) / (v_max - v_min)) (or w_t = 1). A current exp(-j k0 . r) peaks at
///    +k0. The power |X|^2 (Euclidean norm of the Cartesian components) on a square grid of
///    spacing dk up to |k_u|, |k_v| <= k_max is binned by |k| / k0: every bin holds the fraction of
///    the grid power in its annulus, and above_k0 the fraction with |k| > k0 (evanescent in
///    vacuum: SPPs sit at Re k_spp = 1.055 k0 for Ag at 500 nm, propagating cavity fields at
///    |k| < k0). The Hann main lobe has a half width of 2 (2 pi / extent), 0.25 k0 for a 4 um face
///    at 500 nm: a single ring is located far more precisely than that, two components closer than
///    the half width are not resolved.
///  * Matching of the samples of two solutions on a common face region (same top-face triangles,
///    e.g. depth 2 and 4 um, or the central L x L part of a 1.2 L patch) by their centroids, so
///    that the spectrum of the current difference can be taken.
///  * Interior probe lines of the extinction-theorem residual (experiment (ii), WP-A2 placement
///    rule): inside R2 at z_p = max xi + m (m = 3 h), laterally |x|, |y| <= L / 2 - m, and down to
///    the bottom plate (z <= depth - m).
///  * Leading (j + 1) x j block of a GMRES Hessenberg: the Arnoldi relation after j steps, so its
///    harmonic Ritz values are those GMRES had at iteration j (experiment (iv), equal-k
///    comparison of solves with different iteration counts).
///  * Fit of a travelling depth wave to far fields at several box depths (experiment (iii)):
///    F_i ~ F_inf + A exp(-j beta D_i) with direction-independent beta, for each beta of a grid the
///    least-squares residual over all directions and components, relative to the depth-dependent
///    part sum_i |F_i - mean F|^2. A wave that runs down the walls and back with the SPP wavenumber
///    gives beta = 2 Re k_spp = 2.11 k0, a vacuum standing wave of the box interior 2 k_z < 2 k0.
///  * Stored run directories (io::open_npy_directory): .npy and attributes.json readers, and the
///    checks that a stored run matches the command-line configuration (--load-currents, --diff)
///    or that several runs differ only in the box depth (--depth-fit).
#include "specklebem/basis/rwg.hpp"
#include "specklebem/core/path.hpp"
#include "specklebem/core/types.hpp"
#include "specklebem/geometry/mesh.hpp"
#include "specklebem/postprocessing/fields.hpp"

#include <Eigen/Core>
#include <Eigen/QR>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ag_absorption_support.hpp"

namespace ag_box_mechanism {

using namespace specklebem;

/// Face of the closed box.
enum class Face : int { Top = 0, WallXm = 1, WallXp = 2, WallYm = 3, WallYp = 4, Bottom = 5 };

inline const char* face_name(Face f) {
    switch (f) {
        case Face::Top:
            return "top";
        case Face::WallXm:
            return "wall-x";
        case Face::WallXp:
            return "wall+x";
        case Face::WallYm:
            return "wall-y";
        case Face::WallYp:
            return "wall+y";
        case Face::Bottom:
            return "bottom";
    }
    return "?";
}

/// Face of a triangle with unit normal n (see the file comment). The thresholds |n_z| = 0.5 put a
/// top-face triangle steeper than 60 degrees (local slope > tan 60 deg = 1.73) among the walls;
/// box_validity therefore checks that the number of top-classified triangles equals the
/// generator's top-face count 2 (n_x - 1)(n_y - 1) (sample_face's expected_count). The WP-V3
/// surface (sigma = 50 nm, Lc = 500 nm) has no such triangle.
inline Face classify(const Vec3& n) {
    if (n.z() < -0.5)
        return Face::Top;
    if (n.z() > 0.5)
        return Face::Bottom;
    if (std::abs(n.x()) >= std::abs(n.y()))
        return n.x() < 0.0 ? Face::WallXm : Face::WallXp;
    return n.y() < 0.0 ? Face::WallYm : Face::WallYp;
}

/// In-face coordinates (u, v) of a point of face f.
inline std::pair<Real, Real> face_coordinates(Face f, const Vec3& r) {
    switch (f) {
        case Face::WallXm:
        case Face::WallXp:
            return {r.y(), r.z()};
        case Face::WallYm:
        case Face::WallYp:
            return {r.x(), r.z()};
        case Face::Top:
        case Face::Bottom:
            break;
    }
    return {r.x(), r.y()};
}

/// Currents at the triangle centroids of one face (or a subset of it).
struct FaceSamples {
    std::vector<Vec3> points;  ///< centroids [m]
    std::vector<Real> u, v;    ///< in-face coordinates [m]
    std::vector<Real> area;    ///< triangle areas [m^2]
    std::vector<Vec3c> J, M;   ///< [A/m], [V/m]
    [[nodiscard]] std::size_t size() const { return points.size(); }
};

/// Samples of the solution on the triangles of face f. With expected_count >= 0 the number of
/// triangles classified as face f must equal it (the generator's count, see classify()).
/// @throws std::runtime_error if it does not.
inline FaceSamples sample_face(const post::SurfaceSolution& s, Face f, Index expected_count = -1) {
    const geometry::TriangleMesh& mesh = s.problem->space->mesh();
    FaceSamples out;
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        if (classify(mesh.normal(t)) != f)
            continue;
        const Vec3 c = mesh.centroid(t);
        const auto [J, M] = ag_absorption::currents_at(s, t, c);
        const auto [u, v] = face_coordinates(f, c);
        out.points.push_back(c);
        out.u.push_back(u);
        out.v.push_back(v);
        out.area.push_back(mesh.area(t));
        out.J.push_back(J);
        out.M.push_back(M);
    }
    if (expected_count >= 0 && static_cast<Index>(out.size()) != expected_count)
        throw std::runtime_error(std::string("sample_face: ") + std::to_string(out.size()) + " " +
                                 face_name(f) + " triangles by the normal, the generator has " +
                                 std::to_string(expected_count) +
                                 " (a triangle steeper than 60 degrees?)");
    return out;
}

/// Pairs (index in a, index in b) of samples whose points coincide within `tol` (per coordinate),
/// in the order of a. Points without a partner are skipped.
/// @throws std::invalid_argument for tol not finite and > 0.
inline std::vector<std::pair<std::size_t, std::size_t>> match_points(const std::vector<Vec3>& a,
                                                                     const std::vector<Vec3>& b,
                                                                     Real tol) {
    if (!(tol > 0.0) || !std::isfinite(tol))
        throw std::invalid_argument("match_points: tol must be finite and > 0");
    using Key = std::array<std::int64_t, 3>;
    const auto key = [tol](const Vec3& p) {
        return Key{static_cast<std::int64_t>(std::llround(p.x() / tol)),
                   static_cast<std::int64_t>(std::llround(p.y() / tol)),
                   static_cast<std::int64_t>(std::llround(p.z() / tol))};
    };
    std::map<Key, std::size_t> index;
    for (std::size_t i = 0; i < b.size(); ++i) index.emplace(key(b[i]), i);
    std::vector<std::pair<std::size_t, std::size_t>> out;
    for (std::size_t i = 0; i < a.size(); ++i) {
        // A point near a cell boundary may round to a neighbouring key: look at the 27 cells.
        const Key k = key(a[i]);
        bool found = false;
        for (std::int64_t dx = -1; dx <= 1 && !found; ++dx)
            for (std::int64_t dy = -1; dy <= 1 && !found; ++dy)
                for (std::int64_t dz = -1; dz <= 1 && !found; ++dz) {
                    const auto it = index.find(Key{k[0] + dx, k[1] + dy, k[2] + dz});
                    if (it != index.end() && (b[it->second] - a[i]).cwiseAbs().maxCoeff() <= tol) {
                        out.emplace_back(i, it->second);
                        found = true;
                    }
                }
    }
    return out;
}

/// Samples of a - b on the matched points (coordinates and areas of a).
inline FaceSamples difference(const FaceSamples& a, const FaceSamples& b,
                              const std::vector<std::pair<std::size_t, std::size_t>>& pairs) {
    FaceSamples d;
    for (const auto& [i, j] : pairs) {
        d.points.push_back(a.points[i]);
        d.u.push_back(a.u[i]);
        d.v.push_back(a.v[i]);
        d.area.push_back(a.area[i]);
        d.J.push_back(a.J[i] - b.J[j]);
        d.M.push_back(a.M[i] - b.M[j]);
    }
    return d;
}

/// Subset of the samples on the matched points of a (the common region of a pair).
inline FaceSamples subset(const FaceSamples& a,
                          const std::vector<std::pair<std::size_t, std::size_t>>& pairs) {
    FaceSamples d;
    for (const auto& [i, j] : pairs) {
        (void)j;
        d.points.push_back(a.points[i]);
        d.u.push_back(a.u[i]);
        d.v.push_back(a.v[i]);
        d.area.push_back(a.area[i]);
        d.J.push_back(a.J[i]);
        d.M.push_back(a.M[i]);
    }
    return d;
}

/// RWG coefficients of the basis functions of one face (both triangles on face f) of the solution
/// `x_from` on `from`, placed at the geometrically identical basis functions of `to` (matched by
/// edge midpoint within tol; the sign follows the plus / minus orientation: +1 if the plus
/// triangles of the two functions have the same centroid within tol, -1 if the plus and minus
/// triangles are exchanged (the function of `to` points the other way)); all other entries are
/// zero. Returns the [J; M] vector of length 2 N_to and the number of matched basis functions.
/// Used to radiate the current difference of one face alone (e.g. the top face of the depth 4
/// and 2 um boxes).
/// @throws std::invalid_argument for x_from of the wrong size or tol not finite and > 0.
/// @throws std::runtime_error for two functions on the same edge whose triangles differ (neither
///         plus = plus nor plus = minus and minus = plus): the meshes differ there.
inline std::pair<VectorXc, std::size_t> transfer_face_currents(const basis::RwgSpace& from,
                                                               const VectorXc& x_from,
                                                               const basis::RwgSpace& to, Face f,
                                                               Real tol) {
    if (x_from.size() != 2 * from.size())
        throw std::invalid_argument("transfer_face_currents: x_from must have 2 N entries");
    const auto collect = [f](const basis::RwgSpace& s, std::vector<Index>& idx,
                             std::vector<Vec3>& mid, std::vector<Vec3>& plus,
                             std::vector<Vec3>& minus) {
        const geometry::TriangleMesh& mesh = s.mesh();
        for (Index n = 0; n < s.size(); ++n) {
            if (classify(mesh.normal(s.plus_triangle(n))) != f ||
                classify(mesh.normal(s.minus_triangle(n))) != f)
                continue;
            idx.push_back(n);
            mid.push_back(0.5 * (mesh.vertices().row(mesh.edges()(n, 0)).transpose() +
                                 mesh.vertices().row(mesh.edges()(n, 1)).transpose()));
            plus.push_back(mesh.centroid(s.plus_triangle(n)));
            minus.push_back(mesh.centroid(s.minus_triangle(n)));
        }
    };
    std::vector<Index> ia, ib;
    std::vector<Vec3> ma, mb, pa, pb, qa, qb;
    collect(from, ia, ma, pa, qa);
    collect(to, ib, mb, pb, qb);
    const auto pairs = match_points(ma, mb, tol);
    const Index nf = from.size(), nt = to.size();
    VectorXc out = VectorXc::Zero(2 * nt);
    const auto same = [tol](const Vec3& p, const Vec3& q) {
        return (p - q).cwiseAbs().maxCoeff() <= tol;
    };
    for (const auto& [i, j] : pairs) {
        Real sign = 1.0;
        if (!same(pa[i], pb[j]) || !same(qa[i], qb[j])) {
            if (!same(pa[i], qb[j]) || !same(qa[i], pb[j]))
                throw std::runtime_error(
                    "transfer_face_currents: basis functions on the same edge have different "
                    "triangles (the meshes differ there)");
            sign = -1.0;
        }
        out(ib[j]) = sign * x_from(ia[i]);
        out(nt + ib[j]) = sign * x_from(nf + ia[i]);
    }
    return {out, pairs.size()};
}

/// Radially binned 2D spectrum of J and M on a face (see the file comment).
struct RadialSpectrum {
    Real k0 = 0;                    ///< [rad/m]
    Real bin_width = 0;             ///< in units of k0
    Real k_max = 0;                 ///< grid half width in units of k0
    std::vector<Real> bin_centre;   ///< |k| / k0
    std::vector<Real> j, m;         ///< power fraction per bin (sums to 1 over the bins)
    Real j_above = 0, m_above = 0;  ///< fraction with |k| > k0
    Real j_total = 0, m_total = 0;  ///< sum of |X|^2 over the grid (raw)
    Real rms_j = 0, rms_m = 0;      ///< sqrt(sum A |X|^2 / sum A) of the samples
};

/// The spectrum of the samples with grid spacing dk = dk_factor k0 up to |k_u|, |k_v| <= k_max =
/// k_max_factor k0, bins of bin_width k0 from |k| = 0 to sqrt(2) k_max.
/// @throws std::invalid_argument for fewer than 4 samples, a degenerate extent, k0 not finite and
///         > 0, or factors not finite and > 0.
inline RadialSpectrum radial_spectrum(const FaceSamples& s, Real k0, Real k_max_factor,
                                      Real dk_factor, Real bin_width, bool hann) {
    if (s.size() < 4)
        throw std::invalid_argument("radial_spectrum: need at least 4 samples");
    if (!(k0 > 0.0) || !std::isfinite(k0) || !(k_max_factor > 0.0) || !(dk_factor > 0.0) ||
        !(bin_width > 0.0) || !std::isfinite(k_max_factor) || !std::isfinite(dk_factor) ||
        !std::isfinite(bin_width))
        throw std::invalid_argument("radial_spectrum: k0 and the factors must be finite and > 0");
    const auto [umin, umax] = std::minmax_element(s.u.begin(), s.u.end());
    const auto [vmin, vmax] = std::minmax_element(s.v.begin(), s.v.end());
    const Real u0 = *umin, u1 = *umax, v0 = *vmin, v1 = *vmax;
    if (!(u1 > u0) || !(v1 > v0))
        throw std::invalid_argument("radial_spectrum: degenerate face extent");
    const auto half = static_cast<Index>(std::ceil(k_max_factor / dk_factor - 1e-9));
    const Index nk = 2 * half + 1;
    const Real dk = dk_factor * k0;
    const auto ns = static_cast<Index>(s.size());
    MatrixXc Eu(nk, ns), Ev(nk, ns);
    MatrixXc A(ns, 6);
    Real area_sum = 0, sj = 0, sm = 0;
    for (Index i = 0; i < ns; ++i) {
        const auto q = static_cast<std::size_t>(i);
        const Real w = hann ? std::pow(std::sin(constants::pi * (s.u[q] - u0) / (u1 - u0)), 2) *
                                  std::pow(std::sin(constants::pi * (s.v[q] - v0) / (v1 - v0)), 2)
                            : 1.0;
        for (Index a = 0; a < nk; ++a) {
            const Real k = dk * static_cast<Real>(a - half);
            Eu(a, i) = std::polar(1.0, k * s.u[q]);
            Ev(a, i) = std::polar(1.0, k * s.v[q]);
        }
        for (int c = 0; c < 3; ++c) {
            A(i, c) = w * s.area[q] * s.J[q](c);
            A(i, 3 + c) = w * s.area[q] * s.M[q](c);
        }
        area_sum += s.area[q];
        sj += s.area[q] * s.J[q].squaredNorm();
        sm += s.area[q] * s.M[q].squaredNorm();
    }
    MatrixXr pj = MatrixXr::Zero(nk, nk), pm = MatrixXr::Zero(nk, nk);
    for (int c = 0; c < 6; ++c) {
        // X(a, b) = sum_i Eu(a, i) A(i, c) Ev(b, i)
        const MatrixXc X = Eu * (A.col(c).asDiagonal() * Ev.transpose());
        (c < 3 ? pj : pm) += X.cwiseAbs2();
    }
    RadialSpectrum out;
    out.k0 = k0;
    out.bin_width = bin_width;
    out.k_max = dk_factor * static_cast<Real>(half);
    const auto n_bins = static_cast<std::size_t>(std::ceil(std::sqrt(2.0) * out.k_max / bin_width));
    out.bin_centre.resize(n_bins);
    out.j.assign(n_bins, 0.0);
    out.m.assign(n_bins, 0.0);
    for (std::size_t b = 0; b < n_bins; ++b)
        out.bin_centre[b] = (static_cast<Real>(b) + 0.5) * bin_width;
    Real tj = 0, tm = 0, aj = 0, am = 0;
    for (Index a = 0; a < nk; ++a)
        for (Index b = 0; b < nk; ++b) {
            const Real kr =
                dk_factor * std::hypot(static_cast<Real>(a - half), static_cast<Real>(b - half));
            const auto bin = std::min(n_bins - 1, static_cast<std::size_t>(kr / bin_width));
            out.j[bin] += pj(a, b);
            out.m[bin] += pm(a, b);
            tj += pj(a, b);
            tm += pm(a, b);
            if (kr > 1.0) {
                aj += pj(a, b);
                am += pm(a, b);
            }
        }
    out.j_total = tj;
    out.m_total = tm;
    if (tj > 0.0) {
        for (Real& x : out.j) x /= tj;
        out.j_above = aj / tj;
    }
    if (tm > 0.0) {
        for (Real& x : out.m) x /= tm;
        out.m_above = am / tm;
    }
    out.rms_j = std::sqrt(sj / area_sum);
    out.rms_m = std::sqrt(sm / area_sum);
    return out;
}

/// Bin of largest power and power-weighted mean |k| / k0 of a binned spectrum within [lo, hi]
/// (bin centres, units of k0), and the power fraction in that band.
struct BandPeak {
    Real peak = 0, centroid = 0, fraction = 0;
};

/// @throws std::invalid_argument if no bin centre lies in [lo, hi] or the sizes differ.
inline BandPeak band_peak(const std::vector<Real>& centre, const std::vector<Real>& power, Real lo,
                          Real hi) {
    if (centre.size() != power.size())
        throw std::invalid_argument("band_peak: sizes differ");
    BandPeak p;
    Real best = -1.0, sum = 0.0, wsum = 0.0;
    bool any = false;
    for (std::size_t b = 0; b < centre.size(); ++b) {
        if (centre[b] < lo || centre[b] > hi)
            continue;
        any = true;
        sum += power[b];
        wsum += power[b] * centre[b];
        if (power[b] > best) {
            best = power[b];
            p.peak = centre[b];
        }
    }
    if (!any)
        throw std::invalid_argument("band_peak: empty band");
    p.fraction = sum;
    p.centroid = sum > 0.0 ? wsum / sum : 0.5 * (lo + hi);
    return p;
}

/// Re k_spp / k0 = Re sqrt(eps / (1 + eps)) of a flat interface vacuum / eps (bound SPP for
/// Re eps < -1; exp(+jwt) branch with Im k_spp <= 0). Returns the complex ratio.
/// @throws std::invalid_argument for eps = -1 or a non-finite eps.
inline Complex spp_ratio(Complex eps) {
    if (!std::isfinite(eps.real()) || !std::isfinite(eps.imag()) ||
        std::abs(eps + Complex(1.0, 0.0)) == 0.0)
        throw std::invalid_argument("spp_ratio: eps must be finite and != -1");
    Complex r = std::sqrt(eps / (Complex(1.0, 0.0) + eps));
    if (r.real() < 0.0)
        r = -r;
    return r;
}

// ---------------------------------------------------------------------------------------------
// Interior probe lines (extinction-theorem residual, WP-A2 placement rule).

struct ProbeLine {
    std::string name;  ///< "x" / "y" (lateral) or "z" (vertical through the centre)
    Real level = 0;    ///< z of a lateral line, x = y = 0 for the vertical one [m]
    Vertices points;
    std::vector<Real> coordinate;  ///< x, y or z of each point [m]
};

/// Lines inside the box of a patch of edge L (centred at the origin), box depth `depth` below the
/// mean plane, highest surface point max_height (max xi, towards +z = into the object) and mesh
/// size h: margin m = 3 h; z_p = max_height + m; lateral lines along x (y = 0) at z_p, at the
/// levels z_p + extra (each extra > 0, kept if <= depth - m) and at depth - m (the bottom line),
/// one along y (x = 0) at z_p, and the vertical line x = y = 0 from z_p to depth - m; spacing
/// `step`, laterally |x|, |y| <= L / 2 - m.
/// @throws std::invalid_argument for non-finite or non-positive L, depth, h, step, or a box too
///         shallow for the margins (z_p >= depth - m) or too narrow (L / 2 - m <= 0).
inline std::vector<ProbeLine> interior_probe_lines(Real L, Real depth, Real max_height, Real h,
                                                   Real step, const std::vector<Real>& extra) {
    if (!(L > 0.0) || !(depth > 0.0) || !(h > 0.0) || !(step > 0.0) || !std::isfinite(L) ||
        !std::isfinite(depth) || !std::isfinite(h) || !std::isfinite(step) ||
        !std::isfinite(max_height))
        throw std::invalid_argument("interior_probe_lines: need finite L, depth, h, step > 0");
    const Real m = 3.0 * h;
    const Real zp = max_height + m;
    const Real zb = depth - m;
    const Real x1 = 0.5 * L - m;
    if (!(zp < zb) || !(x1 > 0.0))
        throw std::invalid_argument("interior_probe_lines: box too small for the margins");
    const auto n_lat = static_cast<Index>(std::floor(2.0 * x1 / step + 1e-9)) + 1;
    const Real u0 = -0.5 * static_cast<Real>(n_lat - 1) * step;
    const auto lateral = [&](const char* name, Real z, bool along_x) {
        ProbeLine l;
        l.name = name;
        l.level = z;
        l.points.resize(n_lat, 3);
        for (Index i = 0; i < n_lat; ++i) {
            const Real u = u0 + static_cast<Real>(i) * step;
            l.points.row(i) = along_x ? Vec3(u, 0.0, z).transpose() : Vec3(0.0, u, z).transpose();
            l.coordinate.push_back(u);
        }
        return l;
    };
    std::vector<ProbeLine> out;
    out.push_back(lateral("x", zp, true));
    for (const Real e : extra) {
        if (!(e > 0.0))
            throw std::invalid_argument("interior_probe_lines: extra levels must be > 0");
        if (zp + e < zb)
            out.push_back(lateral("x", zp + e, true));
    }
    out.push_back(lateral("x", zb, true));
    out.push_back(lateral("y", zp, false));
    const auto n_z = static_cast<Index>(std::floor((zb - zp) / step + 1e-9)) + 1;
    ProbeLine vz;
    vz.name = "z";
    vz.level = 0.0;
    vz.points.resize(n_z, 3);
    for (Index i = 0; i < n_z; ++i) {
        const Real z = zp + static_cast<Real>(i) * step;
        vz.points.row(i) = Vec3(0.0, 0.0, z).transpose();
        vz.coordinate.push_back(z);
    }
    out.push_back(vz);
    return out;
}

/// rms over the rows of a field matrix: sqrt(sum |E_i|^2 / n).
inline Real rms(const Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E) {
    if (E.rows() == 0)
        return 0.0;
    return std::sqrt(E.squaredNorm() / static_cast<Real>(E.rows()));
}

/// Hann-tapered DFT |sum_i w_i E_i exp(+j k u_i)| along a line, evaluated on `ks`; returns the k
/// of the largest value and that value relative to the untapered sum of |E_i| w_i (a pure
/// exp(-j k0 u) field gives 1 at k0).
/// @throws std::invalid_argument for fewer than 2 points, sizes that differ or an empty k list.
inline std::pair<Real, Real> line_peak(const Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E,
                                       const std::vector<Real>& u, const std::vector<Real>& ks) {
    if (u.size() < 2 || static_cast<Index>(u.size()) != E.rows() || ks.empty())
        throw std::invalid_argument("line_peak: need >= 2 points, matching sizes, some k");
    const Real a = u.front(), b = u.back();
    std::vector<Real> w(u.size());
    Real norm = 0;
    for (std::size_t i = 0; i < u.size(); ++i) {
        w[i] = std::pow(std::sin(constants::pi * (u[i] - a) / (b - a)), 2);
        norm += w[i] * E.row(static_cast<Index>(i)).norm();
    }
    Real best_k = ks.front(), best = -1.0;
    for (const Real k : ks) {
        Vec3c acc = Vec3c::Zero();
        for (std::size_t i = 0; i < u.size(); ++i)
            acc += w[i] * std::polar(1.0, k * u[i]) * E.row(static_cast<Index>(i)).transpose();
        if (acc.norm() > best) {
            best = acc.norm();
            best_k = k;
        }
    }
    return {best_k, norm > 0.0 ? best / norm : 0.0};
}

// ---------------------------------------------------------------------------------------------
// GMRES Hessenberg and depth-wave fit.

/// Leading (j + 1) x j block of a (k + 1) x k Hessenberg.
/// @throws std::invalid_argument for j < 1, j > k or a matrix that is not (k + 1) x k.
inline MatrixXc leading_hessenberg(const MatrixXc& hbar, Index j) {
    const Index k = hbar.cols();
    if (k < 1 || hbar.rows() != k + 1 || j < 1 || j > k)
        throw std::invalid_argument(
            "leading_hessenberg: need a (k + 1) x k matrix and 1 <= j <= k");
    return hbar.topLeftCorner(j + 1, j);
}

/// Residual of the depth-wave model for one beta (see the file comment): rows of `F` are the runs
/// (depths D_i), columns the (direction, component) samples. Returns sum |(I - P) F|^2 /
/// sum |F - mean F|^2, P the projection onto span{1, exp(-j beta D)}.
/// @throws std::invalid_argument for fewer than 3 runs, sizes that differ, or runs that do not
///         differ (zero depth-dependent part).
inline Real depth_wave_residual(const MatrixXc& F, const std::vector<Real>& depth, Real beta) {
    const Index n = F.rows();
    if (n < 3 || static_cast<Index>(depth.size()) != n)
        throw std::invalid_argument("depth_wave_residual: need >= 3 runs and one depth per run");
    MatrixXc B(n, 2);
    for (Index i = 0; i < n; ++i) {
        B(i, 0) = 1.0;
        B(i, 1) = std::polar(1.0, -beta * depth[static_cast<std::size_t>(i)]);
    }
    // Orthonormal basis of span(B) (Householder QR; a beta with exp(-j beta D) constant over the
    // depths leaves a rank-1 B, whose projection is that of the constant column).
    const Eigen::HouseholderQR<MatrixXc> qr(B);
    const MatrixXc Q = qr.householderQ() * MatrixXc::Identity(n, 2);
    const Real r11 = std::abs(qr.matrixQR()(1, 1));
    const MatrixXc Qe = r11 > 1e-10 * std::abs(qr.matrixQR()(0, 0)) ? Q : MatrixXc(Q.leftCols(1));
    const MatrixXc R = F - Qe * (Qe.adjoint() * F);
    const MatrixXc C = F.rowwise() - F.colwise().mean();
    const Real den = C.squaredNorm();
    if (!(den > 0.0))
        throw std::invalid_argument("depth_wave_residual: the runs do not differ");
    return R.squaredNorm() / den;
}

/// depth_wave_residual on a grid of beta; returns (beta of the smallest residual, residual).
/// @throws std::invalid_argument for an empty grid and what depth_wave_residual rejects.
inline std::pair<Real, Real> fit_depth_wave(const MatrixXc& F, const std::vector<Real>& depth,
                                            const std::vector<Real>& betas) {
    if (betas.empty())
        throw std::invalid_argument("fit_depth_wave: empty beta grid");
    Real best_b = betas.front(), best_r = 2.0;
    for (const Real b : betas) {
        const Real r = depth_wave_residual(F, depth, b);
        if (r < best_r) {
            best_r = r;
            best_b = b;
        }
    }
    return {best_b, best_r};
}

// ---------------------------------------------------------------------------------------------
// Stored run directories (io::open_npy_directory): arrays, attributes and configuration checks.

/// Minimal reader for the C-order .npy files written by io::open_npy_directory: dtype '<f8'
/// (Scalar = Real) or '<c16' (Scalar = Complex), shape (rows,) or (rows, cols); a vector is
/// returned as a rows x 1 matrix.
/// @throws std::runtime_error if the file cannot be opened, is not an .npy file of that dtype in C
///         order, or is truncated.
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> read_npy(const std::filesystem::path& path) {
    static_assert(std::is_same_v<Scalar, Real> || std::is_same_v<Scalar, Complex>);
    constexpr bool is_complex = std::is_same_v<Scalar, Complex>;
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + core::path_to_utf8(path));
    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("not an .npy file: " + core::path_to_utf8(path));
    std::size_t header_len = 0;
    if (magic[6] == 1) {
        unsigned char b[2];
        f.read(reinterpret_cast<char*>(b), 2);
        header_len = static_cast<std::size_t>(b[0]) | (static_cast<std::size_t>(b[1]) << 8);
    } else {
        unsigned char b[4];
        f.read(reinterpret_cast<char*>(b), 4);
        header_len = static_cast<std::size_t>(b[0]) | (static_cast<std::size_t>(b[1]) << 8) |
                     (static_cast<std::size_t>(b[2]) << 16) |
                     (static_cast<std::size_t>(b[3]) << 24);
    }
    if (!f)
        throw std::runtime_error("truncated header in " + core::path_to_utf8(path));
    std::string header(header_len, ' ');
    f.read(header.data(), static_cast<std::streamsize>(header_len));
    if (!f || header.find(is_complex ? "'<c16'" : "'<f8'") == std::string::npos ||
        header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error("unsupported .npy layout in " + core::path_to_utf8(path));
    const std::size_t s0 = header.find("'shape': (");
    const std::size_t s1 = s0 == std::string::npos ? s0 : header.find(')', s0);
    if (s1 == std::string::npos)
        throw std::runtime_error("no shape in " + core::path_to_utf8(path));
    const std::string shape = header.substr(s0 + 10, s1 - s0 - 10);
    const std::size_t comma = shape.find(',');
    const long long rows = std::stoll(shape.substr(0, comma));
    long long cols = 1;
    if (comma != std::string::npos && shape.find_first_of("0123456789", comma) != std::string::npos)
        cols = std::stoll(shape.substr(comma + 1));
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m(rows, cols);
    f.read(reinterpret_cast<char*>(m.data()),
           static_cast<std::streamsize>(sizeof(Scalar) * static_cast<std::size_t>(rows * cols)));
    if (!f)
        throw std::runtime_error("truncated " + core::path_to_utf8(path));
    return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(m);
}

/// Attributes of a stored run, (key, value) in file order.
using Attributes = std::vector<std::pair<std::string, std::string>>;

/// Minimal reader for the flat attributes.json written by io::open_npy_directory (one
/// `"key": value` pair per line, string or number values). String values are returned without
/// the quotes (the study writes no escaped characters).
/// @throws std::runtime_error if the file cannot be opened or a line is malformed.
inline Attributes read_attributes(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + core::path_to_utf8(path));
    Attributes attrs;
    std::string line;
    while (std::getline(f, line)) {
        const std::size_t k0 = line.find('"');
        if (k0 == std::string::npos)
            continue;
        const std::size_t k1 = line.find('"', k0 + 1);
        const std::size_t colon = line.find(':', k1 == std::string::npos ? k0 : k1);
        if (k1 == std::string::npos || colon == std::string::npos)
            throw std::runtime_error("malformed line in " + core::path_to_utf8(path));
        std::string value = line.substr(colon + 1);
        while (!value.empty() &&
               (value.back() == ',' || value.back() == '\r' || value.back() == ' '))
            value.pop_back();
        const std::size_t v0 = value.find_first_not_of(' ');
        value = v0 == std::string::npos ? std::string() : value.substr(v0);
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
            value = value.substr(1, value.size() - 2);
        attrs.emplace_back(line.substr(k0 + 1, k1 - k0 - 1), value);
    }
    return attrs;
}

/// Value of `key`, if present.
inline std::optional<std::string> find_attribute(const Attributes& a, const std::string& key) {
    for (const auto& [k, v] : a)
        if (k == key)
            return v;
    return std::nullopt;
}

/// Whether two attribute values agree: numerically within a relative 1e-12 if both parse
/// completely as numbers (so "1.0" equals "1" and "4e-06" equals "4e-6"), else as strings.
inline bool same_attribute_value(const std::string& a, const std::string& b) {
    const auto number = [](const std::string& s) -> std::optional<Real> {
        if (s.empty())
            return std::nullopt;
        char* end = nullptr;
        const Real x = std::strtod(s.c_str(), &end);
        if (end != s.c_str() + s.size() || !std::isfinite(x))
            return std::nullopt;
        return x;
    };
    const std::optional<Real> x = number(a), y = number(b);
    if (x && y)
        return std::abs(*x - *y) <= 1e-12 * std::max(std::abs(*x), std::abs(*y));
    return a == b;
}

/// The keys of `expected` whose stored value is missing or differs (same_attribute_value), one
/// message "key: stored S, expected E" each.
inline std::vector<std::string> attribute_mismatches(const Attributes& stored,
                                                     const Attributes& expected) {
    std::vector<std::string> out;
    for (const auto& [key, value] : expected) {
        const std::optional<std::string> s = find_attribute(stored, key);
        if (!s)
            out.push_back(key + ": missing, expected " + value);
        else if (!same_attribute_value(*s, value))
            out.push_back(key + ": stored " + *s + ", expected " + value);
    }
    return out;
}

/// @throws std::runtime_error naming `context` and every mismatch of attribute_mismatches.
inline void require_attributes(const Attributes& stored, const Attributes& expected,
                               const std::string& context) {
    const std::vector<std::string> m = attribute_mismatches(stored, expected);
    if (m.empty())
        return;
    std::string msg = context + ": the stored run does not match the configuration (";
    for (std::size_t i = 0; i < m.size(); ++i) msg += (i ? "; " : "") + m[i];
    throw std::runtime_error(msg + ")");
}

/// Keys of a stored box_validity run that define its configuration: material, beam, surface, box,
/// mesh, solver and far-field settings; not the outcome (tag, unknowns, triangles, iterations,
/// residuals, timings).
inline const std::vector<std::string>& configuration_keys() {
    static const std::vector<std::string> keys = {"material",
                                                  "beam",
                                                  "box_mesh_size",
                                                  "fine_band",
                                                  "wavelength",
                                                  "edge_length_L",
                                                  "map_L",
                                                  "seed",
                                                  "rms_roughness",
                                                  "waist_radius",
                                                  "box_depth",
                                                  "top_triangles",
                                                  "eps_r_real",
                                                  "eps_r_imag",
                                                  "diagnostic",
                                                  "gmres_tolerance",
                                                  "compression",
                                                  "accuracy_digits",
                                                  "jacobi",
                                                  "leaf_radius_quantile",
                                                  "ff_degree",
                                                  "hemisphere_dtheta_deg",
                                                  "hemisphere_dphi_deg",
                                                  "cut_step_deg"};
    return keys;
}

/// Requires that all runs have the same configuration (configuration_keys(), same_attribute_value)
/// apart from the keys in `ignore`; a key absent from every run is not compared, a key absent from
/// some runs is a mismatch. `names` label the runs in the message.
/// @throws std::invalid_argument for sizes that differ or fewer than 2 runs.
/// @throws std::runtime_error naming the first run that differs from runs[0] and its keys.
inline void require_same_configuration(const std::vector<Attributes>& runs,
                                       const std::vector<std::string>& names,
                                       const std::vector<std::string>& ignore) {
    if (runs.size() != names.size() || runs.size() < 2)
        throw std::invalid_argument("require_same_configuration: need >= 2 runs, one name each");
    for (const std::string& key : configuration_keys()) {
        if (std::find(ignore.begin(), ignore.end(), key) != ignore.end())
            continue;
        bool any = false;
        for (const Attributes& r : runs) any = any || find_attribute(r, key).has_value();
        if (!any)
            continue;
        const std::optional<std::string> ref = find_attribute(runs[0], key);
        for (std::size_t i = 1; i < runs.size(); ++i) {
            const std::optional<std::string> v = find_attribute(runs[i], key);
            if (!ref || !v || !same_attribute_value(*ref, *v))
                throw std::runtime_error("runs " + names[0] + " and " + names[i] + " differ in " +
                                         key + " (" + ref.value_or("missing") + " vs " +
                                         v.value_or("missing") + ")");
        }
    }
}

}  // namespace ag_box_mechanism
