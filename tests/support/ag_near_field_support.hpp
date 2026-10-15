#pragma once
/// @file ag_near_field_support.hpp
/// Near field of the Ag sphere in the xz-plane against Mie (WP24, docs/05 row "Near field Ag
/// sphere, xz-plane": relative error per point < 0.1 on the lit side, < 1 in the shadow; docs/01
/// Phase 5 DoD). Shared by tests/validation_large/test_ag_near_field_large.cpp and the study
/// executable benchmarks/ag_near_field.cpp, so it must not depend on Catch2. The case (mesh,
/// material, MLFMM, GMRES) is the WP22a setup of ag_sphere_mlfmm_support.hpp; this header adds the
/// observation points, the evaluation of the computed total field and the error statistics.
///
/// Metric (per observation point r in the plane y = 0):
///   e(r) = |E_sim(r) - E_Mie(r)| / |E_Mie(r)|      (complex Euclidean norms of the 3-vectors),
/// with E_sim = post::total_field (incident + scattered in R1, the R2 representation inside) and
/// E_Mie = x_hat exp(-j k1 z) + MieSolution::scattered_E outside the sphere, internal_E inside.
/// Also recorded: the magnitude error ||E_sim| - |E_Mie|| / |E_Mie| and the absolute error
/// |E_sim - E_Mie| in units of |E_inc| = 1 V/m.
///
/// Zones (the distance from the surface is d = |r| - a, a the sphere radius). The faceted mesh
/// deviates from the sphere by its sagitta: h^2 / 8a per edge = 0.06 nm on the 4 um paper mesh,
/// 0.09 nm on the 1 um icosphere n = 5, 0.36 nm on n = 4 and ~1.4 nm on the CI mesh n = 3
/// (~1.9 nm by the facet-centre depth h^2 / 6a), all below the 5 nm = lambda / 100 evaluation
/// cut-off and below d_min.
///  * band:     |d| < d_min (near-surface exclusion, see below); not part of any statistic.
///  * shadow:   the geometric shadow: d >= d_min, z > 0 and x^2 + y^2 < a^2 (in the plane y = 0:
///              |x| < a). Behind the sphere the forward-scattered wave cancels the incident one,
///              so |E| can be << 1 and the relative error has a small denominator. Criterion
///              e < 1.
///  * lit:      every other point with d >= d_min ("lit / outside"): the illuminated side z <= 0,
///              the x-axis z = 0 through the centre ("along x" of the DoD) and the forward region
///              beside the sphere (z > 0, |x| >= a). There |E| = O(1), and the docs/05 criterion
///              is e < 0.1. The x-axis and the forward region beside the sphere are reported
///              separately as subsets.
///  * interior: d <= -d_min: inside the Ag sphere the field amplitude decays to 1/e within
///              lambda / (2 pi |Im n|) = 25.4 nm (the intensity penetration depth lambda /
///              (4 pi |Im n|) is 12.7 nm), so |E_Mie| is tiny away from the surface and a
///              relative error is not meaningful; reported as the absolute error |E_sim - E_Mie|
///              relative to max |E_inc| = 1 V/m (diagnostic, no acceptance).
/// Near-surface exclusion: the field evaluation integrates the RWG currents with a fixed Dunavant
/// rule per triangle (FieldOptions::quad_degree), which is inaccurate for observation points at
/// distances comparable to or below the triangle size h (nearly singular integrands), and the
/// faceted RWG surface itself deviates from the smooth sphere on the scale h. The exclusion
/// distance d_min is chosen from the measured error profile on circles at distances d = lambda /
/// 100 ... lambda around the sphere (benchmarks/results/ag_near_field.md), see kDMinPerEdge.
#include "specklebem/core/types.hpp"
#include "specklebem/postprocessing/fields.hpp"
#include "specklebem/reference/mie.hpp"
#include "specklebem/simulation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ag_sphere_mlfmm_support.hpp"

namespace ag_near_field {

using namespace specklebem;
using ag_sphere_mlfmm::kLambda;
using FieldMatrix = Eigen::Matrix<Complex, Eigen::Dynamic, 3>;

/// Near-surface exclusion distance d_min in units of the mean edge length h of the mesh, from the
/// WP24 profile measurement (benchmarks/results/ag_near_field.md, d = 1 um at h = lambda / 13.2
/// and lambda / 26.5; d = 4 um at lambda / 16.5): for d >= h / 2 (i) the default Dunavant degree 6
/// agrees with degree 19 to <= 3 % of the error e (the nearly singular quadrature error is
/// negligible), and (ii) the maximum of e over a circle at distance d is within ~1.5 x its value at
/// d = h; below h / 2 both the quadrature error (degree 6) and the discretisation error (flat
/// facets and piecewise linear RWG currents deviate from the smooth sphere on the scale h; their
/// field decays roughly like exp(-2 pi d / h)) grow steeply. d_min = h / 2 is lambda / 33 on the
/// paper's 4 um mesh.
inline constexpr Real kDMinPerEdge = 0.5;

enum class Zone : int { lit = 0, shadow = 1, interior = 2, band = 3 };

inline const char* zone_name(Zone z) {
    switch (z) {
        case Zone::lit:
            return "lit";
        case Zone::shadow:
            return "shadow";
        case Zone::interior:
            return "interior";
        case Zone::band:
            return "band";
    }
    return "?";
}

/// Signed distance |r| - a from the sphere surface (> 0 outside).
inline Real surface_distance(const Vec3& r, Real radius) {
    return r.norm() - radius;
}

/// Tolerance |z| < kAxisTolerance a of the points counted as lying on the x-axis z = 0: absorbs
/// the rounding of e.g. (a + d) cos(pi / 2) ~ 1e-22 m of the shell points at psi = 90 deg.
inline constexpr Real kAxisTolerance = 1e-9;

/// True if r lies in the geometric shadow cylinder of the sphere: z > 0 and x^2 + y^2 < a^2.
inline bool in_geometric_shadow(const Vec3& r, Real radius) {
    return r.z() > 0.0 && std::hypot(r.x(), r.y()) < radius;
}

/// Zone of r for the exclusion distance d_min >= 0 (see the file comment). Exterior points on or
/// near the x-axis are always lit (|x| > a there), whatever the sign of the rounding of z.
inline Zone classify(const Vec3& r, Real radius, Real d_min) {
    const Real d = surface_distance(r, radius);
    if (std::abs(d) < d_min)
        return Zone::band;
    if (d < 0.0)
        return Zone::interior;
    return in_geometric_shadow(r, radius) ? Zone::shadow : Zone::lit;
}

/// Mie total field at r (V/m, |E_inc| = 1 V/m): incident + scattered for |r| >= a, internal inside.
inline Vec3c mie_total_E(const reference::MieSolution& mie, const Vec3& r, Real radius) {
    if (r.norm() < radius)
        return mie.internal_E(r);
    const Real k1 = 2.0 * constants::pi / kLambda;  // vacuum background
    const Complex phase = std::exp(Complex(0.0, -k1 * r.z()));
    return Vec3c(phase, 0.0, 0.0) + mie.scattered_E(r);
}

/// Quadrature of the field evaluation: Dunavant degree quad_degree for points with
/// |d| >= near_band, near_quad_degree for points closer to the surface (both positive-interior
/// degrees, post::FieldOptions). min_distance_factor stays at its default.
struct EvalOptions {
    int quad_degree = 6;
    int near_quad_degree = 6;
    Real near_band = 0.0;  ///< [m]
};

/// Computed and reference fields at a set of points.
struct Evaluation {
    FieldMatrix e_sim;            ///< post::total_field
    FieldMatrix e_mie;            ///< mie_total_E
    VectorXr rel;                 ///< |E_sim - E_Mie| / |E_Mie|
    VectorXr abs;                 ///< |E_sim - E_Mie| [V/m] (|E_inc| = 1 V/m)
    VectorXr mag_rel;             ///< ||E_sim| - |E_Mie|| / |E_Mie|
    Index region_mismatches = 0;  ///< points where the mesh region differs from |r| < a
    Real field_seconds = 0;       ///< post::total_field (incl. its region test)
    Real mie_seconds = 0;
};

/// Total field of the solution at the rows of `pts` (see EvalOptions) against Mie.
/// @throws std::invalid_argument if the mesh region (winding number) of a point disagrees with
///         the sphere (|r| < a), and what post::total_field throws for.
inline Evaluation evaluate(const post::SurfaceSolution& s, const reference::MieSolution& mie,
                           const Vertices& pts, Real radius, const EvalOptions& o) {
    using Clock = std::chrono::steady_clock;
    const Index np = pts.rows();
    Evaluation ev;
    ev.e_sim.resize(np, 3);
    std::vector<Index> idx[2];  // far, near
    for (Index p = 0; p < np; ++p) {
        const Real d = surface_distance(pts.row(p).transpose(), radius);
        idx[std::abs(d) < o.near_band ? 1 : 0].push_back(p);
    }
    const auto t0 = Clock::now();
    for (int set = 0; set < 2; ++set) {
        const std::vector<Index>& id = idx[set];
        if (id.empty())
            continue;
        Vertices sub(static_cast<Index>(id.size()), 3);
        for (std::size_t i = 0; i < id.size(); ++i) sub.row(static_cast<Index>(i)) = pts.row(id[i]);
        post::FieldOptions fo;
        fo.quad_degree = set == 0 ? o.quad_degree : o.near_quad_degree;
        FieldMatrix E;
        FieldMatrix H;
        post::total_field(s, sub, E, H, fo);
        // Consistency check of the zones against the mesh: this repeats the winding-number region
        // test that post::total_field performs internally (about 1/13 of the field cost).
        const Eigen::Matrix<int, Eigen::Dynamic, 1> region =
            post::observation_regions(s.problem->space->mesh(), sub);
        for (std::size_t i = 0; i < id.size(); ++i) {
            const auto row = static_cast<Index>(i);
            ev.e_sim.row(id[i]) = E.row(row);
            const bool inside = pts.row(id[i]).norm() < radius;
            if ((region(row) == 2) != inside)
                ++ev.region_mismatches;
        }
    }
    ev.field_seconds = std::chrono::duration<Real>(Clock::now() - t0).count();
    if (ev.region_mismatches > 0) {
        throw std::invalid_argument(
            "ag_near_field::evaluate: " + std::to_string(ev.region_mismatches) +
            " points lie on different sides of the mesh and the sphere");
    }
    const auto t1 = Clock::now();
    ev.e_mie.resize(np, 3);
    ev.rel.resize(np);
    ev.abs.resize(np);
    ev.mag_rel.resize(np);
    for (Index p = 0; p < np; ++p) {
        const Vec3c ref = mie_total_E(mie, pts.row(p).transpose(), radius);
        ev.e_mie.row(p) = ref.transpose();
        const Vec3c sim = ev.e_sim.row(p).transpose();
        const Real nref = ref.norm();
        ev.abs(p) = (sim - ref).norm();
        ev.rel(p) = ev.abs(p) / nref;
        ev.mag_rel(p) = std::abs(sim.norm() - nref) / nref;
    }
    ev.mie_seconds = std::chrono::duration<Real>(Clock::now() - t1).count();
    return ev;
}

/// Square grid in the xz-plane, x, z in [-half, half], n x n points (post::plane_grid: row =
/// i * n + j, x_i outer, z_j inner). An odd n contains the axes x = 0 and z = 0 exactly.
inline Vertices xz_grid(Real half, Index n) {
    return post::plane_grid(Vec3(-half, 0.0, -half), Vec3(2.0 * half, 0.0, 0.0),
                            Vec3(0.0, 0.0, 2.0 * half), static_cast<int>(n), static_cast<int>(n));
}

/// Grid size for spacing <= max_spacing over [-half, half] (odd, so that the axes are included).
/// Note: for the record's grids 2 half / max_spacing is 120.00000000000001 (1 um, lambda / 20) and
/// 240.00000000000003 (4 um, lambda / 10) in floating point, so ceil gives 121 / 241 and the grids
/// are 123 x 123 and 243 x 243 instead of 121 x 121 and 241 x 241. Kept deliberately: the recorded
/// acceptance runs (benchmarks/results/ag_near_field.md) used these sizes, and the unit test pins
/// them.
inline Index grid_points(Real half, Real max_spacing) {
    auto n = static_cast<Index>(std::ceil(2.0 * half / max_spacing)) + 1;
    if (n % 2 == 0)
        ++n;
    return n;
}

/// Concentric circles ("shells") in the xz-plane at signed surface distances d (> 0 outside,
/// < 0 inside): n_angles points psi_k = 360 k / n_angles deg, (x, z) = (a + d) (sin psi, cos psi)
/// (psi = 0: the shadow pole +z, 90 deg: the +x axis, 180 deg: the lit pole -z). Row =
/// d_index * n_angles + k.
inline Vertices shells(const std::vector<Real>& d, Index n_angles, Real radius) {
    Vertices pts(static_cast<Index>(d.size()) * n_angles, 3);
    Index row = 0;
    for (const Real dd : d) {
        const Real r = radius + dd;
        for (Index k = 0; k < n_angles; ++k) {
            const Real psi =
                2.0 * constants::pi * static_cast<Real>(k) / static_cast<Real>(n_angles);
            pts.row(row++) = Vec3(r * std::sin(psi), 0.0, r * std::cos(psi)).transpose();
        }
    }
    return pts;
}

/// n log-spaced values in [lo, hi] (lo, hi > 0, n >= 2).
inline std::vector<Real> log_space(Real lo, Real hi, int n) {
    if (!(lo > 0.0 && hi > lo && n >= 2))
        throw std::invalid_argument("log_space: needs 0 < lo < hi and n >= 2");
    std::vector<Real> v(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        v[static_cast<std::size_t>(i)] =
            lo * std::pow(hi / lo, static_cast<Real>(i) / static_cast<Real>(n - 1));
    return v;
}

/// Order statistics of a set of values (nearest-rank percentiles).
struct Stats {
    Index count = 0;
    Real max = 0, p99 = 0, p95 = 0, median = 0;
    Vec3 argmax = Vec3::Zero();  ///< point of the maximum
};

inline Stats stats(const std::vector<Real>& values, const std::vector<Vec3>& where) {
    Stats s;
    s.count = static_cast<Index>(values.size());
    if (values.empty())
        return s;
    // Sorted copy as an Eigen vector (std::vector::back() after sort trips a GCC 16
    // -Wnull-dereference false positive).
    VectorXr v(s.count);
    for (Index i = 0; i < s.count; ++i) v(i) = values[static_cast<std::size_t>(i)];
    std::sort(v.data(), v.data() + v.size());
    const auto rank = [&v](Real q) {
        const auto k = static_cast<Index>(std::ceil(q * static_cast<Real>(v.size())));
        return v(std::clamp<Index>(k, 1, v.size()) - 1);
    };
    s.max = v(v.size() - 1);
    s.p99 = rank(0.99);
    s.p95 = rank(0.95);
    s.median = rank(0.5);
    const auto it = std::max_element(values.begin(), values.end());
    s.argmax = where[static_cast<std::size_t>(it - values.begin())];
    return s;
}

/// Statistics of an evaluation per zone for the exclusion distance d_min.
struct ZoneStats {
    Real d_min = 0;
    Stats lit;                  ///< e, lit zone (every exterior point outside the shadow)
    Stats lit_x_axis;           ///< e, lit points on the line z = 0 (|z| < kAxisTolerance a)
    Stats lit_forward;          ///< e, lit points beside the shadow: z > 0, |x| >= a
    Stats shadow;               ///< e, shadow zone (geometric shadow |x| < a, z > 0)
    Stats lit_mag;              ///< magnitude error, lit zone
    Stats shadow_mag;           ///< magnitude error, shadow zone
    Stats interior_abs;         ///< |E_sim - E_Mie| / max|E_inc|, interior zone
    Real interior_max_mie = 0;  ///< max |E_Mie| over the interior points
    Index band = 0;             ///< excluded points
};

/// Zone statistics of the evaluation `ev` at the rows of `pts` (the points `ev` was evaluated at).
inline ZoneStats zone_stats(const Vertices& pts, const Evaluation& ev, Real radius, Real d_min) {
    if (ev.rel.size() != pts.rows() || ev.abs.size() != pts.rows() ||
        ev.mag_rel.size() != pts.rows() || ev.e_mie.rows() != pts.rows())
        throw std::invalid_argument("ag_near_field::zone_stats: evaluation and points differ");
    std::vector<Real> v[7];
    std::vector<Vec3> w[7];
    ZoneStats z;
    z.d_min = d_min;
    const auto add = [&v, &w](int k, Real value, const Vec3& r) {
        v[k].push_back(value);
        w[k].push_back(r);
    };
    for (Index p = 0; p < pts.rows(); ++p) {
        const Vec3 r = pts.row(p).transpose();
        switch (classify(r, radius, d_min)) {
            case Zone::band:
                ++z.band;
                break;
            case Zone::interior:
                add(6, ev.abs(p), r);
                z.interior_max_mie =
                    std::max(z.interior_max_mie, ev.e_mie.row(p).transpose().norm());
                break;
            case Zone::lit:
                add(0, ev.rel(p), r);
                add(4, ev.mag_rel(p), r);
                if (std::abs(r.z()) < kAxisTolerance * radius)
                    add(1, ev.rel(p), r);
                else if (r.z() > 0.0)
                    add(2, ev.rel(p), r);
                break;
            case Zone::shadow:
                add(3, ev.rel(p), r);
                add(5, ev.mag_rel(p), r);
                break;
        }
    }
    z.lit = stats(v[0], w[0]);
    z.lit_x_axis = stats(v[1], w[1]);
    z.lit_forward = stats(v[2], w[2]);
    z.shadow = stats(v[3], w[3]);
    z.lit_mag = stats(v[4], w[4]);
    z.shadow_mag = stats(v[5], w[5]);
    z.interior_abs = stats(v[6], w[6]);
    return z;
}

/// docs/05 criterion: max e < 0.1 (lit: every exterior point beyond d_min outside the geometric
/// shadow), < 1 (shadow: the geometric shadow).
inline constexpr Real kLitTolerance = 0.1;
inline constexpr Real kShadowTolerance = 1.0;

inline bool passes(const ZoneStats& z) {
    return z.lit.count > 0 && z.shadow.count > 0 && z.lit.max < kLitTolerance &&
           z.shadow.max < kShadowTolerance;
}

/// xz grid of one case: x, z in [-half, half] at spacing <= max_spacing; grid points closer than
/// d_eval to the surface are not evaluated at all.
struct Grid {
    Real radius = 0;
    Real half = 0;            ///< [m]
    Index n = 0;              ///< points per axis
    Real spacing = 0;         ///< [m]
    Real d_eval = 0;          ///< [m]
    Vertices points;          ///< all n x n points
    std::vector<Index> rows;  ///< evaluated rows (|d| >= d_eval)
    Vertices eval;            ///< points.rows(rows)
};

inline Grid make_grid(Real radius, Real half, Real max_spacing, Real d_eval) {
    Grid g;
    g.radius = radius;
    g.half = half;
    g.n = grid_points(half, max_spacing);
    g.spacing = 2.0 * half / static_cast<Real>(g.n - 1);
    g.d_eval = d_eval;
    g.points = xz_grid(half, g.n);
    for (Index r = 0; r < g.points.rows(); ++r) {
        if (std::abs(surface_distance(g.points.row(r).transpose(), radius)) >= d_eval)
            g.rows.push_back(r);
    }
    g.eval.resize(static_cast<Index>(g.rows.size()), 3);
    for (std::size_t i = 0; i < g.rows.size(); ++i)
        g.eval.row(static_cast<Index>(i)) = g.points.row(g.rows[i]);
    return g;
}

/// Shell set of the near-surface profile (see shells()).
struct ShellSet {
    Real radius = 0;
    Index n_angles = 0;
    std::vector<Real> d;  ///< signed distances [m], ascending
    Vertices points;
};

/// Default shells of WP24: 360 angles (1 deg), 9 distances lambda / 100 ... lambda / 5 inside and
/// 25 distances lambda / 100 ... lambda outside (log-spaced).
inline ShellSet default_shells(Real radius) {
    ShellSet s;
    s.radius = radius;
    s.n_angles = 360;
    const std::vector<Real> in = log_space(kLambda / 100.0, kLambda / 5.0, 9);
    for (auto it = in.rbegin(); it != in.rend(); ++it) s.d.push_back(-*it);
    for (const Real v : log_space(kLambda / 100.0, kLambda, 25)) s.d.push_back(v);
    s.points = shells(s.d, s.n_angles, radius);
    return s;
}

/// Statistics of one shell (distance index k) of an evaluation on a ShellSet, split into the
/// front half z <= 0 (|z| < kAxisTolerance a counts as z = 0) and the back half z > 0: e for
/// d > 0, |E_sim - E_Mie| for d < 0. This is the near-surface profile diagnostic of the record, not
/// the acceptance zones (zone_stats: the back half of an exterior shell is partly lit, partly
/// geometric shadow).
struct ShellStats {
    Real d = 0;
    Stats front, back;
};

inline ShellStats shell_stats(const ShellSet& s, const Evaluation& ev, std::size_t k) {
    ShellStats st;
    st.d = s.d[k];
    std::vector<Real> v[2];
    std::vector<Vec3> w[2];
    for (Index a = 0; a < s.n_angles; ++a) {
        const Index row = static_cast<Index>(k) * s.n_angles + a;
        const Vec3 r = s.points.row(row).transpose();
        const int side = r.z() < kAxisTolerance * s.radius ? 0 : 1;
        v[side].push_back(st.d > 0.0 ? ev.rel(row) : ev.abs(row));
        w[side].push_back(r);
    }
    st.front = stats(v[0], w[0]);
    st.back = stats(v[1], w[1]);
    return st;
}

/// Field evaluation of the acceptance check for mean edge h: Dunavant degree 6 (the
/// FieldOptions default) and degree 12 within one edge of the surface (see the record:
/// degree 12 agrees with degree 19 within 1 % of e for |d| >= 0.2 h).
inline EvalOptions acceptance_eval(Real h) {
    EvalOptions o;
    o.quad_degree = 6;
    o.near_quad_degree = 12;
    o.near_band = h;
    return o;
}

/// Exclusion distances d_min = f h of the d_min scan of a check (f in units of the mean edge h).
/// f = 0 excludes nothing: every evaluated point counts, i.e. down to the evaluation cut-off
/// lambda / 100 of the grid and the innermost shells (at |d| = lambda / 100 up to rounding).
inline constexpr Real kDMinScanPerEdge[] = {0.0, 0.25, 0.5, 1.0, 2.0};

/// docs/05 check of one solved case: the xz grid x, z in [-1.5 d, 1.5 d] at spacing
/// <= max_spacing and the default shells, evaluated with acceptance_eval(h), zone statistics
/// over the union of both point sets for d_min = kDMinPerEdge h (the acceptance statistic), the
/// same for grid and shells alone, and the union statistics for the d_min of kDMinScanPerEdge.
struct Check {
    Real h = 0;      ///< mean edge [m]
    Real d_min = 0;  ///< [m]
    Index grid_n = 0;
    Real grid_spacing = 0;
    Index points = 0;             ///< evaluated points (grid + shells)
    ZoneStats zones;              ///< union, d_min (acceptance)
    ZoneStats grid_zones;         ///< grid alone, d_min
    ZoneStats shell_zones;        ///< shells alone, d_min
    std::vector<ZoneStats> scan;  ///< union, one entry per kDMinScanPerEdge
    Real field_seconds = 0;
    Real mie_seconds = 0;
};

inline Check check(const post::SurfaceSolution& s, Real diameter, Real max_spacing) {
    const Real radius = 0.5 * diameter;
    Check c;
    c.h = octasphere::edge_stats(s.problem->space->mesh()).mean;
    c.d_min = kDMinPerEdge * c.h;
    const Grid g = make_grid(radius, 1.5 * diameter, max_spacing, kLambda / 100.0);
    const ShellSet sh = default_shells(radius);
    c.grid_n = g.n;
    c.grid_spacing = g.spacing;
    const Index ng = g.eval.rows();
    const Index ns = sh.points.rows();
    Vertices pts(ng + ns, 3);
    pts << g.eval, sh.points;
    c.points = pts.rows();
    const reference::MieSolution mie(
        reference::MieParams{radius, kLambda, material::silver_500nm(), material::vacuum(), 0});
    const Evaluation ev = evaluate(s, mie, pts, radius, acceptance_eval(c.h));
    c.field_seconds = ev.field_seconds;
    c.mie_seconds = ev.mie_seconds;
    c.zones = zone_stats(pts, ev, radius, c.d_min);
    const auto part = [&](Index first, Index count) {
        Evaluation sub;
        sub.e_mie = ev.e_mie.middleRows(first, count);
        sub.rel = ev.rel.segment(first, count);
        sub.abs = ev.abs.segment(first, count);
        sub.mag_rel = ev.mag_rel.segment(first, count);
        return zone_stats(pts.middleRows(first, count), sub, radius, c.d_min);
    };
    c.grid_zones = part(0, ng);
    c.shell_zones = part(ng, ns);
    for (const Real f : kDMinScanPerEdge) c.scan.push_back(zone_stats(pts, ev, radius, f * c.h));
    return c;
}

/// Multi-line summary of a check.
inline std::string summary(const Check& c) {
    std::ostringstream os;
    const auto line = [&os](const char* name, const Stats& s) {
        os << "  " << name << ": " << s.count << " points, max " << s.max << ", p99 " << s.p99
           << ", p95 " << s.p95 << ", median " << s.median << ", max at (x, z) = ("
           << s.argmax.x() * 1e9 << ", " << s.argmax.z() * 1e9 << ") nm\n";
    };
    os << "near field: grid " << c.grid_n << " x " << c.grid_n << " at " << c.grid_spacing * 1e9
       << " nm + shells, " << c.points << " points, h = " << c.h * 1e9
       << " nm, d_min = " << c.d_min * 1e9 << " nm (" << kDMinPerEdge << " h, lambda / "
       << kLambda / c.d_min << "), " << c.zones.band << " points excluded; field "
       << c.field_seconds << " s, Mie " << c.mie_seconds << " s\n";
    os << " union of grid and shells (acceptance; lit = outside the geometric shadow):\n";
    line("lit e", c.zones.lit);
    line("lit e on z = 0", c.zones.lit_x_axis);
    line("lit e beside the shadow (z > 0, |x| >= a)", c.zones.lit_forward);
    line("shadow e (geometric shadow |x| < a, z > 0)", c.zones.shadow);
    line("lit magnitude error", c.zones.lit_mag);
    line("shadow magnitude error", c.zones.shadow_mag);
    line("interior |dE| / |E_inc|", c.zones.interior_abs);
    os << "  interior max |E_Mie| = " << c.zones.interior_max_mie << "\n";
    os << " grid alone:\n";
    line("lit e", c.grid_zones.lit);
    line("shadow e", c.grid_zones.shadow);
    line("interior |dE| / |E_inc|", c.grid_zones.interior_abs);
    os << " shells alone:\n";
    line("lit e", c.shell_zones.lit);
    line("shadow e", c.shell_zones.shadow);
    os << " d_min scan (union): d_min [nm] (d_min / h): lit max / shadow max (excluded points)";
    for (const ZoneStats& z : c.scan) {
        os << "\n  " << z.d_min * 1e9 << " (" << z.d_min / c.h << "): " << z.lit.max << " / "
           << z.shadow.max << " (" << z.band << ")";
    }
    return os.str();
}

}  // namespace ag_near_field
