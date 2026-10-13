/// @file open_surface.cpp
/// Open rough surface with taper, gap and PML zones (ADR 0009, decision 1); see
/// open_surface.hpp for the geometry and the rounding rules.
///
/// Implementation: all band boundaries are integer cell counts on the mesh grid, so the
/// window, the zones and the flat part (z = 0 exactly) are decided by integer comparisons
/// and never by floating-point coordinates. Along each axis (M cells, M + 1 nodes):
///   o = n_t + n_g + n_pml          first patch node,
///   patch nodes  o .. o + M_0,     taper n_t cells, gap n_g cells, PML n_pml cells outside.
/// Node distance from the patch, d = max(o - i, i - (o + M_0), 0) cells, gives the window
/// w = 1 (d = 0), 0.5 (1 + cos(pi d / n_t)) (0 < d < n_t), 0 (d >= n_t). A cell's ring index
/// (0 on the patch, 1 for the first cell outside) gives its zone; both triangles of a cell
/// have their centroid inside the cell, so this is the centroid classification.

#include "specklebem/geometry/open_surface.hpp"

#include "specklebem/core/logging.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace specklebem::geometry {

namespace {

/// Values within kGridTol * h of a grid line or multiple count as exact.
constexpr Real kGridTol = 1e-6;
/// Upper bound on the mesh nodes per axis (memory guard, as generate_gaussian_height_map()).
constexpr Index kMaxNodesPerAxis = Index{1} << 20;

[[noreturn]] void fail(const std::string& what) {
    throw std::invalid_argument("make_open_surface_mesh: " + what);
}

/// Smallest integer n >= value / h - kGridTol (and >= lower); value >= 0 finite, h > 0.
Index cells_up(Real value, Real h, Index lower) {
    const Real q = std::ceil(value / h - kGridTol);
    if (!(q <= static_cast<Real>(kMaxNodesPerAxis))) {
        fail("a zone width spans more than 2^20 grid cells");
    }
    return std::max(lower, static_cast<Index>(q));
}

}  // namespace

Real taper_window(Real t, Real patch_half, Real taper_width) {
    if (!std::isfinite(t) || !std::isfinite(patch_half) || !std::isfinite(taper_width)) {
        throw std::invalid_argument("taper_window: arguments must be finite");
    }
    if (patch_half < 0.0 || taper_width <= 0.0) {
        throw std::invalid_argument("taper_window: need patch_half >= 0 and taper_width > 0");
    }
    const Real d = std::abs(t) - patch_half;
    if (d <= 0.0) {
        return 1.0;
    }
    if (d >= taper_width) {
        return 0.0;
    }
    return 0.5 * (1.0 + std::cos(constants::pi * d / taper_width));
}

OpenSurfaceMesh make_open_surface_mesh(const HeightMap& h, const OpenSurfaceParams& p) {
    // --- validation -------------------------------------------------------------------
    if (!std::isfinite(h.dx) || !std::isfinite(h.dy) || h.dx <= 0.0 || h.dy <= 0.0) {
        fail("the map spacing must be positive and finite");
    }
    if (std::abs(h.dx - h.dy) > 1e-9 * std::max(h.dx, h.dy)) {
        std::ostringstream os;
        os << "the map spacing must be isotropic (dx = " << h.dx << " m, dy = " << h.dy
           << " m); the square bands need common grid lines";
        fail(os.str());
    }
    if (!std::isfinite(p.patch_size) || !std::isfinite(p.taper_width) ||
        !std::isfinite(p.gap_width) || !std::isfinite(p.pml_width)) {
        fail("all OpenSurfaceParams fields must be finite");
    }
    if (p.patch_size <= 0.0) {
        fail("patch_size must be > 0");
    }
    if (p.pml_width <= 0.0) {
        fail("pml_width must be > 0");
    }
    if (p.gap_width < 0.0) {
        fail("gap_width must be >= 0");
    }
    const Real hs = h.dx;
    if (p.taper_width < hs * (1.0 - kGridTol)) {
        std::ostringstream os;
        os << "taper_width (" << p.taper_width << " m) must be at least the map spacing (" << hs
           << " m)";
        fail(os.str());
    }
    const Index cx = h.z.rows() - 1;  // map cells along x
    const Index cy = h.z.cols() - 1;
    if (cx < 1 || cy < 1) {
        fail("the height map needs at least 2 x 2 points");
    }
    if ((cx - cy) % 2 != 0) {
        fail(
            "n_x - 1 and n_y - 1 must have the same parity (common grid lines of the square "
            "bands)");
    }

    // --- rounding to the grid -----------------------------------------------------------
    // Patch: M_0 cells across, M_0 >= L / h - 2 tol with the parity of the map grid.
    Index m0 = cells_up(p.patch_size, hs, 0);
    if ((m0 - cx) % 2 != 0) {
        ++m0;
    }
    if (m0 < 1) {
        fail("patch_size is narrower than one grid cell");
    }
    const Index nt = cells_up(p.taper_width, hs, 1);
    const Index ng = cells_up(p.gap_width, hs, 0);
    const Index np = cells_up(p.pml_width, hs, 1);
    const Index kx = (cx - m0) / 2;  // map cells outside the patch on each side
    const Index ky = (cy - m0) / 2;
    if (kx < nt || ky < nt) {
        std::ostringstream os;
        const Real need = static_cast<Real>(m0 + 2 * nt) * hs;
        os << "the height map (" << static_cast<Real>(cx) * hs << " m x "
           << static_cast<Real>(cy) * hs << " m) does not cover the patch and the taper, " << need
           << " m x " << need << " m (after rounding to the grid)";
        fail(os.str());
    }
    const Index o = nt + ng + np;  // first patch node of the mesh grid
    const Index m = m0 + 2 * o;    // mesh cells per axis
    if (m + 1 > kMaxNodesPerAxis) {
        fail("too many mesh nodes per axis (" + std::to_string(m + 1) + ")");
    }

    OpenSurfaceMesh out;
    const Real half_h = 0.5 * hs;
    out.patch_half = static_cast<Real>(m0) * half_h;
    out.taper_end = static_cast<Real>(m0 + 2 * nt) * half_h;
    out.pml_start = static_cast<Real>(m0 + 2 * (nt + ng)) * half_h;
    out.outer_half = static_cast<Real>(m) * half_h;

    const Real tol = kGridTol * hs;
    const Real wt = static_cast<Real>(nt) * hs;
    const Real wg = static_cast<Real>(ng) * hs;
    const Real wp = static_cast<Real>(np) * hs;
    if (std::abs(out.patch_half - 0.5 * p.patch_size) > tol || std::abs(wt - p.taper_width) > tol ||
        std::abs(wg - p.gap_width) > tol || std::abs(wp - p.pml_width) > tol) {
        SBEM_INFO(
            "make_open_surface_mesh: zones rounded up to the grid (h = {:.6g} m): L {:.6g} -> "
            "{:.6g} m, W_t {:.6g} -> {:.6g} m, W_g {:.6g} -> {:.6g} m, W_pml {:.6g} -> {:.6g} m",
            hs, p.patch_size, 2.0 * out.patch_half, p.taper_width, wt, p.gap_width, wg, p.pml_width,
            wp);
    }

    // --- window and node distance -------------------------------------------------------
    const Index n = m + 1;  // nodes per axis
    std::vector<Real> w(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i) {
        const Index d = std::max({o - i, i - (o + m0), Index{0}});
        Real wi = 0.0;
        if (d == 0) {
            wi = 1.0;
        } else if (d < nt) {
            wi = 0.5 *
                 (1.0 + std::cos(constants::pi * static_cast<Real>(d) / static_cast<Real>(nt)));
        }
        w[static_cast<std::size_t>(i)] = wi;
    }

    // --- vertices -----------------------------------------------------------------------
    Vertices v(n * n, 3);
    const Index shift_x = kx - o;  // map index = mesh index + shift
    const Index shift_y = ky - o;
    for (Index i = 0; i < n; ++i) {
        const Real x = static_cast<Real>(2 * i - m) * half_h;
        const Real wx = w[static_cast<std::size_t>(i)];
        for (Index j = 0; j < n; ++j) {
            const Real y = static_cast<Real>(2 * j - m) * half_h;
            const Real wy = w[static_cast<std::size_t>(j)];
            Real z = 0.0;
            if (wx != 0.0 && wy != 0.0) {
                const Real xi = h.z(i + shift_x, j + shift_y);
                if (!std::isfinite(xi)) {
                    std::ostringstream os;
                    os << "non-finite height at map node (" << i + shift_x << ", " << j + shift_y
                       << ")";
                    fail(os.str());
                }
                z = xi * wx * wy;
            }
            const Index row = i * n + j;
            v(row, 0) = x;
            v(row, 1) = y;
            v(row, 2) = z;
        }
    }

    // --- triangles and zones ------------------------------------------------------------
    // Ring index of cell c along one axis: 0 on the patch, k for the k-th cell outside.
    const auto ring = [o, m0](Index c) -> Index {
        if (c < o) {
            return o - c;
        }
        if (c >= o + m0) {
            return c - (o + m0) + 1;
        }
        return 0;
    };
    const Index nf = 2 * m * m;
    Triangles f(nf, 3);
    out.zone.resize(static_cast<std::size_t>(nf));
    for (Index i = 0; i < m; ++i) {
        const Index ri = ring(i);
        for (Index j = 0; j < m; ++j) {
            const Index t = 2 * (i * m + j);
            // Cell corners a (i,j), b (i+1,j), c (i+1,j+1), d (i,j+1); diagonal a-c as on the
            // box top face. Corner indices go into locals before the stores (see the GCC 13
            // vectoriser note in rough_surface.cpp).
            const Index a = i * n + j;
            const Index b = a + n;
            const Index c = b + 1;
            const Index d = a + 1;
            // Clockwise seen from +z = counter-clockwise seen from R1 (z < 0): n_z < 0.
            f(t, 0) = a;
            f(t, 1) = c;
            f(t, 2) = b;
            f(t + 1, 0) = a;
            f(t + 1, 1) = d;
            f(t + 1, 2) = c;
            const Index r = std::max(ri, ring(j));
            SurfaceZone zt = SurfaceZone::Pml;
            if (r == 0) {
                zt = SurfaceZone::Patch;
            } else if (r <= nt) {
                zt = SurfaceZone::Taper;
            } else if (r <= nt + ng) {
                zt = SurfaceZone::Gap;
            }
            out.zone[static_cast<std::size_t>(t)] = zt;
            out.zone[static_cast<std::size_t>(t + 1)] = zt;
        }
    }

    out.mesh = TriangleMesh(std::move(v), std::move(f));
    return out;
}

CoordinateStretch make_pml_stretch(const OpenSurfaceMesh& m, Real sigma_max, Real order) {
    const StretchProfile prof{m.pml_start, m.outer_half - m.pml_start, sigma_max, order};
    return {prof, prof};
}

}  // namespace specklebem::geometry
