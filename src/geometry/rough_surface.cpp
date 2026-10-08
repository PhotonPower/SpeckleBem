/// @file rough_surface.cpp
/// Gaussian random rough surfaces (Fu et al. 2023, Eqs. 8-9) and the closed box mesh
/// built on them (ADR 0006).
///
/// Height map
///  * Grid: n = round(L / mesh_size) + 1 points per axis, dx = dy = L / (n - 1).
///  * Kernel: W(x, y) = exp(-2 (x^2 + y^2) / Lc^2), sampled on the grid offsets
///    (a dx, b dy), truncated at radius R_W = 4 Lc (W(R_W) = e^-32) and normalised to
///    sum_k W_k^2 = 1. For unit white noise the output variance is then exactly 1 and the
///    normalised autocorrelation is sum_k W_k W_{k+r} = exp(-r^2 / Lc^2) (up to the
///    negligible truncation and sampling errors), i.e. Lc is the 1/e radius.
///  * Stationarity: the noise is drawn on the grid enlarged by R = floor(R_W / dx) cells on
///    every side; only the "valid" part of the linear convolution (kernel fully inside the
///    noise grid) is kept, which is exactly the L x L patch.
///  * FFT path: 2D circular convolution of size P_x x P_y with P >= the noise grid size,
///    P the next 5-smooth number (factors 2, 3, 5: fast kissfft radices). For the valid
///    outputs the circular wrap-around never reaches the noise, so the result equals the
///    zero-padded linear convolution. Eigen's unsupported FFT module (header-only kissfft
///    backend) supplies the 1D transforms; the 2D transform is row-column.
///  * Direct path: the double loop over output points and kernel entries on the same
///    noise; it is the reference oracle for the FFT path.
///  * Randomness: std::normal_distribution / std::uniform_real_distribution are
///    implementation-defined, so the Gaussian draws are implemented here:
///    std::mt19937_64 (fully specified by the standard) -> 53-bit uniform in (0, 1) ->
///    Box-Muller (both outputs of each pair are used). The noise is drawn in x-major order
///    (outer loop over x, inner over y) on the enlarged grid. This makes a fixed seed
///    reproducible across standard libraries up to last-bit differences of log/cos/sin.
///
/// Box mesh: see make_mesh_from_height_map() in rough_surface.hpp.
///  * Uniform box (M = 0 coarsening levels; the WP2 mesh, kept bit-identical). Vertex
///    layout: top grid (n_x n_y), bottom grid (n_x n_y), then the interior wall vertices
///    (n_rim (n_z - 1)), where n_rim = 2 (n_x - 1) + 2 (n_y - 1). Triangle count:
///    4 (n_x - 1)(n_y - 1) + 2 n_rim n_z.
///  * Graded box (M >= 1, WP2b). Vertex layout: top grid (n_x n_y), bottom grid
///    ((n_cx + 1)(n_cy + 1), the level-M nodes), then the interior wall rows k = 1 .. K-1,
///    each a ring of R_m vertices, R_m = 2 n_x,m + 2 n_y,m for the level m of the row
///    (n_x,m cells of level m along x). Triangles: top face (2 (n_x - 1)(n_y - 1)), walls
///    strip by strip (2 R_m between two rows of level m, R_m + R_(m+1) for a transition
///    strip from level m to m + 1), bottom plate (2 n_cx n_cy).
#include "specklebem/geometry/rough_surface.hpp"

#include "specklebem/core/logging.hpp"
#include "specklebem/core/timer.hpp"

#include <unsupported/Eigen/FFT>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace specklebem::geometry {

namespace {

constexpr Real kDefaultBoxDepth = 2e-6;     ///< ADR 0006 fixed minimum [m]
constexpr Real kKernelRadiusInLc = 4.0;     ///< kernel truncation radius / Lc
constexpr Real kRoundingSlack = 1e-9;       ///< relative slack for floor/ceil of ratios
constexpr Real kMaxGridPoints = 1048576.0;  ///< 2^20 grid points per axis
const Real kInvE = std::exp(-1.0);

void require_positive_finite(Real v, const char* name) {
    if (!(v > 0.0) || !std::isfinite(v)) {
        throw std::invalid_argument(std::string("rough surface: ") + name +
                                    " must be positive and finite, got " + std::to_string(v));
    }
}

/// Grid points per axis for the generated map.
Index grid_points(const RoughSurfaceParams& p) {
    require_positive_finite(p.edge_length_L, "edge_length_L");
    require_positive_finite(p.mesh_size, "mesh_size");
    require_positive_finite(p.correlation_length, "correlation_length");
    if (!(p.rms_roughness >= 0.0) || !std::isfinite(p.rms_roughness)) {
        throw std::invalid_argument("rough surface: rms_roughness must be non-negative and finite");
    }
    // Check the ratio before rounding / allocating: the grid must have 2 <= n <= 2^20
    // points per axis.
    const Real cells = p.edge_length_L / p.mesh_size;
    if (!(cells + 1.0 <= kMaxGridPoints)) {
        throw std::invalid_argument(
            "rough surface: edge_length_L / mesh_size = " + std::to_string(cells) +
            " gives more than 2^20 grid points " + "per axis");
    }
    const Index n = static_cast<Index>(std::llround(cells)) + 1;
    if (n < 2) {
        throw std::invalid_argument(
            "rough surface: edge_length_L / mesh_size must round to at least 1 (grid >= 2 x 2)");
    }
    // An under-resolved Lc degenerates the kernel towards a single point (white noise).
    const Real d = p.edge_length_L / static_cast<Real>(n - 1);
    if (p.correlation_length < 2.0 * d * (1.0 - kRoundingSlack)) {
        std::ostringstream os;
        os << "rough surface: correlation_length " << p.correlation_length
           << " m is under-resolved by the grid: it must be at least 2 * max(dx, dy) = " << 2.0 * d
           << " m";
        throw std::invalid_argument(os.str());
    }
    return n;
}

Real default_or(std::optional<Real> depth) {
    return depth.has_value() ? *depth : kDefaultBoxDepth;
}

/// Seeded 64-bit Mersenne twister; seed 0 draws a seed from std::random_device.
std::uint64_t effective_seed(std::uint64_t seed) {
    if (seed != 0)
        return seed;
    std::random_device rd;
    const auto hi = static_cast<std::uint64_t>(rd());
    const auto lo = static_cast<std::uint64_t>(rd());
    return (hi << 32) ^ lo;
}

/// Uniform in the open interval (0, 1) with 53 random bits.
Real uniform_open01(std::mt19937_64& rng) {
    return (static_cast<Real>(rng() >> 11) + 0.5) * 0x1.0p-53;
}

/// n_x x n_y standard normal samples (Box-Muller), filled in x-major order.
MatrixXr gaussian_noise(Index nx, Index ny, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    MatrixXr f(nx, ny);
    bool have_spare = false;
    Real spare = 0.0;
    for (Index i = 0; i < nx; ++i) {
        for (Index j = 0; j < ny; ++j) {
            if (have_spare) {
                f(i, j) = spare;
                have_spare = false;
                continue;
            }
            const Real u1 = uniform_open01(rng);
            const Real u2 = uniform_open01(rng);
            const Real r = std::sqrt(-2.0 * std::log(u1));
            const Real phi = 2.0 * constants::pi * u2;
            f(i, j) = r * std::cos(phi);
            spare = r * std::sin(phi);
            have_spare = true;
        }
    }
    return f;
}

/// Kernel samples W(a dx, b dy) for |a| <= rx, |b| <= ry, zero outside radius 4 Lc,
/// normalised to sum W^2 = 1. Index (a + rx, b + ry).
MatrixXr gaussian_kernel(Real lc, Real dx, Real dy, Index rx, Index ry) {
    const Real radius = kKernelRadiusInLc * lc;
    const Real r2_max = radius * radius * (1.0 + kRoundingSlack);
    MatrixXr w = MatrixXr::Zero(2 * rx + 1, 2 * ry + 1);
    for (Index a = -rx; a <= rx; ++a) {
        for (Index b = -ry; b <= ry; ++b) {
            const Real x = static_cast<Real>(a) * dx;
            const Real y = static_cast<Real>(b) * dy;
            const Real r2 = x * x + y * y;
            if (r2 <= r2_max)
                w(a + rx, b + ry) = std::exp(-2.0 * r2 / (lc * lc));
        }
    }
    w /= w.norm();  // Frobenius norm: sum W^2 = 1
    return w;
}

/// Smallest m >= n whose only prime factors are 2, 3 and 5.
Index next_smooth(Index n) {
    for (Index m = std::max<Index>(n, 1);; ++m) {
        Index r = m;
        for (const Index p : {Index{2}, Index{3}, Index{5}}) {
            while (r % p == 0) r /= p;
        }
        if (r == 1)
            return m;
    }
}

/// In-place 2D DFT (row-column) of a column-major complex matrix; the inverse is scaled
/// by 1 / (rows cols) (Eigen::FFT default).
void fft2(MatrixXc& a, bool inverse) {
    Eigen::FFT<Real> fft;
    const Index rows = a.rows();
    const Index cols = a.cols();
    std::vector<Complex> in(static_cast<std::size_t>(std::max(rows, cols)));
    std::vector<Complex> out(in.size());
    for (Index c = 0; c < cols; ++c) {
        for (Index r = 0; r < rows; ++r) in[static_cast<std::size_t>(r)] = a(r, c);
        if (inverse)
            fft.inv(out.data(), in.data(), rows);
        else
            fft.fwd(out.data(), in.data(), rows);
        for (Index r = 0; r < rows; ++r) a(r, c) = out[static_cast<std::size_t>(r)];
    }
    for (Index r = 0; r < rows; ++r) {
        for (Index c = 0; c < cols; ++c) in[static_cast<std::size_t>(c)] = a(r, c);
        if (inverse)
            fft.inv(out.data(), in.data(), cols);
        else
            fft.fwd(out.data(), in.data(), cols);
        for (Index c = 0; c < cols; ++c) a(r, c) = out[static_cast<std::size_t>(c)];
    }
}

/// Valid part of the linear convolution noise (*) w via zero-padded FFTs. The output
/// has size (noise.rows() - w.rows() + 1) x (noise.cols() - w.cols() + 1).
MatrixXr convolve_valid_fft(const MatrixXr& noise, const MatrixXr& w) {
    const Index px = next_smooth(noise.rows());
    const Index py = next_smooth(noise.cols());
    MatrixXc a = MatrixXc::Zero(px, py);
    a.topLeftCorner(noise.rows(), noise.cols()) = noise.cast<Complex>();
    MatrixXc k = MatrixXc::Zero(px, py);
    k.topLeftCorner(w.rows(), w.cols()) = w.cast<Complex>();
    fft2(a, false);
    fft2(k, false);
    a.array() *= k.array();
    fft2(a, true);
    const Index ox = w.rows() - 1;
    const Index oy = w.cols() - 1;
    return a.block(ox, oy, noise.rows() - ox, noise.cols() - oy).real();
}

/// Same as convolve_valid_fft by the direct double loop (reference oracle).
MatrixXr convolve_valid_direct(const MatrixXr& noise, const MatrixXr& w) {
    const Index mx = w.rows();
    const Index my = w.cols();
    const Index nx = noise.rows() - mx + 1;
    const Index ny = noise.cols() - my + 1;
    MatrixXr out(nx, ny);
    for (Index i = 0; i < nx; ++i) {
        for (Index j = 0; j < ny; ++j) {
            // out(i, j) = sum_{a, b} w(a, b) noise(i + mx - 1 - a, j + my - 1 - b)
            Real s = 0.0;
            for (Index a = 0; a < mx; ++a) {
                for (Index b = 0; b < my; ++b) {
                    const Real wab = w(a, b);
                    if (wab != 0.0)
                        s += wab * noise(i + mx - 1 - a, j + my - 1 - b);
                }
            }
            out(i, j) = s;
        }
    }
    return out;
}

/// Lag (in samples) of the first 1/e crossing of the normalised autocorrelation along
/// the rows (along x) of d, averaged over the columns; linear interpolation.
Real first_crossing_along_rows(const MatrixXr& d, Real c0) {
    const Index n = d.rows();
    Real prev = 1.0;
    for (Index k = 1; k < n; ++k) {
        const Index m = n - k;
        const Real ck = (d.topRows(m).array() * d.bottomRows(m).array()).sum() /
                        (static_cast<Real>(m * d.cols()) * c0);
        if (ck < kInvE)
            return static_cast<Real>(k - 1) + (prev - kInvE) / (prev - ck);
        prev = ck;
    }
    throw std::runtime_error(
        "HeightMap::estimated_correlation_length: autocorrelation has no 1/e crossing within "
        "the map");
}

/// Wall rows for the given depth and spacing.
Index wall_rows(Real depth, Real h) {
    return std::max<Index>(1, static_cast<Index>(std::ceil(depth / h * (1.0 - kRoundingSlack))));
}

Index rim_points(Index nx, Index ny) {
    return 2 * (nx - 1) + 2 * (ny - 1);
}

Index box_triangle_count(Index nx, Index ny, Index nz) {
    return 4 * (nx - 1) * (ny - 1) + 2 * rim_points(nx, ny) * nz;
}

void validate_height_map(const HeightMap& h, Real depth) {
    if (h.z.rows() < 2 || h.z.cols() < 2) {
        throw std::invalid_argument("make_mesh_from_height_map: grid must be at least 2 x 2, got " +
                                    std::to_string(h.z.rows()) + " x " +
                                    std::to_string(h.z.cols()));
    }
    require_positive_finite(h.dx, "dx");
    require_positive_finite(h.dy, "dy");
    if (!h.z.allFinite()) {
        throw std::invalid_argument("make_mesh_from_height_map: heights must be finite");
    }
    if (!std::isfinite(depth)) {
        throw std::invalid_argument("make_mesh_from_height_map: box depth must be finite");
    }
    const Real zmax = h.z.maxCoeff();
    const Real hmax = std::max(h.dx, h.dy);
    if (!(depth > zmax + hmax)) {
        std::ostringstream os;
        os << "make_mesh_from_height_map: box depth " << depth
           << " m must exceed max(xi) + max(dx, dy) = " << zmax + hmax
           << " m (the bottom plate would intersect the rough top face)";
        throw std::invalid_argument(os.str());
    }
}

/// Uniform box (WP2 mesh, M = 0): walls and bottom at the top-face spacing. Kept unchanged
/// so that box_mesh_size = top-face spacing reproduces the WP2 arrays bit for bit.
/// The caller has validated the map and the depth.
std::pair<Vertices, Triangles> uniform_box_arrays(const HeightMap& h, Real depth) {
    const Index nx = h.z.rows();
    const Index ny = h.z.cols();
    const Index nz = wall_rows(depth, std::min(h.dx, h.dy));
    const Index n_rim = rim_points(nx, ny);
    const Index n_grid = nx * ny;
    const Index nv = 2 * n_grid + n_rim * (nz - 1);
    const Index nf = box_triangle_count(nx, ny, nz);

    const Real x0 = -0.5 * static_cast<Real>(nx - 1) * h.dx;
    const Real y0 = -0.5 * static_cast<Real>(ny - 1) * h.dy;
    const auto xi = [&](Index i) { return x0 + static_cast<Real>(i) * h.dx; };
    const auto yj = [&](Index j) { return y0 + static_cast<Real>(j) * h.dy; };
    const auto top = [ny](Index i, Index j) { return i * ny + j; };
    const auto bottom = [ny, n_grid](Index i, Index j) { return n_grid + i * ny + j; };

    Vertices v(nv, 3);
    const auto set_vertex = [&v](Index k, Real x, Real y, Real z) {
        v(k, 0) = x;
        v(k, 1) = y;
        v(k, 2) = z;
    };
    for (Index i = 0; i < nx; ++i) {
        for (Index j = 0; j < ny; ++j) {
            const Index k = top(i, j);  // bottom(i, j) = k + n_grid
            set_vertex(k, xi(i), yj(j), h.z(i, j));
            set_vertex(k + n_grid, xi(i), yj(j), depth);
        }
    }

    // Rim walked counter-clockwise seen from +z: (0,0) -> (nx-1,0) -> (nx-1,ny-1) ->
    // (0,ny-1) -> (0,0).
    std::vector<std::pair<Index, Index>> rim;
    rim.reserve(static_cast<std::size_t>(n_rim));
    for (Index i = 0; i < nx - 1; ++i) rim.emplace_back(i, 0);
    for (Index j = 0; j < ny - 1; ++j) rim.emplace_back(nx - 1, j);
    for (Index i = nx - 1; i > 0; --i) rim.emplace_back(i, ny - 1);
    for (Index j = ny - 1; j > 0; --j) rim.emplace_back(0, j);

    // Wall column r, level k (0 = rim on the top face, nz = bottom plate).
    const auto column = [&](Index r, Index k) -> Index {
        const auto [i, j] = rim[static_cast<std::size_t>(r)];
        if (k == 0)
            return top(i, j);
        if (k == nz)
            return bottom(i, j);
        return 2 * n_grid + r * (nz - 1) + (k - 1);
    };
    for (Index r = 0; r < n_rim; ++r) {
        const auto [i, j] = rim[static_cast<std::size_t>(r)];
        const Real z_rim = h.z(i, j);
        for (Index k = 1; k < nz; ++k) {
            const Real t = static_cast<Real>(k) / static_cast<Real>(nz);
            set_vertex(column(r, k), xi(i), yj(j), z_rim + (depth - z_rim) * t);
        }
    }

    // Corner indices are computed into locals first and every triangle is written at an
    // index computed from its cell. Calling the top() / bottom() helpers inside the
    // set_triangle() arguments was miscompiled: the bottom-plate triangles came out wrong.
    // Verified: no UB; minimal reproducer in plain C
    // (docs/compiler_notes/gcc13_vectorizer_store_permutation.c); GCC 13.3 x86-64 -O3 loop
    // vectoriser (interleaved store-group permutation); fine with -fno-tree-loop-vectorize,
    // -O2 and clang 18. The unit tests check the emitted topology and orientation in the
    // release build.
    Triangles f(nf, 3);
    const auto set_triangle = [&f](Index t, Index a, Index b, Index c) {
        f(t, 0) = a;
        f(t, 1) = b;
        f(t, 2) = c;
    };
    for (Index i = 0; i + 1 < nx; ++i) {
        for (Index j = 0; j + 1 < ny; ++j) {
            const Index t = 4 * (i * (ny - 1) + j);
            // Cell corners a (i,j), b (i+1,j), c (i+1,j+1), d (i,j+1); diagonal a-c.
            const Index a = top(i, j);
            const Index b = a + ny;
            const Index c = b + 1;
            const Index d = a + 1;
            // Top face: clockwise seen from +z, i.e. counter-clockwise seen from z < 0
            // (outside the box), so n_z < 0.
            set_triangle(t, a, c, b);
            set_triangle(t + 1, a, d, c);
            // Bottom plate (bottom(i, j) = top(i, j) + n_grid): counter-clockwise seen
            // from +z, n_z > 0.
            set_triangle(t + 2, n_grid + a, n_grid + b, n_grid + c);
            set_triangle(t + 3, n_grid + a, n_grid + c, n_grid + d);
        }
    }
    // Walls: for the rim walked counter-clockwise seen from +z, the quad (r0, r1, s1, s0)
    // with s below r (larger z) is counter-clockwise seen from outside.
    const Index wall_base = 4 * (nx - 1) * (ny - 1);
    for (Index r = 0; r < n_rim; ++r) {
        const Index rn = (r + 1) % n_rim;
        for (Index k = 0; k < nz; ++k) {
            const Index t = wall_base + 2 * (r * nz + k);
            const Index r0 = column(r, k);
            const Index r1 = column(rn, k);
            const Index s0 = column(r, k + 1);
            const Index s1 = column(rn, k + 1);
            set_triangle(t, r0, r1, s1);
            set_triangle(t + 1, r0, s1, s0);
        }
    }
    if (wall_base + 2 * n_rim * nz != nf) {
        throw std::logic_error("uniform_box_arrays: triangle count mismatch");
    }
    return {std::move(v), std::move(f)};
}

/// Mean height of the rim points (the wall columns start there).
Real mean_rim_height(const HeightMap& h) {
    const Index nx = h.z.rows();
    const Index ny = h.z.cols();
    Real sum = 0.0;
    for (Index i = 0; i + 1 < nx; ++i) sum += h.z(i, 0) + h.z(i + 1, ny - 1);
    for (Index j = 0; j + 1 < ny; ++j) sum += h.z(nx - 1, j) + h.z(0, j + 1);
    return sum / static_cast<Real>(rim_points(nx, ny));
}

/// 2^m as an integer and as a Real.
Index pow2i(Index m) {
    return Index{1} << m;
}
Real pow2r(Index m) {
    return static_cast<Real>(pow2i(m));
}

/// Nested node sets of one axis with n cells: levels[m] holds the sorted grid indices
/// (0 .. n) of the level-m nodes. levels[M] has n_c = ceil(n / 2^M) cells, cell k spanning
/// floor(k n / n_c) .. floor((k + 1) n / n_c); each lower level halves every cell (larger
/// half first), so levels[0] = {0, 1, ..., n} and every level-(m+1) cell consists of one or
/// two level-m cells.
std::vector<std::vector<Index>> axis_levels(Index n, Index levels) {
    std::vector<std::vector<Index>> out(static_cast<std::size_t>(levels + 1));
    const Index p2 = pow2i(levels);
    const Index nc = (n + p2 - 1) / p2;
    std::vector<Index>& coarsest = out.back();
    for (Index k = 0; k <= nc; ++k) {
        const Index node = k * n / nc;
        coarsest.push_back(node);
    }
    for (Index m = levels; m > 0; --m) {
        const std::vector<Index>& coarse = out[static_cast<std::size_t>(m)];
        std::vector<Index>& fine = out[static_cast<std::size_t>(m - 1)];
        for (std::size_t s = 0; s + 1 < coarse.size(); ++s) {
            const Index a = coarse[s];
            const Index b = coarse[s + 1];
            fine.push_back(a);
            if (b - a >= 2) {
                const Index mid = a + (b - a + 1) / 2;
                fine.push_back(mid);
            }
        }
        fine.push_back(n);
    }
    const std::vector<Index>& finest = out.front();
    bool complete = static_cast<Index>(finest.size()) == n + 1;
    for (std::size_t s = 0; complete && s < finest.size(); ++s) {
        complete = finest[s] == static_cast<Index>(s);
    }
    if (!complete) {
        throw std::logic_error("axis_levels: the finest level is not the full grid");
    }
    return out;
}

using GridNode = std::pair<Index, Index>;

/// Ring of level nodes (grid indices (i, j) on the rim) walked counter-clockwise seen
/// from +z, starting at (0, 0); for the full axes this is the rim of uniform_box_arrays().
std::vector<GridNode> level_ring(const std::vector<Index>& px, const std::vector<Index>& py) {
    const Index cx = px.back();
    const Index cy = py.back();
    const std::size_t sx = px.size() - 1;
    const std::size_t sy = py.size() - 1;
    std::vector<GridNode> ring;
    ring.reserve(2 * (sx + sy));
    for (std::size_t s = 0; s < sx; ++s) ring.emplace_back(px[s], 0);
    for (std::size_t s = 0; s < sy; ++s) ring.emplace_back(cx, py[s]);
    for (std::size_t s = sx; s > 0; --s) ring.emplace_back(px[s], cy);
    for (std::size_t s = sy; s > 0; --s) ring.emplace_back(0, py[s]);
    return ring;
}

/// Resolved layout of the closing box.
struct BoxPlan {
    Index levels = 0;                    ///< M (0 = uniform WP2 box)
    Index levels_unreduced = 0;          ///< M before the rim check (transition_cells_ok)
    Real target = 0.0;                   ///< requested / automatic coarse spacing h_c [m]
    bool target_below_top = false;       ///< requested h_c < h / sqrt(2) (warned once)
    Real h = 0.0;                        ///< top-face spacing min(dx, dy) [m]
    Real hb = 0.0;                       ///< level spacing max(dx, dy) [m] (sets M)
    Real hg = 0.0;                       ///< band spacing sqrt(dx dy) [m] (row heights)
    Real height = 0.0;                   ///< wall height H = depth - mean rim height [m]
    std::vector<std::vector<Index>> px;  ///< node levels along x (axis_levels)
    std::vector<std::vector<Index>> py;  ///< node levels along y
    std::vector<Index> row_level;        ///< level of wall row k = 0 .. K
    std::vector<Real> row_fraction;      ///< height fraction of wall row k (0 rim, 1 bottom)

    [[nodiscard]] Index cells_x(Index m) const {
        return static_cast<Index>(px[static_cast<std::size_t>(m)].size()) - 1;
    }
    [[nodiscard]] Index cells_y(Index m) const {
        return static_cast<Index>(py[static_cast<std::size_t>(m)].size()) - 1;
    }
    [[nodiscard]] Index ring_size(Index m) const { return 2 * cells_x(m) + 2 * cells_y(m); }
    [[nodiscard]] Index rows() const { return static_cast<Index>(row_level.size()) - 1; }
};

/// Node levels and wall rows of the plan for M = m (see make_mesh_from_height_map()).
void set_levels(BoxPlan& plan, Index cx, Index cy, Real depth, Index m) {
    plan.levels = m;
    plan.px = axis_levels(cx, m);
    plan.py = axis_levels(cy, m);
    plan.row_level.clear();
    plan.row_fraction.clear();
    if (m == 0) {
        // Uniform box (uniform_box_arrays): n_z rows of level 0.
        const Index nz = wall_rows(depth, plan.h);
        for (Index k = 0; k <= nz; ++k) {
            plan.row_level.push_back(0);
            plan.row_fraction.push_back(static_cast<Real>(k) / static_cast<Real>(nz));
        }
        return;
    }
    // Rim row, one fine row (height h_g), M transition rows (heights 2^l h_g,
    // l = 0 .. M-1), then uniform coarse rows of height <= 2^M h_g down to the bottom.
    // h_g = sqrt(dx dy) balances the two wall orientations of anisotropic maps: with
    // min(dx, dy) the 2:1 cells on the walls along the coarser axis become flat slivers
    // (aspect ratio 8.6 for dx / dy = 4), with max(dx, dy) the cells on the walls along the
    // finer axis that are halved unevenly (up to 2x narrower) become 1:8 needles (4.6).
    std::vector<Real> zeta{0.0, plan.hg};
    plan.row_level = {0, 0};
    for (Index l = 0; l < m; ++l) {
        const Real band = pow2r(l) * plan.hg;
        zeta.push_back(zeta.back() + band);
        plan.row_level.push_back(l + 1);
    }
    const Real coarse = pow2r(m) * plan.hg;
    const Real start = zeta.back();
    const Real rest = plan.height - start;
    const Index n_coarse =
        std::max<Index>(1, static_cast<Index>(std::ceil(rest / coarse * (1.0 - kRoundingSlack))));
    for (Index q = 1; q <= n_coarse; ++q) {
        zeta.push_back(start + rest * static_cast<Real>(q) / static_cast<Real>(n_coarse));
        plan.row_level.push_back(m);
    }
    for (const Real z : zeta) plan.row_fraction.push_back(z / plan.height);
    plan.row_fraction.back() = 1.0;
}

/// Shape check of the 2:1 transition cells. Wall row k follows the rim scaled towards
/// the bottom (z = z_rim + (depth - z_rim) t_k), so a strongly curved rim can push the
/// middle upper node r1 of a 2:1 cell below the lower edge s0-s1 (inverted triangle
/// (r1, s1, s0)). Required: r1 lies at least half its nominal band height
/// (depth - z_rim)(t_(k+1) - t_k) above the edge s0-s1 (linear interpolation at r1).
/// Quad (1:1) cells cannot invert because every wall column is monotone in z.
bool transition_cells_ok(const HeightMap& h, Real depth, const BoxPlan& plan) {
    // Position along the side (one of i, j is constant on a side, so this is monotone).
    const auto along = [&h](const GridNode& n) {
        return static_cast<Real>(n.first) * h.dx + static_cast<Real>(n.second) * h.dy;
    };
    const auto row_z = [&h, depth](const GridNode& n, Real t) {
        const Real z_rim = h.z(n.first, n.second);
        return z_rim + (depth - z_rim) * t;
    };
    for (std::size_t k = 0; k + 1 < plan.row_level.size(); ++k) {
        const Index m = plan.row_level[k];
        if (plan.row_level[k + 1] == m)
            continue;
        const auto ms = static_cast<std::size_t>(m);
        const std::vector<GridNode> upper = level_ring(plan.px[ms], plan.py[ms]);
        const std::vector<GridNode> lower = level_ring(plan.px[ms + 1], plan.py[ms + 1]);
        const Real t_up = plan.row_fraction[k];
        const Real t_lo = plan.row_fraction[k + 1];
        std::size_t u = 0;
        for (std::size_t s = 0; s < lower.size() && u < upper.size(); ++s) {
            const GridNode& s0 = lower[s];
            const GridNode& s1 = lower[(s + 1) % lower.size()];
            const GridNode& r1 = upper[(u + 1) % upper.size()];
            if (r1 == s1) {
                u += 1;
                continue;
            }
            u += 2;
            const Real w = (along(r1) - along(s0)) / (along(s1) - along(s0));
            const Real z_edge = (1.0 - w) * row_z(s0, t_lo) + w * row_z(s1, t_lo);
            const Real z_r1 = row_z(r1, t_up);
            const Real band = (depth - h.z(r1.first, r1.second)) * (t_lo - t_up);
            if (!(z_edge - z_r1 >= 0.5 * band))
                return false;
        }
    }
    return true;
}

/// Validates the input and resolves the grading (rules: make_mesh_from_height_map() in
/// rough_surface.hpp). Logs nothing: the warnings of the plan are emitted once per mesh by
/// box_mesh() (warn_plan()), not by the detail:: inspection functions.
BoxPlan make_box_plan(const HeightMap& h, Real depth, std::optional<Real> box_mesh_size) {
    validate_height_map(h, depth);
    const Index cx = h.z.rows() - 1;
    const Index cy = h.z.cols() - 1;
    BoxPlan plan;
    plan.h = std::min(h.dx, h.dy);
    plan.hb = std::max(h.dx, h.dy);
    plan.hg = std::sqrt(h.dx * h.dy);
    const Real sqrt2 = std::sqrt(2.0);
    if (box_mesh_size.has_value()) {
        require_positive_finite(*box_mesh_size, "box_mesh_size");
        plan.target = *box_mesh_size;
        plan.target_below_top = plan.target < plan.h / sqrt2;
    } else {
        const Real side = std::min(static_cast<Real>(cx) * h.dx, static_cast<Real>(cy) * h.dy);
        plan.target = std::min({0.5 * depth, 0.125 * side, 10.0 * plan.h});
    }
    plan.height = depth - mean_rim_height(h);

    // M = round(log2(h_c / h_b)), evaluated by doubling (no overflow for huge ratios), with
    // 2^M <= min(cx, cy) and 1.5 * 2^M h_g <= H (room for the fine row, the transition rows
    // and at least half a coarse row).
    const Real ratio = plan.target / plan.hb;
    const Index cmin = std::min(cx, cy);
    Index m = 0;
    while (pow2i(m + 1) <= cmin && ratio >= sqrt2 * pow2r(m) &&
           1.5 * pow2r(m + 1) * plan.hg <= plan.height) {
        ++m;
    }
    plan.levels_unreduced = m;
    // A rim too rough for the transition strips reduces M (documented in the header).
    for (;; --m) {
        set_levels(plan, cx, cy, depth, m);
        if (m == 0 || transition_cells_ok(h, depth, plan))
            break;
    }
    return plan;
}

/// The warnings of a plan, one line each, emitted once per mesh built.
void warn_plan(const BoxPlan& plan) {
    if (plan.target_below_top) {
        SBEM_WARN(
            "rough box mesh: box_mesh_size {} m is below the top-face spacing {} m (the walls "
            "are never finer than the top face); using the uniform box",
            plan.target, plan.h);
    }
    if (plan.levels < plan.levels_unreduced) {
        SBEM_WARN(
            "rough box mesh: the rim is too rough for {} coarsening levels (a 2:1 transition "
            "cell would be inverted or a sliver); using {} levels{}",
            plan.levels_unreduced, plan.levels, plan.levels == 0 ? " (uniform box)" : "");
    }
}

/// Graded box (M >= 1); see the file comment for the layout. The caller has validated
/// the input (make_box_plan()).
std::pair<Vertices, Triangles> graded_box_arrays(const HeightMap& h, Real depth,
                                                 const BoxPlan& plan) {
    const Index nx = h.z.rows();
    const Index ny = h.z.cols();
    const Index m_top = plan.levels;
    const Index n_rows = plan.rows();  // K: row 0 = rim, row K = bottom-plate boundary
    const Index ncx = plan.cells_x(m_top);
    const Index ncy = plan.cells_y(m_top);
    const Index n_grid = nx * ny;
    const Index n_bottom = (ncx + 1) * (ncy + 1);

    std::vector<std::vector<GridNode>> rings;
    for (Index m = 0; m <= m_top; ++m) {
        const auto ms = static_cast<std::size_t>(m);
        rings.push_back(level_ring(plan.px[ms], plan.py[ms]));
    }
    const auto ring_of_row = [&](Index k) -> const std::vector<GridNode>& {
        const Index level = plan.row_level[static_cast<std::size_t>(k)];
        return rings[static_cast<std::size_t>(level)];
    };

    // First vertex of every interior wall row; vertex and triangle counts.
    std::vector<Index> row_offset(static_cast<std::size_t>(n_rows + 1), -1);
    Index nv = n_grid + n_bottom;
    for (Index k = 1; k < n_rows; ++k) {
        row_offset[static_cast<std::size_t>(k)] = nv;
        nv += static_cast<Index>(ring_of_row(k).size());
    }
    Index nf = 2 * (nx - 1) * (ny - 1) + 2 * ncx * ncy;
    for (Index k = 0; k < n_rows; ++k) {
        nf += static_cast<Index>(ring_of_row(k).size() + ring_of_row(k + 1).size());
    }

    // Bottom-plate index of a level-M grid line.
    const std::vector<Index>& bx = plan.px.back();
    const std::vector<Index>& by = plan.py.back();
    std::vector<Index> coarse_x(static_cast<std::size_t>(nx), -1);
    std::vector<Index> coarse_y(static_cast<std::size_t>(ny), -1);
    for (std::size_t a = 0; a < bx.size(); ++a) {
        coarse_x[static_cast<std::size_t>(bx[a])] = static_cast<Index>(a);
    }
    for (std::size_t b = 0; b < by.size(); ++b) {
        coarse_y[static_cast<std::size_t>(by[b])] = static_cast<Index>(b);
    }

    const Real x0 = -0.5 * static_cast<Real>(nx - 1) * h.dx;
    const Real y0 = -0.5 * static_cast<Real>(ny - 1) * h.dy;
    const auto xi = [&](Index i) { return x0 + static_cast<Real>(i) * h.dx; };
    const auto yj = [&](Index j) { return y0 + static_cast<Real>(j) * h.dy; };

    Vertices v(nv, 3);
    const auto set_vertex = [&v](Index k, Real x, Real y, Real z) {
        v(k, 0) = x;
        v(k, 1) = y;
        v(k, 2) = z;
    };
    for (Index i = 0; i < nx; ++i) {
        for (Index j = 0; j < ny; ++j) {
            const Index k = i * ny + j;
            set_vertex(k, xi(i), yj(j), h.z(i, j));
        }
    }
    for (Index a = 0; a <= ncx; ++a) {
        const Index i = bx[static_cast<std::size_t>(a)];
        for (Index b = 0; b <= ncy; ++b) {
            const Index j = by[static_cast<std::size_t>(b)];
            const Index k = n_grid + a * (ncy + 1) + b;
            set_vertex(k, xi(i), yj(j), depth);
        }
    }
    for (Index k = 1; k < n_rows; ++k) {
        const std::vector<GridNode>& ring = ring_of_row(k);
        const Real t = plan.row_fraction[static_cast<std::size_t>(k)];
        const Index base = row_offset[static_cast<std::size_t>(k)];
        for (std::size_t p = 0; p < ring.size(); ++p) {
            const auto [i, j] = ring[p];
            const Real z_rim = h.z(i, j);
            const Index idx = base + static_cast<Index>(p);
            set_vertex(idx, xi(i), yj(j), z_rim + (depth - z_rim) * t);
        }
    }

    // Vertex of wall row k at ring position p (row 0: top face, row K: bottom plate).
    const auto wall_vertex = [&](Index k, Index p) -> Index {
        const auto [i, j] = ring_of_row(k)[static_cast<std::size_t>(p)];
        if (k == 0)
            return i * ny + j;
        if (k == n_rows) {
            const Index a = coarse_x[static_cast<std::size_t>(i)];
            const Index b = coarse_y[static_cast<std::size_t>(j)];
            return n_grid + a * (ncy + 1) + b;
        }
        return row_offset[static_cast<std::size_t>(k)] + p;
    };

    // GCC 13 -O3 vectoriser workaround (see uniform_box_arrays()): every vertex index is
    // computed into a local first and emit() only receives plain locals (no index
    // arithmetic and no helper calls inside the call arguments).
    Triangles f(nf, 3);
    Index t = 0;
    const auto emit = [&f, &t](Index a, Index b, Index c) {
        f(t, 0) = a;
        f(t, 1) = b;
        f(t, 2) = c;
        ++t;
    };
    // Top face: as in uniform_box_arrays(), counter-clockwise seen from z < 0 (n_z < 0).
    for (Index i = 0; i + 1 < nx; ++i) {
        for (Index j = 0; j + 1 < ny; ++j) {
            const Index a = i * ny + j;
            const Index b = a + ny;
            const Index c = b + 1;
            const Index d = a + 1;
            emit(a, c, b);
            emit(a, d, c);
        }
    }
    // Walls, strip by strip. The rings are walked counter-clockwise seen from +z, so the
    // quad (r0, r1, s1, s0) with s below r is counter-clockwise seen from outside. A 2:1
    // cell with upper nodes r0, r1, r2 and lower nodes s0, s1 gives (r0, r1, s0),
    // (r1, r2, s1), (r1, s1, s0), all counter-clockwise seen from outside. The lower ring
    // is a subset of the upper one and both start at the corner (0, 0).
    for (Index k = 0; k < n_rows; ++k) {
        const Index k1 = k + 1;
        const std::vector<GridNode>& upper = ring_of_row(k);
        const std::vector<GridNode>& lower = ring_of_row(k1);
        const auto n_up = static_cast<Index>(upper.size());
        const auto n_lo = static_cast<Index>(lower.size());
        Index u = 0;
        for (Index s = 0; s < n_lo && u < n_up; ++s) {
            const Index sn = (s + 1) % n_lo;
            const Index un = (u + 1) % n_up;
            const GridNode& lower_next = lower[static_cast<std::size_t>(sn)];
            if (upper[static_cast<std::size_t>(u)] != lower[static_cast<std::size_t>(s)]) {
                throw std::logic_error("graded_box_arrays: wall rings are not nested");
            }
            const Index r0 = wall_vertex(k, u);
            const Index r1 = wall_vertex(k, un);
            const Index s0 = wall_vertex(k1, s);
            const Index s1 = wall_vertex(k1, sn);
            if (upper[static_cast<std::size_t>(un)] == lower_next) {
                emit(r0, r1, s1);
                emit(r0, s1, s0);
                u += 1;
            } else {
                const Index un2 = (u + 2) % n_up;
                if (upper[static_cast<std::size_t>(un2)] != lower_next) {
                    throw std::logic_error("graded_box_arrays: transition cell is not 2:1");
                }
                const Index r2 = wall_vertex(k, un2);
                emit(r0, r1, s0);
                emit(r1, r2, s1);
                emit(r1, s1, s0);
                u += 2;
            }
        }
        if (u != n_up) {
            throw std::logic_error("graded_box_arrays: wall strip does not close");
        }
    }
    // Bottom plate (grid of the level-M nodes): counter-clockwise seen from +z, n_z > 0.
    for (Index a = 0; a < ncx; ++a) {
        for (Index b = 0; b < ncy; ++b) {
            const Index c00 = n_grid + a * (ncy + 1) + b;
            const Index c10 = c00 + (ncy + 1);
            const Index c11 = c10 + 1;
            const Index c01 = c00 + 1;
            emit(c00, c10, c11);
            emit(c00, c11, c01);
        }
    }
    if (t != nf) {
        throw std::logic_error("graded_box_arrays: triangle count mismatch");
    }
    return {std::move(v), std::move(f)};
}

std::pair<Vertices, Triangles> box_arrays(const HeightMap& h, Real depth, const BoxPlan& plan) {
    if (plan.levels == 0)
        return uniform_box_arrays(h, depth);
    return graded_box_arrays(h, depth, plan);
}

/// Builds the TriangleMesh and logs one line with the top / wall / bottom counts.
TriangleMesh box_mesh(const HeightMap& h, Real depth, std::optional<Real> box_mesh_size) {
    const BoxPlan plan = make_box_plan(h, depth, box_mesh_size);
    warn_plan(plan);
    auto [vertices, triangles] = box_arrays(h, depth, plan);
    const Index ncx = plan.cells_x(plan.levels);
    const Index ncy = plan.cells_y(plan.levels);
    const Index n_top = 2 * (h.z.rows() - 1) * (h.z.cols() - 1);
    const Index n_bottom = 2 * ncx * ncy;
    const Index n_wall = triangles.rows() - n_top - n_bottom;
    SBEM_INFO(
        "rough box mesh: {} x {} grid, {} top + {} wall + {} bottom triangles (closing box / "
        "top = {:.3f}), depth {} m, {} coarsening levels, bottom {} x {} cells (target "
        "spacing {} m)",
        h.z.rows(), h.z.cols(), n_top, n_wall, n_bottom,
        static_cast<Real>(n_wall + n_bottom) / static_cast<Real>(n_top), depth, plan.levels, ncx,
        ncy, plan.target);
    return {std::move(vertices), std::move(triangles)};
}

std::string grid_label(const char* what, Index nx, Index ny, Real depth) {
    std::ostringstream os;
    os << what << ": " << nx << " x " << ny << " grid, box depth " << depth << " m";
    return os.str();
}

}  // namespace

namespace detail {

BoxGrading box_grading(const HeightMap& h, Real depth, std::optional<Real> box_mesh_size) {
    const BoxPlan plan = make_box_plan(h, depth, box_mesh_size);
    BoxGrading g;
    g.levels = plan.levels;
    g.levels_unreduced = plan.levels_unreduced;
    g.coarse_cells_x = plan.cells_x(plan.levels);
    g.coarse_cells_y = plan.cells_y(plan.levels);
    g.target_spacing = plan.target;
    for (const Index level : plan.row_level) g.row_ring_sizes.push_back(plan.ring_size(level));
    return g;
}

std::pair<Vertices, Triangles> rough_box_arrays(const HeightMap& h, Real depth,
                                                std::optional<Real> box_mesh_size) {
    return box_arrays(h, depth, make_box_plan(h, depth, box_mesh_size));
}

}  // namespace detail

Real HeightMap::rms() const {
    if (z.size() == 0) {
        throw std::logic_error("HeightMap::rms: empty height map");
    }
    const Real mean = z.mean();
    return std::sqrt((z.array() - mean).square().mean());
}

Real HeightMap::estimated_correlation_length() const {
    if (z.rows() < 2 || z.cols() < 2) {
        throw std::runtime_error(
            "HeightMap::estimated_correlation_length: map must be at least 2 x 2");
    }
    const MatrixXr d = z.array() - z.mean();
    const Real c0 = d.squaredNorm() / static_cast<Real>(d.size());
    if (!(c0 > 0.0)) {
        throw std::runtime_error("HeightMap::estimated_correlation_length: flat height map");
    }
    const Real lx = first_crossing_along_rows(d, c0) * dx;
    const MatrixXr dt = d.transpose();
    const Real ly = first_crossing_along_rows(dt, c0) * dy;
    return 0.5 * (lx + ly);
}

HeightMap generate_gaussian_height_map(const RoughSurfaceParams& p) {
    const Index n = grid_points(p);
    const Real d = p.edge_length_L / static_cast<Real>(n - 1);
    const Real lc = p.correlation_length;
    const Index r =
        static_cast<Index>(std::floor(kKernelRadiusInLc * lc / d * (1.0 + kRoundingSlack)));
    const std::uint64_t seed = effective_seed(p.seed);
    if (p.seed == 0) {
        SBEM_INFO(
            "generate_gaussian_height_map: seed 0 requested, drew seed {} from "
            "std::random_device (pass it as RoughSurfaceParams::seed to reproduce this map)",
            seed);
    }

    const MatrixXr w = gaussian_kernel(lc, d, d, r, r);
    const MatrixXr noise = gaussian_noise(n + 2 * r, n + 2 * r, seed);
    HeightMap h;
    h.dx = d;
    h.dy = d;
    h.z = p.use_fft ? convolve_valid_fft(noise, w) : convolve_valid_direct(noise, w);
    h.z *= p.rms_roughness;
    SBEM_DEBUG(
        "generate_gaussian_height_map: {} x {} grid, dx {} m, kernel radius {} cells, seed {}, "
        "{} convolution",
        n, n, d, r, seed, p.use_fft ? "FFT" : "direct");
    return h;
}

TriangleMesh make_mesh_from_height_map(const HeightMap& h, std::optional<Real> box_depth,
                                       std::optional<Real> box_mesh_size) {
    const Real depth = default_or(box_depth);
    const ScopedTimer timer(grid_label("make_mesh_from_height_map", h.z.rows(), h.z.cols(), depth));
    return box_mesh(h, depth, box_mesh_size);
}

TriangleMesh make_rough_surface_mesh(const RoughSurfaceParams& p) {
    const Index n = grid_points(p);
    const Real depth = default_or(p.box_depth);
    const ScopedTimer timer(grid_label("make_rough_surface_mesh", n, n, depth));
    return box_mesh(generate_gaussian_height_map(p), depth, p.box_mesh_size);
}

}  // namespace specklebem::geometry
