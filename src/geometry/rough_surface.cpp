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
/// Box mesh: see make_mesh_from_height_map() in rough_surface.hpp. Vertex layout:
/// top grid (n_x n_y), bottom grid (n_x n_y), then the interior wall vertices
/// (n_rim (n_z - 1)), where n_rim = 2 (n_x - 1) + 2 (n_y - 1). Triangle count:
/// 4 (n_x - 1)(n_y - 1) + 2 n_rim n_z.
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

/// Builds the TriangleMesh; logging is done by the callers.
TriangleMesh box_mesh(const HeightMap& h, Real depth) {
    auto [vertices, triangles] = detail::rough_box_arrays(h, depth);
    return {std::move(vertices), std::move(triangles)};
}

std::string grid_label(const char* what, Index nx, Index ny, Index nf, Real depth) {
    std::ostringstream os;
    os << what << ": " << nx << " x " << ny << " grid, " << nf << " triangles, box depth " << depth
       << " m";
    return os.str();
}

}  // namespace

namespace detail {

std::pair<Vertices, Triangles> rough_box_arrays(const HeightMap& h, Real depth) {
    validate_height_map(h, depth);
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
        throw std::logic_error("rough_box_arrays: triangle count mismatch");
    }
    return {std::move(v), std::move(f)};
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

TriangleMesh make_mesh_from_height_map(const HeightMap& h, std::optional<Real> box_depth) {
    const Real depth = default_or(box_depth);
    validate_height_map(h, depth);
    const Index nz = wall_rows(depth, std::min(h.dx, h.dy));
    const ScopedTimer timer(grid_label("make_mesh_from_height_map", h.z.rows(), h.z.cols(),
                                       box_triangle_count(h.z.rows(), h.z.cols(), nz), depth));
    return box_mesh(h, depth);
}

TriangleMesh make_rough_surface_mesh(const RoughSurfaceParams& p) {
    const Index n = grid_points(p);
    const Real depth = default_or(p.box_depth);
    const Real d = p.edge_length_L / static_cast<Real>(n - 1);
    const Index nz = wall_rows(depth, d);
    const ScopedTimer timer(
        grid_label("make_rough_surface_mesh", n, n, box_triangle_count(n, n, nz), depth));
    return box_mesh(generate_gaussian_height_map(p), depth);
}

}  // namespace specklebem::geometry
