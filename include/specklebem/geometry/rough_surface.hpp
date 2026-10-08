#pragma once
/// @file rough_surface.hpp
/// Gaussian random rough surfaces (Eqs. 8-9 of Fu et al. 2023):
///
///   xi_u(x,y) = sigma * f(x,y),   f ~ N(0,1) i.i.d. on the grid
///   xi(x,y)   = xi_u (*) W,       W(x,y) = exp(-2 (x^2 + y^2) / Lc^2)
///
/// The surface is a height map z = xi(x,y) on an L x L patch, meshed with
/// structured triangles. Light arrives from z < 0 and travels in +z; the
/// object (R2) is the half space below the rough interface (z > xi). The mesh is closed
/// with side walls and a bottom plate far enough away that the Gaussian-beam
/// illumination does not reach the artificial boundary (see
/// docs/02_architecture.md, "Truncated surfaces").

#include "specklebem/geometry/mesh.hpp"

#include <cstdint>
#include <optional>
#include <utility>

namespace specklebem::geometry {

struct RoughSurfaceParams {
    Real edge_length_L = 10e-6;  ///< patch size L x L [m]
    /// sigma [m]: standard deviation of the heights about the mean plane z = 0.
    Real rms_roughness = 50e-9;
    /// Lc [m]: 1/e radius of the normalised height autocorrelation,
    ///   C(r) / C(0) = exp(-r^2 / Lc^2),
    /// which is what the kernel W(x,y) = exp(-2 (x^2 + y^2) / Lc^2) of Eq. 9 produces.
    /// The discrete kernel is truncated at radius 4 Lc and normalised to
    /// sum_k W_k^2 = 1, so the output variance is exactly sigma^2.
    Real correlation_length = 500e-9;
    Real mesh_size = 50e-9;  ///< target grid spacing [m] (lambda/10 at 500 nm)
    std::uint64_t seed = 0;  ///< RNG seed (0 => non-deterministic, from std::random_device)
    /// Depth of the closing box below the mean plane: the bottom plate lies at z = +depth.
    /// If unset, the fixed minimum of ADR 0006 (2e-6 m) is used. The ADR's material-based
    /// rule (10 absorption lengths of the object material) cannot be evaluated here
    /// because the generator knows no material: Simulation (later phase) is responsible
    /// for passing a material-based depth. The validation suite checks the far field for
    /// a depth x2 sensitivity (docs/05_validation.md).
    std::optional<Real> box_depth;
    bool use_fft = true;  ///< FFT convolution (fast) vs direct (reference)
};

/// Height map on a regular grid, plus spacing.
///
/// Grid: z.rows() = n_x points along x, z.cols() = n_y points along y, centred at the
/// origin: x_i = -(n_x - 1) dx / 2 + i dx, y_j = -(n_y - 1) dy / 2 + j dy.
struct HeightMap {
    MatrixXr z;  ///< heights [m], z(i,j) at (x_i, y_j)
    Real dx{};
    Real dy{};
    /// Root mean square of the heights about their mean.
    /// @throws std::logic_error for an empty map.
    [[nodiscard]] Real rms() const;
    /// Estimated Lc: first 1/e crossing (linear interpolation) of the normalised,
    /// mean-subtracted autocorrelation along x (averaged over all columns j) and along y
    /// (averaged over all rows i); returns the mean of the two estimates.
    /// @throws std::runtime_error if there is no crossing within the map or the map is flat.
    [[nodiscard]] Real estimated_correlation_length() const;
};

/// Gaussian random height map (Eqs. 8-9). Grid: n = round(L / mesh_size) + 1 points per
/// axis, dx = dy = L / (n - 1). The white noise is drawn on a grid enlarged by the kernel
/// radius 4 Lc on every side and the linear convolution is cropped to the L x L patch,
/// so the statistics at the patch edges equal those in the centre.
/// Randomness: std::mt19937_64 -> 53-bit uniform in (0, 1) -> Box-Muller, so a fixed
/// seed gives the same map on every standard library.
/// Seed 0 draws a seed from std::random_device and logs it (SBEM_INFO) for reproduction.
/// @throws std::invalid_argument for non-positive / non-finite parameters, n < 2,
///         n > 2^20 points per axis, or correlation_length < 2 max(dx, dy) (under-resolved).
HeightMap generate_gaussian_height_map(const RoughSurfaceParams& p);

/// Closed mesh: rough top, flat side walls and bottom plate; equivalent to
/// make_mesh_from_height_map(generate_gaussian_height_map(p), p.box_depth).
TriangleMesh make_rough_surface_mesh(const RoughSurfaceParams& p);

/// Mesh from an externally supplied height map (e.g. measured AFM / WLI data).
///
/// Closed box: the top face is the structured triangulation of z = xi on the grid (two
/// triangles per cell), vertical side walls run from the rim down to the flat bottom plate
/// at z = +depth (the object occupies z > xi). Walls and bottom use the same spacing
/// (n_z = max(1, ceil(depth / min(dx, dy))) wall rows); rim vertices are shared, so the
/// mesh is closed and edge-manifold. Triangles are emitted counter-clockwise seen from
/// outside the box (normals out of R2 into R1: top n_z < 0, bottom n_z > 0, walls
/// outward). The default depth is 2e-6 m (see RoughSurfaceParams::box_depth).
/// @throws std::invalid_argument for a grid smaller than 2 x 2, non-positive spacing,
///         non-finite heights or depth <= max(xi) + max(dx, dy).
TriangleMesh make_mesh_from_height_map(const HeightMap& h, std::optional<Real> box_depth = {});

namespace detail {

/// Internal, exposed for tests; not part of the public API.
/// Raw vertex / triangle arrays of the closed box mesh built by make_mesh_from_height_map()
/// with the given depth (no TriangleMesh construction, so the orientation emitted by the
/// generator can be checked before any repair). Same validation and exceptions as
/// make_mesh_from_height_map().
std::pair<Vertices, Triangles> rough_box_arrays(const HeightMap& h, Real depth);

}  // namespace detail

}  // namespace specklebem::geometry
