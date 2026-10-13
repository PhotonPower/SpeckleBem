#pragma once
/// @file open_surface.hpp
/// Open rough surface terminated by a PML (ADR 0009, decision 1): an alternative to the
/// closed box of make_mesh_from_height_map() (ADR 0006), which stays unchanged.
///
/// Zones, by max(|x|, |y|) of the triangle centroid (concentric square bands):
///
///   Patch  max(|x|, |y|) <= a_0 = L/2              full roughness xi(x, y)
///   Taper  a_0 < .. <= a_1 = a_0 + W_t             xi w(x) w(y), raised-cosine window
///   Gap    a_1 < .. <= a_2 = a_1 + W_g             flat (z = 0), unstretched
///   Pml    a_2 < .. <= a_3 = a_2 + W_pml           flat (z = 0), stretched by
///                                                  CoordinateStretch (make_pml_stretch())
///
/// The outer square max(|x|, |y|) = a_3 is an open edge: its edges are the boundary edges of
/// the TriangleMesh and carry no RWG function (zero normal current, ADR 0009 decision 1).
/// The physical model is a rough spot in a polished (flat) surface of the same materials with
/// absorbing surroundings (ADR 0009, lead decision 1).
///
/// Orientation (docs/06_conventions.md): the object R2 occupies z > xi, light comes from
/// z < 0, so the normal points out of R2 into R1, i.e. n_z < 0 everywhere, and every triangle
/// is counter-clockwise seen from below (R1). TriangleMesh orients only closed components, so
/// the generator emits this orientation itself.

#include "specklebem/geometry/coordinate_stretch.hpp"
#include "specklebem/geometry/mesh.hpp"
#include "specklebem/geometry/rough_surface.hpp"

#include <cstdint>
#include <vector>

namespace specklebem::geometry {

/// Zone of a triangle of the open surface (see open_surface.hpp).
enum class SurfaceZone : std::uint8_t { Patch, Taper, Gap, Pml };

/// Requested zone sizes [m]; make_open_surface_mesh() rounds them to the grid.
struct OpenSurfaceParams {
    Real patch_size{};   ///< L > 0: full roughness on |x|, |y| <= L/2 [m]
    Real taper_width{};  ///< W_t >= map spacing: raised-cosine taper of xi to 0 [m]; ADR 0009 asks
                         ///< W_t >= Lc, which the caller (Simulation, WP-PML3) must enforce
    Real gap_width = 0.0;  ///< W_g >= 0: flat, unstretched band between taper and PML [m]
    Real pml_width{};      ///< W_pml > 0 [m]
};

/// Open, oriented surface mesh with per-triangle zones. All half widths are the rounded
/// values actually meshed (grid lines of the mesh).
struct OpenSurfaceMesh {
    TriangleMesh mesh;              ///< open, oriented: normal out of R2 (z > xi) into R1, n_z < 0
    std::vector<SurfaceZone> zone;  ///< per triangle, by centroid
    Real patch_half{};              ///< a_0 (L/2 after rounding)
    Real taper_end{};               ///< a_1 = a_0 + W_t (after rounding)
    Real pml_start{};               ///< a_2 = a_1 + W_g (after rounding)
    Real outer_half{};              ///< a_3 = a_2 + W_pml (truncation edge)
};

/// Raised-cosine taper window of make_open_surface_mesh(): 1 for |t| <= patch_half,
/// 0.5 (1 + cos(pi (|t| - patch_half) / taper_width)) on the taper, 0 for
/// |t| >= patch_half + taper_width. C^1 (value and slope continuous at both taper ends).
/// @throws std::invalid_argument for a non-finite argument, patch_half < 0 or
///         taper_width <= 0.
[[nodiscard]] Real taper_window(Real t, Real patch_half, Real taper_width);

/// Open rough-surface mesh with taper, gap and PML zones (ADR 0009, decision 1).
///
/// Surface: z = xi(x, y) w(x) w(y) with w = taper_window(., a_0, W_t), i.e. the full map on
/// the patch, tapered to 0 at a_1 (C^1) and exactly 0 (flat at z = 0) beyond, in the gap and
/// in the PML.
///
/// Grid. The mesh continues the grid of the map with the same spacing h = dx = dy out to
/// a_3: vertices at x_i = (2 i - M) h / 2, i = 0 .. M, M = 2 a_3 / h (likewise in y), which
/// coincide with the map nodes x_i = -(n_x - 1) dx / 2 + i dx inside the map. Every cell is
/// split along the diagonal (i, j)-(i+1, j+1) into the triangles (a, c, b) and (a, d, c)
/// (a = (i, j), b = (i+1, j), c = (i+1, j+1), d = (i, j+1)), as on the top face of the closed
/// box, so the patch is triangulated like the box top face of the same map.
///
/// Rounding. The band boundaries are grid lines, so every cell lies in exactly one zone:
///  * a_0 is L/2 rounded up to the next grid line of the map (lines at map_half - k h, k
///    integer, with map_half = (n_x - 1) dx / 2);
///  * W_t, W_g and W_pml are rounded up to integer multiples of h.
/// Values within 1e-6 h of a grid line or multiple count as exact. One SBEM_INFO per mesh
/// reports a change by the rounding. The bands are squares (the same a_k along x and y), so
/// n_x - 1 and n_y - 1 must have the same parity (common grid lines); the map itself may be
/// rectangular. A map from generate_gaussian_height_map() with edge_length_L = L + 2 W_t and
/// a mesh_size dividing L / 2 and W_t meets all of this without rounding.
///
/// Coverage. The map must cover the patch and the taper, |x|, |y| <= a_1 (it may be larger;
/// heights at |x| or |y| >= a_1 are ignored and need not be finite).
///
/// Zones: by the centroid of each triangle (see open_surface.hpp); every cell of a band has
/// both of its triangles in that band, so the projected zone areas are exactly
/// (2 a_k)^2 - (2 a_(k-1))^2.
///
/// patch_size is rounded up to the next grid line (at least one cell, two for an even grid).
/// @throws std::invalid_argument for a non-finite or non-positive spacing, dx != dy (relative
///         1e-9), a non-finite parameter, L <= 0, W_pml <= 0, W_g < 0, W_t < h (taper
///         narrower than one cell), a map that does not
///         cover |x|, |y| <= a_1, n_x - 1 and n_y - 1 of different parity, or a non-finite
///         height inside |x|, |y| < a_1.
OpenSurfaceMesh make_open_surface_mesh(const HeightMap& h, const OpenSurfaceParams& p);

/// The PML stretch of an open surface: the same profile on x and y with
/// start = m.pml_start, width = m.outer_half - m.pml_start (the rounded W_pml). order < 1 gives a
/// sigma that is discontinuous (m = 0) or has an unbounded slope (0 < m < 1) at the PML start.
/// @throws std::invalid_argument as CoordinateStretch (sigma_max < 0, order < 0, non-finite).
[[nodiscard]] CoordinateStretch make_pml_stretch(const OpenSurfaceMesh& m, Real sigma_max,
                                                 Real order = 2.0);

}  // namespace specklebem::geometry
