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
#include <vector>

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
    /// Target coarse spacing [m] of the lower side walls and the bottom plate of the closing
    /// box (graded box, see make_mesh_from_height_map()). If unset, the automatic rule
    ///   box_mesh_size = min(depth / 2, L / 8, 10 * mesh_size)
    /// is used. A value equal to the top-face spacing gives the uniform box (walls and bottom
    /// at the top-face spacing, the WP2 mesh).
    ///
    /// Physics contract (decay-based). The coarse part does not resolve the field; it is
    /// valid only where the field transmitted into the object has decayed, i.e. below a
    /// depth of a few field decay lengths
    ///   delta = lambda_0 / (2 pi |Im n_2|)
    /// under the rough interface. Without a fine band the first coarsened wall row lies
    /// (R + 2) h_g + e below the rim (h_g = sqrt(dx dy); R = 0 relaxation rows and e = 0 for
    /// a smooth rim, i.e. 100 nm at 50 nm spacing; e <= the local rim deviation from its
    /// level-M interpolation) and the full coarse spacing is reached 2^M h_g below it.
    ///  * The automatic rule without fine band is meant for strongly absorbing objects: Ag
    ///    at 500 nm has delta = 25 nm, so the first coarse row (100 nm) is ~4 delta deep.
    ///  * Weakly absorbing objects need a fine band down to ~3 delta (box_fine_depth): Si at
    ///    500 nm has delta = 1.13 um (material::field_decay_length()). The uniform box
    ///    (box_mesh_size = mesh_size) remains valid but costs more.
    ///  * box_mesh_size must be >= the top-face spacing: the walls are never finer than the
    ///    top face, and the wavelength inside the object, lambda_0 / |n_2|, is resolved by
    ///    the top-face mesh size, not by the box. A value below h / sqrt(2) (h = min(dx, dy))
    ///    is accepted with an SBEM_WARN and gives the uniform box.
    /// The generator knows no material: Simulation (later phase) is responsible for passing
    /// the fine band (box_fine_depth = 3 delta) for weakly absorbing objects (ADR 0006).
    std::optional<Real> box_mesh_size;
    /// Fine band of the graded box [m]: the side walls keep the top-face spacing from the
    /// rim down to z = box_fine_depth (measured from the mean plane z = 0 in +z) and grade
    /// only below. Unset: no fine band (the grading starts right below the rim). Must be
    /// positive, finite and below the box depth. Size it from the object material:
    /// box_fine_depth = 3 delta with delta = material::field_decay_length(eps_r, lambda_0)
    /// (Si at 500 nm: delta = 1.13 um). See make_mesh_from_height_map().
    std::optional<Real> box_fine_depth;
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
/// make_mesh_from_height_map(generate_gaussian_height_map(p), p.box_depth, p.box_mesh_size,
/// p.box_fine_depth).
TriangleMesh make_rough_surface_mesh(const RoughSurfaceParams& p);

/// Mesh from an externally supplied height map (e.g. measured AFM / WLI data).
///
/// Closed box: the top face is the structured triangulation of z = xi on the grid (two
/// triangles per cell), vertical side walls run from the rim down to the flat bottom plate
/// at z = +depth (the object occupies z > xi). Rim vertices are shared, so the mesh is
/// closed and edge-manifold. Triangles are emitted counter-clockwise seen from outside the
/// box (normals out of R2 into R1: top n_z < 0, bottom n_z > 0, walls outward), so the
/// TriangleMesh constructor has nothing to repair. The default depth is 2e-6 m (see
/// RoughSurfaceParams::box_depth).
///
/// Grading (box_mesh_size, the target coarse spacing h_c of the lower walls and the bottom):
///  * Physics: valid only below a few field decay lengths of the object material; see the
///    contract at RoughSurfaceParams::box_mesh_size (automatic rule for strongly absorbing
///    objects such as Ag; with box_fine_depth = 3 delta for weakly absorbing ones such as Si).
///  * Automatic rule (unset): h_c = min(depth / 2, L / 8, 10 h), with h = min(dx, dy) the
///    top-face spacing and L = min((n_x - 1) dx, (n_y - 1) dy) the shorter patch side.
///    A requested h_c < h / sqrt(2) logs SBEM_WARN (the walls are never finer than the top
///    face) and gives the uniform box.
///  * Anisotropic maps (dx != dy): the level count uses h_b = max(dx, dy), so the coarse
///    spacing does not exceed h_c along the coarser axis; the wall row heights use
///    h_g = sqrt(dx dy), which balances the cell shapes on the walls along x and along y.
///    For max / min(dx, dy) <= 4 the box triangles keep an aspect ratio R / (2 r) <= 4
///    (dx / dy = 4: 2.65 on a flat map, 3.8 with +-20 nm rim heights; tested); beyond that
///    they degrade like the top face and the uniform box (1:rho cells). For dx = dy all
///    three spacings coincide.
///  * Coarsening levels: M = round(log2(h_c / h_b)) (nearest power-of-two ratio; 0 for
///    h_c < sqrt(2) h_b), limited by 2^M <= min(n_x - 1, n_y - 1) and by the depth: the
///    graded part needs a wall height H = depth - mean(rim heights) >= 1.5 * 2^M h_g and,
///    with a fine band, depth - z_f >= (1.5 * 2^M - 1) h_g (transition rows and half a
///    coarse row below z_f), otherwise M is reduced. M = 0 (in particular h_c = h,
///    depth < 3 h_g, or a fine band reaching to within 2 h_g of the bottom plate) gives the
///    uniform box: n_z = max(1, ceil(depth / h)) wall rows, walls and bottom at the
///    top-face spacing, bit-identical to the WP2 mesh.
///  * Coarse grid per axis: n_c = ceil((n - 1) / 2^M) coarse cells for n - 1 top-face
///    cells; coarse cell k spans the fine cells floor(k (n - 1) / n_c) ..
///    floor((k + 1) (n - 1) / n_c), i.e. 2^M cells when 2^M divides n - 1 and otherwise
///    floor or ceil of (n - 1) / n_c cells (the coarse spacing is recomputed accordingly).
///    The intermediate levels halve each coarse cell recursively (larger half first), so
///    every level-(m+1) cell consists of one or two level-m cells.
///  * Walls, from the rim down (WP2c; sketch for M = 3, R = 1, no fine band):
///
///        rim     z_rim (rough, level 0)          ___ top face
///        R rows  level 0, z_rim -> z_S           | | | | | | | | |  quads, vertical edges
///        anchor  z_S (level 0)                   |-+-+-+-+-+-+-+-|
///        [fine band: level-0 rows <= h_g tall down to z >= z_f]
///        M transition rows, h_g .. 2^(M-1) h_g   |/\|/\|/\|/\|    2:1 cells (3 triangles)
///        coarse rows <= 2^M h_g                  |   |   |   |   |
///        bottom plate z = depth (level M)        +---+---+---+---+
///
///    The anchor row z_S is piecewise linear along the rim between the level-M nodes:
///    z_S = z~ + D + (R + 1) h_g, with z~ the linear interpolation of the rim heights between
///    the level-M nodes, and D the linear interpolation of node offsets D(a) = the largest
///    rim excess z_rim - z~ (>= 0) over the two level-M cells at node a. Hence z_S lies at
///    least (R + 1) h_g below the rim in every column. The R level-0 relaxation rows
///    interpolate each column linearly between z_rim and z_S; their strips are quads with
///    vertical edges, which cannot invert (every column is monotone). R = 0 when the excess
///    e = z_S - z_rim - (R + 1) h_g stays <= 2 h_g (strip heights <= 3 h_g), otherwise
///    R = 1 (cap; one level-0 row costs 2 % of the top-face triangles at L = 10 um, 50 nm).
///    All rows below follow z = z_S + (depth - z_S) tau_k, tau_k increasing from 0 to 1: the
///    fine band (if any), M transition rows of nominal heights h_g, 2 h_g, ...,
///    2^(M-1) h_g (in which every level-(m+1) cell made of two level-m cells is a 2:1 strip
///    of three triangles; a single cell, which only occurs when 2^M does not divide n - 1,
///    gives two triangles), then uniform coarse rows of nominal height <= 2^M h_g. Nominal
///    heights refer to the mean column depth - z_S; each column scales them by its own
///    depth - z_S. A smooth rim (z_rim = z~, e.g. a flat or piecewise-linear rim) gives
///    R = 0 and the WP2b rows (rim, one row at h_g, transitions, coarse rows).
///  * Fine band (box_fine_depth = z_f): level-0 rows of height <= h_g in every column from
///    the anchor row down to a row lying at z >= z_f in every column; the transition rows
///    start there, so all wall triangles above z_f have the top-face spacing (horizontal
///    edges h, rows <= h_g; rows above the anchor as described above). A z_f above the
///    anchor row adds no rows. The bottom plate is unaffected unless the fine band reaches
///    it: if the transitions and half a coarse row do not fit below z_f, M is reduced, down
///    to the uniform box. z_f >= depth throws (the uniform box is box_mesh_size = mesh_size).
///  * Rough rims (WP2c): every row below the rim is linear within each level-M cell, so a
///    2:1 cell is a shear of the flat cell and cannot invert, for any roughness. Rim
///    roughness costs at most one relaxation row and deeper anchor rows. Measured on
///    generated maps (L = 10 um, h = 50 nm, depth 2 um, automatic M = 3, seeds 1-5;
///    closing box / top-face triangles): sigma = 50 nm, Lc = 500 nm: M = 3, R = 0 (7.2 %);
///    sigma = 50 nm, Lc = 100 nm or 200 nm, and sigma = 250 nm, Lc = 100 nm or 500 nm: M = 3,
///    R = 1 (8.9-9.2 %). Under WP2b (rows following the rim) these maps fell back to M = 0-2
///    (up to 180 %). The wall aspect ratios of very rough rims are set by the rim itself
///    (rim steps sheared over one h-wide column, as in the uniform WP2 box); the anchor row
///    of a steep rim also shears the 2:1 cells (graded walls up to 1.7 x the worst uniform
///    wall triangle for sigma = 250 nm, Lc = 500 nm; below it for Lc = 100 nm).
///    Safety net, per M: every column strictly monotone; the middle upper node of every 2:1
///    cell at least half its band height above the edge below it; the transitions and half
///    a coarse row fit below the anchor row (or the fine band) in the mean column; every
///    column keeps at least a quarter of the mean column height depth - z_S; and the
///    largest wall aspect ratio R / (2 r) is at most max(4, 2 x that of the uniform WP2 box
///    walls on the same map). Otherwise M is reduced by one until the rows fit, ultimately
///    down to the uniform box (one SBEM_WARN per mesh). With the construction above only an
///    anchor row close to the bottom plate (deep rim pits in a shallow box: L = 2 um,
///    sigma = 250 nm, Lc = 100 nm, depth 1 um gives the uniform box) or a too deep fine band
///    triggers it.
///  * The bottom plate is the structured grid of the level-M nodes, welded to the lowest
///    wall row.
/// @throws std::invalid_argument for a grid smaller than 2 x 2, non-positive spacing,
///         non-finite heights, depth <= max(xi) + max(dx, dy), a non-positive or
///         non-finite box_mesh_size, or a box_fine_depth that is non-positive, non-finite or
///         >= depth.
TriangleMesh make_mesh_from_height_map(const HeightMap& h, std::optional<Real> box_depth = {},
                                       std::optional<Real> box_mesh_size = {},
                                       std::optional<Real> box_fine_depth = {});

namespace detail {

/// Internal, exposed for tests; not part of the public API.
/// Resolved layout of the closing box (see make_mesh_from_height_map()).
struct BoxGrading {
    Index levels = 0;            ///< M, number of 2:1 coarsening levels (0 = uniform box)
    Index levels_unreduced = 0;  ///< M before the rough-rim reduction (>= levels)
    Index coarse_cells_x = 0;    ///< bottom-plate cells along x (n_x - 1 for M = 0)
    Index coarse_cells_y = 0;    ///< bottom-plate cells along y (n_y - 1 for M = 0)
    Real target_spacing = 0.0;   ///< requested or automatic h_c [m]
    /// R, the largest number of interior relaxation nodes in a wall column (between the rim
    /// and the anchor row; 0 for a smooth rim).
    Index relaxation_rows = 0;
    /// Interior relaxation nodes of all wall columns (vertices between rim and anchor row
    /// that belong to no wall row); the relaxation band adds twice as many triangles to the
    /// strip rim -> anchor row (2 x ring 0 for a smooth rim).
    Index relaxation_vertices = 0;
    Index fine_rows = 0;  ///< level-0 fine-band rows below the anchor row
    /// z of the anchor row [m] in every wall column (M >= 1, else empty), in the order of the
    /// rim walk: from grid node (0, 0) counter-clockwise seen from +z, i.e. (i, 0) for
    /// i = 0 .. n_x - 2, (n_x - 1, j) for j = 0 .. n_y - 2, (i, n_y - 1) for i = n_x - 1 .. 1,
    /// (0, j) for j = n_y - 1 .. 1.
    std::vector<Real> anchor_z;
    /// Largest aspect ratio R / (2 r) of the graded side walls (M >= 1; 0 for M = 0).
    Real wall_aspect = 0.0;
    /// Same for the side walls of the uniform WP2 box on the same map (the reference of the
    /// shape check; computed only when M >= 1 before the checks, else 0).
    Real uniform_wall_aspect = 0.0;
    /// Number of wall vertices in every wall row, from the rim (row 0, shared with the top
    /// face) over the anchor row (row 1 for M >= 1) to the lowest row (shared with the bottom
    /// plate); size = wall rows + 1. The relaxation nodes are not part of any row.
    std::vector<Index> row_ring_sizes;
};

/// Grading for the given map, depth and coarse spacing (unset: automatic rule). Same
/// validation and exceptions as make_mesh_from_height_map(). The detail:: functions log
/// nothing; the grading warnings are emitted once per mesh by make_mesh_from_height_map()
/// and make_rough_surface_mesh().
BoxGrading box_grading(const HeightMap& h, Real depth, std::optional<Real> box_mesh_size = {},
                       std::optional<Real> box_fine_depth = {});

/// Raw vertex / triangle arrays of the closed box mesh built by make_mesh_from_height_map()
/// with the given depth and coarse spacing (no TriangleMesh construction, so the
/// orientation emitted by the generator can be checked before any repair). Same
/// validation and exceptions as make_mesh_from_height_map().
std::pair<Vertices, Triangles> rough_box_arrays(const HeightMap& h, Real depth,
                                                std::optional<Real> box_mesh_size = {},
                                                std::optional<Real> box_fine_depth = {});

}  // namespace detail

}  // namespace specklebem::geometry
