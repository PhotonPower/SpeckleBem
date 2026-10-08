#pragma once
/// @file rough_surface.hpp
/// Gaussian random rough surfaces (Eqs. 8-9 of Fu et al. 2023):
///
///   xi_u(x,y) = sigma * f(x,y),   f ~ N(0,1) i.i.d. on the grid
///   xi(x,y)   = xi_u (*) W,       W(x,y) = exp(-2 (x^2 + y^2) / Lc^2)
///
/// The surface is a height map z = xi(x,y) on an L x L patch, meshed with
/// structured triangles. Light arrives from z < 0 and travels in +z; the
/// object (R2) is the half space below the rough interface. The mesh is closed
/// with side walls and a bottom plate far enough away that the Gaussian-beam
/// illumination does not reach the artificial boundary (see
/// docs/02_architecture.md, "Truncated surfaces").

#include "specklebem/geometry/mesh.hpp"

#include <cstdint>
#include <optional>

namespace specklebem::geometry {

struct RoughSurfaceParams {
    Real edge_length_L = 10e-6;        ///< patch size L x L [m]
    Real rms_roughness = 50e-9;        ///< sigma [m]
    Real correlation_length = 500e-9;  ///< Lc [m]
    Real mesh_size = 50e-9;            ///< target triangle edge [m] (lambda/10 at 500 nm)
    std::uint64_t seed = 0;            ///< RNG seed (0 => non-deterministic)
    std::optional<Real> box_depth;     ///< depth of the closing box; automatic if unset
    bool use_fft = true;               ///< FFT convolution (fast) vs direct (reference)
};

/// Height map on a regular grid, plus spacing.
struct HeightMap {
    MatrixXr z;  ///< heights [m], z(i,j) at (x_i, y_j)
    Real dx{};
    Real dy{};
    [[nodiscard]] Real rms() const;
    [[nodiscard]] Real estimated_correlation_length() const;
};

HeightMap generate_gaussian_height_map(const RoughSurfaceParams& p);

/// Closed mesh: rough top, flat side walls and bottom plate.
TriangleMesh make_rough_surface_mesh(const RoughSurfaceParams& p);

/// Mesh from an externally supplied height map (e.g. measured AFM / WLI data).
TriangleMesh make_mesh_from_height_map(const HeightMap& h, std::optional<Real> box_depth = {});

}  // namespace specklebem::geometry
