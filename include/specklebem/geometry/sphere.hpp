#pragma once
/// @file sphere.hpp
/// Reference geometry: sphere meshes for validation against Mie theory.
#include "specklebem/geometry/mesh.hpp"

namespace specklebem::geometry {

/// Icosphere: regular icosahedron refined `subdivisions` times by midpoint subdivision,
/// vertices projected onto the sphere. 20 * 4^n triangles, 10 * 4^n + 2 vertices,
/// N = 30 * 4^n interior edges; triangles counter-clockwise seen from outside.
/// Mean edge length ~ 1.0515 r / 2^n for n = 0 and ~ 1.149 * 1.0515 r / 2^n for large n.
/// n = 6 gives 81,920 triangles (2N = 245,760 unknowns). The paper's 131,072 triangles
/// (4 um Ag sphere at 500 nm, mesh size ~ lambda/27, 2N = 393,216) equal 8 * 4^7, i.e.
/// an octahedron-based sphere, not an icosphere.
/// @throws std::invalid_argument for subdivisions outside [0, 10], radius <= 0 or a
///         non-finite radius / center.
TriangleMesh make_icosphere(Real radius, int subdivisions, const Vec3& center = Vec3::Zero());

/// Icosphere with the smallest number of subdivisions whose *mean* edge length is at
/// most `target_edge_length`. The maximum edge length is ~ 1.1 x the mean (1.06 at
/// n = 1, 1.095 for large n), the minimum ~ 0.92 x the mean.
/// @throws std::invalid_argument for a non-positive / non-finite radius or target, or a
///         target that needs more than 10 subdivisions.
TriangleMesh make_sphere(Real radius, Real target_edge_length, const Vec3& center = Vec3::Zero());

}  // namespace specklebem::geometry
