#pragma once
/// @file sphere.hpp
/// Reference geometry: sphere meshes for validation against Mie theory.
#include "specklebem/geometry/mesh.hpp"

namespace specklebem::geometry {

/// Icosphere refined `subdivisions` times (20 * 4^n triangles).
/// The paper used 131,072 triangles (n = 6) for a 4 um Ag sphere at 500 nm
/// (mesh size ~ lambda/27), giving 2N = 393,216 unknowns.
TriangleMesh make_icosphere(Real radius, int subdivisions, const Vec3& center = Vec3::Zero());

/// Sphere meshed to a target edge length (chooses subdivisions automatically).
TriangleMesh make_sphere(Real radius, Real target_edge_length, const Vec3& center = Vec3::Zero());

}  // namespace specklebem::geometry
