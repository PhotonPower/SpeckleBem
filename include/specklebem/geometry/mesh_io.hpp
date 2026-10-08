#pragma once
/// @file mesh_io.hpp
/// Import / export of general surface meshes (STL, OBJ, Gmsh .msh). Entry
/// point for arbitrary geometries in later project phases.
#include "specklebem/geometry/mesh.hpp"

#include <string>

namespace specklebem::geometry::io {

TriangleMesh read_mesh(const std::string& path);  ///< format chosen from extension
void write_mesh(const TriangleMesh& m, const std::string& path);

TriangleMesh read_stl(const std::string& path);
TriangleMesh read_obj(const std::string& path);
TriangleMesh read_gmsh(const std::string& path);

}  // namespace specklebem::geometry::io
