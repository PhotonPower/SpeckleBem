#pragma once
/// @file mesh_io.hpp
/// Import / export of general surface meshes (STL, OBJ, Gmsh .msh). Entry
/// point for arbitrary geometries in later project phases.
///
/// Units: files are read and written in metres (SI). No scaling is applied on import or
/// export; a file authored in millimetres or micrometres must be rescaled by the caller.
///
/// Supported formats:
///  * **STL**, binary and ASCII. Binary: 80-byte header, uint32 triangle count, 50-byte
///    little-endian records (float32 normal, 3 x float32 vertices, uint16 attribute).
///    ASCII: `solid` / `facet normal` / `outer loop` / `vertex` / `endloop` / `endfacet` /
///    `endsolid` (several solids in one file are concatenated). A file is binary if its
///    size equals 84 + 50 n for the count n stored at byte offset 80 (a binary file may
///    start with "solid"); otherwise it is parsed as ASCII. The stored facet normals are
///    ignored, the orientation comes from the vertex order. STL stores no connectivity:
///    vertices are welded on import by **exact** equality of the three coordinates (as
///    read; -0 and +0 are equal), numbered in order of first appearance. STL exporters
///    write identical bits for shared vertices, so this recovers the connectivity; no
///    tolerance-based welding is done (nearly coincident vertices stay distinct).
///  * **OBJ** (Wavefront): `v x y z` and `f a b c ...` lines with 1-based indices; the
///    forms `a/b/c`, `a//c` and `a/b` use the first (vertex) index. Faces with more than
///    three vertices are fan-triangulated (a, b, c), (a, c, d), ... with a warning.
///    Negative (relative) indices are rejected. All other statements (`vn`, `vt`, `o`,
///    `g`, `s`, `l`, `mtllib`, `usemtl`, comments, ...) are ignored. All `v` vertices are
///    kept in file order, including unreferenced ones.
///  * **Gmsh** MSH 4.1 ASCII (`$MeshFormat 4.1 0 8` required): `$Nodes` entity blocks
///    with arbitrary (non-contiguous) node tags, non-parametric only; `$Elements` entity
///    blocks of which 3-node triangles (type 2) are imported, points (15) and 2-node
///    lines (1) are skipped, any other element type is an error. Nodes not referenced by
///    a triangle are dropped; the remaining vertices keep the `$Nodes` order. Other
///    sections (`$Entities`, `$PhysicalNames`, ...) are skipped.
///
/// Every reader returns `TriangleMesh(vertices, triangles)` with the triangles in file
/// order and the vertex order inside each triangle as stored, so the constructor's
/// validation and orientation repair apply (an inconsistently wound file is repaired
/// with a warning, a closed surface is oriented outward; see mesh.hpp).
///
/// Writers: binary STL stores float32 coordinates (relative rounding ~6e-8, so a round
/// trip is exact only to ~1e-7 relative) and the mesh normals; ASCII STL, OBJ and Gmsh
/// write every double in its shortest round-trip decimal form (at most 17 significant
/// digits), so a round trip through them reproduces the coordinates bit for bit.
///
/// Errors (all readers):
///  * missing / unreadable file -> std::runtime_error naming the path;
///  * malformed content, no triangles, or a mesh rejected by the TriangleMesh
///    constructor (degenerate / duplicate triangles, non-manifold edges, ...)
///    -> std::runtime_error with the path and, where available, the line number.
/// Writers throw std::invalid_argument for a mesh without triangles and
/// std::runtime_error if the file cannot be written. They write to `path + ".tmp"` and
/// rename it to `path` only after a successful close, so a failed write never leaves a
/// truncated file at the target (the temporary file is removed).
///
/// Text formats accept LF, CRLF and CR-only line endings, a missing final newline and a
/// leading UTF-8 byte-order mark.
#include "specklebem/geometry/mesh.hpp"

#include <string>

namespace specklebem::geometry::io {

/// Encoding used by write_stl.
enum class StlFormat { binary, ascii };

/// Reads a mesh, the format chosen from the case-insensitive extension (`.stl`, `.obj`,
/// `.msh`). @throws std::invalid_argument for any other extension.
TriangleMesh read_mesh(const std::string& path);
/// Writes a mesh, the format chosen from the case-insensitive extension: `.stl` (binary
/// STL), `.obj`, `.msh` (Gmsh 4.1 ASCII). @throws std::invalid_argument for any other
/// extension.
void write_mesh(const TriangleMesh& m, const std::string& path);

/// Binary or ASCII STL (detected from the file size, see above).
TriangleMesh read_stl(const std::string& path);
/// Wavefront OBJ (vertices and faces only).
TriangleMesh read_obj(const std::string& path);
/// Gmsh MSH 4.1 ASCII (3-node triangles only).
TriangleMesh read_gmsh(const std::string& path);

/// STL with the mesh normals as facet normals. Binary (default): 80-byte header that does
/// not start with "solid", float32 coordinates. ASCII: shortest round-trip doubles.
/// @throws std::invalid_argument for a mesh without triangles or more than 2^32 - 1
///         triangles (binary).
void write_stl(const TriangleMesh& m, const std::string& path,
               StlFormat format = StlFormat::binary);
/// OBJ with one `v` line per vertex and one 1-based `f` line per triangle.
void write_obj(const TriangleMesh& m, const std::string& path);
/// Gmsh MSH 4.1 ASCII: one surface entity (`$Entities` with its bounding box), one 2-D
/// node block with tags 1..V and one triangle block with element tags 1..F.
void write_gmsh(const TriangleMesh& m, const std::string& path);

}  // namespace specklebem::geometry::io
