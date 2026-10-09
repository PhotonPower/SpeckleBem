#pragma once
/// @file result_writer.hpp
/// Persistent output of meshes, currents, fields and statistics. Implemented: a
/// dependency-free directory of NumPy .npy files (open_npy_directory). Planned: HDF5
/// (open_hdf5, self-describing, readable from Python via h5py), which throws until then.
#include "specklebem/core/types.hpp"
#include "specklebem/geometry/mesh.hpp"

#include <memory>
#include <string>

namespace specklebem::io {

/// Sink for named arrays, meshes and scalar attributes.
///
/// Names (and mesh groups) consist of components of the characters [A-Za-z0-9_.-]
/// separated by '/', which denotes a group (a sub-directory for the .npy writer). Empty
/// components, components "." and "..", components ending in '.' (Windows strips trailing
/// dots), components whose stem (the part before the first '.') is a Windows device name
/// (CON, PRN, AUX, NUL, COM1-COM9, LPT1-LPT9, case-insensitive; e.g. "nul" or "Com1.x"),
/// leading or trailing '/' and any other character are rejected with std::invalid_argument
/// on all platforms. Writing a name twice overwrites the earlier data. On case-insensitive
/// file systems (Windows, macOS by default) names differing only in case denote the same
/// file, so the later write overwrites the earlier one. I/O failures throw
/// std::runtime_error naming the path. A writer is not thread-safe.
class ResultWriter {
public:
    virtual ~ResultWriter() = default;
    /// Vertices (V x 3, Real) and triangles (F x 3, 0-based Index, orientation as stored).
    virtual void write_mesh(const std::string& group, const geometry::TriangleMesh& m) = 0;
    virtual void write_vector(const std::string& name, const VectorXc& v) = 0;
    virtual void write_matrix(const std::string& name, const MatrixXr& m) = 0;
    virtual void write_matrix(const std::string& name, const MatrixXc& m) = 0;
    /// String attribute; must be valid UTF-8 (std::invalid_argument otherwise).
    virtual void write_attribute(const std::string& name, const std::string& value) = 0;
    /// Real attribute; must be finite (std::invalid_argument otherwise).
    virtual void write_attribute(const std::string& name, Real value) = 0;
};

/// HDF5 writer. Not implemented yet: always throws std::runtime_error (with a message
/// saying whether the library was built with SPECKLEBEM_ENABLE_HDF5).
std::unique_ptr<ResultWriter> open_hdf5(const std::string& path);

/// Writer into a directory of NumPy .npy files (format version 1.0, little-endian, C order),
/// created with all parents if needed (an existing directory is reused; files of the same
/// name are overwritten). Every file is written to a temporary file "<file>~tmp" in the same
/// directory and renamed over the target only when complete, so a target is always either
/// the complete old or the complete new file; on failure the temporary file is removed and
/// the old file is kept. A target that is an existing directory is a std::runtime_error.
///  * write_vector(name, v)  -> <dir>/<name>.npy, dtype '<c16', shape (n,);
///  * write_matrix(name, m)  -> <dir>/<name>.npy, dtype '<f8' or '<c16', shape (rows, cols),
///    stored row-major ('fortran_order': False), so numpy.load(...)[i, j] == m(i, j);
///  * write_mesh(group, m)   -> <dir>/<group>/vertices.npy ('<f8', (V, 3)) and
///    <dir>/<group>/triangles.npy ('<i8', (F, 3));
///  * write_attribute(name, value) -> one JSON object in <dir>/attributes.json (UTF-8), keys in
///    order of first write, strings escaped per RFC 8259, Reals in shortest round-trip form
///    (always with '.' or an exponent, so Python reads them as float). The file is rewritten
///    completely (and atomically, see above) on every write_attribute call, so it is complete
///    at any time, in particular after the writer is destroyed. A new writer starts with no
///    attributes: its first write_attribute replaces an attributes.json left in the directory
///    by an earlier run.
/// @throws std::invalid_argument for an empty path, std::runtime_error if the directory
///         cannot be created.
std::unique_ptr<ResultWriter> open_npy_directory(const std::string& dir);

}  // namespace specklebem::io
