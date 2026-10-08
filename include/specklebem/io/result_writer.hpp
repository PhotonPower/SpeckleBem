#pragma once
/// @file result_writer.hpp
/// Persistent output of meshes, currents, fields and statistics. HDF5 is the
/// primary format (self-describing, readable from Python via h5py); NumPy .npy
/// is available as a dependency-free fallback.
#include "specklebem/core/types.hpp"
#include "specklebem/geometry/mesh.hpp"

#include <string>

namespace specklebem::io {

class ResultWriter {
public:
    virtual ~ResultWriter() = default;
    virtual void write_mesh(const std::string& group, const geometry::TriangleMesh& m) = 0;
    virtual void write_vector(const std::string& name, const VectorXc& v) = 0;
    virtual void write_matrix(const std::string& name, const MatrixXr& m) = 0;
    virtual void write_matrix(const std::string& name, const MatrixXc& m) = 0;
    virtual void write_attribute(const std::string& name, const std::string& value) = 0;
    virtual void write_attribute(const std::string& name, Real value) = 0;
};

std::unique_ptr<ResultWriter> open_hdf5(
    const std::string& path);  ///< requires SPECKLEBEM_ENABLE_HDF5
std::unique_ptr<ResultWriter> open_npy_directory(const std::string& dir);

}  // namespace specklebem::io
