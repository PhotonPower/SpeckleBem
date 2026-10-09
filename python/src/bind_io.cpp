/// Bindings of the mesh file I/O (geometry/mesh_io.hpp) and the result writer
/// (io/result_writer.hpp). Paths: str or os.PathLike, passed to C++ as UTF-8 (ADR 0007).
#include "specklebem/geometry/mesh_io.hpp"
#include "specklebem/io/result_writer.hpp"

#include <memory>
#include <string>

#include "common.hpp"

namespace specklebem::python {

namespace {

using geometry::TriangleMesh;
using io::ResultWriter;

/// Wraps a C++ reader `f(const std::string&)` as `f(path)` with the GIL released.
template <TriangleMesh (*F)(const std::string&)>
TriangleMesh read(const py::object& path) {
    const std::string p = utf8_path(path);
    py::gil_scoped_release release;
    return F(p);
}

template <void (*F)(const TriangleMesh&, const std::string&)>
void write(const TriangleMesh& mesh, const py::object& path) {
    const std::string p = utf8_path(path);
    py::gil_scoped_release release;
    F(mesh, p);
}

/// Real or complex numeric array, checked for its dtype kind; ValueError otherwise.
py::array numeric_array(const py::object& obj, const char* what, int ndim) {
    const auto a = py::array::ensure(obj);
    const char k = a ? a.dtype().kind() : '?';
    if ((k != 'c' && k != 'f' && k != 'i' && k != 'u') || a.ndim() != ndim) {
        throw py::value_error(std::string(what) + ": expected a " + std::to_string(ndim) +
                              "-D real or complex numeric array");
    }
    return a;
}

template <class T>
using CArray = py::array_t<T, py::array::c_style | py::array::forcecast>;
using MatrixXcRow = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

}  // namespace

void bind_io(py::module_& m) {
    m.def("read_mesh", &read<&geometry::io::read_mesh>, py::arg("path"),
          "Read a TriangleMesh [m]; format from the extension (.stl, .obj, .msh). path: str or "
          "os.PathLike (UTF-8, ADR 0007). RuntimeError for unreadable or malformed files.");
    m.def("write_mesh", &write<&geometry::io::write_mesh>, py::arg("mesh"), py::arg("path"),
          "Write a TriangleMesh [m]; format from the extension (.stl binary, .obj, .msh Gmsh "
          "4.1). path: str or os.PathLike (UTF-8, ADR 0007).");
    m.def("read_stl", &read<&geometry::io::read_stl>, py::arg("path"),
          "Read binary or ASCII STL (vertices welded by exact equality).");
    m.def("read_obj", &read<&geometry::io::read_obj>, py::arg("path"), "Read Wavefront OBJ.");
    m.def("read_gmsh", &read<&geometry::io::read_gmsh>, py::arg("path"),
          "Read Gmsh MSH 4.1 ASCII (3-node triangles).");
    m.def(
        "write_stl",
        [](const TriangleMesh& mesh, const py::object& path, bool ascii) {
            const std::string p = utf8_path(path);
            py::gil_scoped_release release;
            geometry::io::write_stl(mesh, p,
                                    ascii ? geometry::io::StlFormat::ascii
                                          : geometry::io::StlFormat::binary);
        },
        py::arg("mesh"), py::arg("path"), py::arg("ascii") = false,
        "Write STL: binary (float32 coordinates) or ASCII (shortest round-trip doubles).");
    m.def("write_obj", &write<&geometry::io::write_obj>, py::arg("mesh"), py::arg("path"),
          "Write Wavefront OBJ (bit-exact coordinates).");
    m.def("write_gmsh", &write<&geometry::io::write_gmsh>, py::arg("mesh"), py::arg("path"),
          "Write Gmsh MSH 4.1 ASCII (bit-exact coordinates).");

    py::class_<ResultWriter>(m, "ResultWriter", R"doc(
Sink for named arrays, meshes and attributes (C++ io::ResultWriter); see open_npy_directory.

Names consist of components [A-Za-z0-9_.-] separated by '/' (a group = sub-directory);
invalid names raise ValueError, I/O failures RuntimeError. Writing a name twice overwrites
it. Usable as a context manager (every write is complete on return; nothing to flush).
Not thread-safe.
)doc")
        .def(
            "write_vector",
            [](ResultWriter& w, const std::string& name, const py::object& v) {
                const auto c = CArray<Complex>::ensure(numeric_array(v, "v", 1));
                const VectorXc x =
                    Eigen::Map<const VectorXc>(c.data(), static_cast<Index>(c.shape(0)));
                py::gil_scoped_release release;
                w.write_vector(name, x);
            },
            py::arg("name"), py::arg("v"), "Write a 1-D array as <name>.npy (complex128).")
        .def(
            "write_matrix",
            [](ResultWriter& w, const std::string& name, const py::object& mat) {
                const py::array a = numeric_array(mat, "m", 2);
                const auto rows = static_cast<Index>(a.shape(0));
                const auto cols = static_cast<Index>(a.shape(1));
                if (a.dtype().kind() == 'c') {
                    const auto c = CArray<Complex>::ensure(a);
                    const MatrixXc x = Eigen::Map<const MatrixXcRow>(c.data(), rows, cols);
                    py::gil_scoped_release release;
                    w.write_matrix(name, x);
                } else {
                    const auto r = CArray<double>::ensure(a);
                    const MatrixXr x = Eigen::Map<const MatrixXrRow>(r.data(), rows, cols);
                    py::gil_scoped_release release;
                    w.write_matrix(name, x);
                }
            },
            py::arg("name"), py::arg("m"),
            "Write a 2-D array as <name>.npy: float64 for real, complex128 for complex input.")
        .def(
            "write_mesh",
            [](ResultWriter& w, const std::string& group, const TriangleMesh& mesh) {
                py::gil_scoped_release release;
                w.write_mesh(group, mesh);
            },
            py::arg("group"), py::arg("mesh"),
            "Write <group>/vertices.npy (float64 [m]) and <group>/triangles.npy (int64).")
        .def(
            "write_attribute",
            [](ResultWriter& w, const std::string& name, const std::string& value) {
                w.write_attribute(name, value);
            },
            py::arg("name"), py::arg("value"), "Write a string attribute (attributes.json).")
        .def(
            "write_attribute",
            [](ResultWriter& w, const std::string& name, Real value) {
                w.write_attribute(name, value);
            },
            py::arg("name"), py::arg("value"),
            "Write a finite float attribute (attributes.json); ValueError for inf / nan.")
        .def("__enter__", [](const py::object& self) { return self; })
        .def("__exit__", [](const py::object&, const py::args&) { return false; });

    m.def(
        "open_npy_directory",
        [](const py::object& path) {
            const std::string p = utf8_path(path);
            py::gil_scoped_release release;
            return io::open_npy_directory(p);
        },
        py::arg("path"), R"doc(
ResultWriter into a directory of NumPy .npy files (created with its parents if needed).

Parameters
----------
path : str or os.PathLike
    Directory (UTF-8, ADR 0007). Arrays become <path>/<name>.npy, attributes one JSON object
    in <path>/attributes.json, read back with numpy.load / json.load.
)doc");
}

}  // namespace specklebem::python
