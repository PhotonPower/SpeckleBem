/// Bindings of geometry/: TriangleMesh (zero-copy read-only views), spheres, rough surfaces.
#include "specklebem/geometry/mesh.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/geometry/sphere.hpp"

#include <pybind11/stl.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "common.hpp"

namespace specklebem::python {

namespace {

using geometry::HeightMap;
using geometry::MeshQuality;
using geometry::RoughSurfaceParams;
using geometry::TriangleMesh;
using OptReal = std::optional<Real>;

RoughSurfaceParams rough_params(Real L, Real sigma, Real Lc, Real mesh_size, std::uint64_t seed,
                                bool use_fft) {
    RoughSurfaceParams p;
    p.edge_length_L = L;
    p.rms_roughness = sigma;
    p.correlation_length = Lc;
    p.mesh_size = mesh_size;
    p.seed = seed;
    p.use_fft = use_fft;
    return p;
}

}  // namespace

void bind_geometry(py::module_& m) {
    py::class_<MeshQuality>(m, "MeshQuality", "Statistics of TriangleMesh.quality_report().")
        .def_readonly("num_vertices", &MeshQuality::num_vertices)
        .def_readonly("num_unreferenced_vertices", &MeshQuality::num_unreferenced_vertices)
        .def_readonly("num_edges", &MeshQuality::num_edges)
        .def_readonly("num_boundary_edges", &MeshQuality::num_boundary_edges)
        .def_readonly("num_nonmanifold_edges", &MeshQuality::num_nonmanifold_edges)
        .def_readonly("num_triangles", &MeshQuality::num_triangles)
        .def_readonly("euler_characteristic", &MeshQuality::euler_characteristic)
        .def_readonly("num_components", &MeshQuality::num_components)
        .def_readonly("num_vertex_components", &MeshQuality::num_vertex_components)
        .def_readonly("closed", &MeshQuality::closed)
        .def_readonly("consistently_oriented", &MeshQuality::consistently_oriented)
        .def_readonly("min_edge_length", &MeshQuality::min_edge_length)
        .def_readonly("max_edge_length", &MeshQuality::max_edge_length)
        .def_readonly("mean_edge_length", &MeshQuality::mean_edge_length)
        .def_readonly("max_aspect_ratio", &MeshQuality::max_aspect_ratio)
        .def("__str__", [](const MeshQuality& q) { return geometry::to_string(q); });

    const auto view = py::return_value_policy::reference_internal;
    py::class_<TriangleMesh>(m, "TriangleMesh",
                             "Closed or open triangle surface mesh (normals out of R2 into R1).")
        .def(py::init([](const py::object& vertices, const py::object& triangles) {
                 Vertices v = points_from_array(vertices, "vertices");
                 Triangles t = triangles_from_array(triangles);
                 py::gil_scoped_release release;
                 return TriangleMesh(std::move(v), std::move(t));
             }),
             py::arg("vertices"), py::arg("triangles"))
        // Read-only zero-copy views; the array keeps the mesh alive (base object).
        .def_property_readonly(
            "vertices", [](const TriangleMesh& s) -> const Vertices& { return s.vertices(); }, view)
        .def_property_readonly(
            "triangles", [](const TriangleMesh& s) -> const Triangles& { return s.triangles(); },
            view)
        .def_property_readonly(
            "edges", [](const TriangleMesh& s) -> const Edges& { return s.edges(); }, view)
        .def_property_readonly("num_vertices", &TriangleMesh::num_vertices)
        .def_property_readonly("num_triangles", &TriangleMesh::num_triangles)
        .def_property_readonly("num_edges", &TriangleMesh::num_edges)
        .def_property_readonly("num_boundary_edges", &TriangleMesh::num_boundary_edges)
        .def_property_readonly("num_components", &TriangleMesh::num_components)
        .def("is_closed", &TriangleMesh::is_closed)
        .def("signed_volume", &TriangleMesh::signed_volume)
        .def("bounding_box", &TriangleMesh::bounding_box)
        .def("quality_report", &TriangleMesh::quality, py::call_guard<py::gil_scoped_release>())
        .def("flip_normals", &TriangleMesh::flip_normals)
        .def("__repr__", [](const TriangleMesh& s) {
            return "<TriangleMesh V=" + std::to_string(s.num_vertices()) +
                   " F=" + std::to_string(s.num_triangles()) +
                   " E=" + std::to_string(s.num_edges()) + ">";
        });

    m.def("make_icosphere", &geometry::make_icosphere, py::arg("radius"), py::arg("subdivisions"),
          py::arg("center") = Vec3::Zero(), py::call_guard<py::gil_scoped_release>());
    m.def("make_sphere", &geometry::make_sphere, py::arg("radius"), py::arg("target_edge_length"),
          py::arg("center") = Vec3::Zero(), py::call_guard<py::gil_scoped_release>());

    py::class_<HeightMap>(m, "HeightMap", "Height map z(i, j) at (x_i, y_j), centred grid [m].")
        .def(py::init([](const py::object& z, Real dx, Real dy) {
                 const auto a = real_array(z, "z");
                 if (a.ndim() != 2) {
                     throw py::value_error("z: expected an (n_x, n_y) array");
                 }
                 const auto rows = static_cast<Index>(a.shape(0));
                 const auto cols = static_cast<Index>(a.shape(1));
                 return HeightMap{Eigen::Map<const MatrixXrRow>(a.data(), rows, cols), dx, dy};
             }),
             py::arg("z"), py::arg("dx"), py::arg("dy"))
        // C-contiguous float64 copy (n_x, n_y) [m].
        .def_property_readonly("z", [](const HeightMap& h) { return MatrixXrRow(h.z); })
        .def_readonly("dx", &HeightMap::dx)
        .def_readonly("dy", &HeightMap::dy)
        .def("rms", &HeightMap::rms)
        .def("estimated_correlation_length", &HeightMap::estimated_correlation_length,
             py::call_guard<py::gil_scoped_release>());

    const RoughSurfaceParams d;
    m.def(
        "generate_gaussian_height_map",
        [](Real L, Real sigma, Real Lc, Real mesh_size, std::uint64_t seed, bool use_fft) {
            return geometry::generate_gaussian_height_map(
                rough_params(L, sigma, Lc, mesh_size, seed, use_fft));
        },
        py::kw_only(), py::arg("L") = d.edge_length_L, py::arg("sigma") = d.rms_roughness,
        py::arg("Lc") = d.correlation_length, py::arg("mesh_size") = d.mesh_size,
        py::arg("seed") = d.seed, py::arg("use_fft") = d.use_fft,
        py::call_guard<py::gil_scoped_release>());
    m.def(
        "make_rough_surface_mesh",
        [](Real L, Real sigma, Real Lc, Real mesh_size, std::uint64_t seed, OptReal box_depth,
           OptReal box_mesh_size, OptReal box_fine_depth, bool use_fft) {
            RoughSurfaceParams p = rough_params(L, sigma, Lc, mesh_size, seed, use_fft);
            p.box_depth = box_depth;
            p.box_mesh_size = box_mesh_size;
            p.box_fine_depth = box_fine_depth;
            return geometry::make_rough_surface_mesh(p);
        },
        py::kw_only(), py::arg("L") = d.edge_length_L, py::arg("sigma") = d.rms_roughness,
        py::arg("Lc") = d.correlation_length, py::arg("mesh_size") = d.mesh_size,
        py::arg("seed") = d.seed, py::arg("box_depth") = py::none(),
        py::arg("box_mesh_size") = py::none(), py::arg("box_fine_depth") = py::none(),
        py::arg("use_fft") = d.use_fft, py::call_guard<py::gil_scoped_release>());
    m.def("make_mesh_from_height_map", &geometry::make_mesh_from_height_map, py::arg("height_map"),
          py::kw_only(), py::arg("box_depth") = py::none(), py::arg("box_mesh_size") = py::none(),
          py::arg("box_fine_depth") = py::none(), py::call_guard<py::gil_scoped_release>());
}

}  // namespace specklebem::python
