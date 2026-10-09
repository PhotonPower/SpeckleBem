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
    py::class_<MeshQuality>(m, "MeshQuality",
                            "Mesh statistics of TriangleMesh.quality() (edge lengths [m]); "
                            "str(q) gives the text of TriangleMesh.quality_report().")
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
        .def("__str__", [](const MeshQuality& q) { return geometry::to_string(q); })
        .def("__repr__", [](const MeshQuality& q) {
            return "<MeshQuality V=" + std::to_string(q.num_vertices) +
                   " E=" + std::to_string(q.num_edges) + " F=" + std::to_string(q.num_triangles) +
                   " chi=" + std::to_string(q.euler_characteristic) +
                   " closed=" + (q.closed ? "True" : "False") + ">";
        });

    const auto view = py::return_value_policy::reference_internal;
    py::class_<TriangleMesh>(m, "TriangleMesh", R"doc(
Closed or open triangle surface mesh; normals point out of R2 (object) into R1.

Parameters
----------
vertices : array_like, shape (V, 3)
    Real vertex coordinates [m] (copied).
triangles : array_like, shape (F, 3)
    Integer vertex indices, counter-clockwise seen from R1 (copied).

Notes
-----
The constructor re-orients the triangles so that every component is consistently oriented
and closed components face outward, so ``mesh.triangles`` may differ from the input.
``vertices``, ``triangles`` and ``edges`` are read-only zero-copy views that keep the mesh
alive and reflect a later ``flip_normals()``.

Raises
------
ValueError
    For bad shapes or dtypes, out-of-range or repeated indices, zero-area or duplicate
    triangles, non-finite coordinates, non-manifold edges or a non-orientable surface.
)doc")
        .def(py::init([](const py::object& vertices, const py::object& triangles) {
                 Vertices v = points_from_array(vertices, "vertices");
                 Triangles t = triangles_from_array(triangles);
                 py::gil_scoped_release release;
                 return TriangleMesh(std::move(v), std::move(t));
             }),
             py::arg("vertices"), py::arg("triangles"))
        // Read-only zero-copy views; the array keeps the mesh alive (base object).
        .def_property_readonly(
            "vertices", [](const TriangleMesh& s) -> const Vertices& { return s.vertices(); }, view,
            "(V, 3) float64 vertex coordinates [m], read-only view.")
        .def_property_readonly(
            "triangles", [](const TriangleMesh& s) -> const Triangles& { return s.triangles(); },
            view, "(F, 3) int64 vertex indices (after re-orientation), read-only view.")
        .def_property_readonly(
            "edges", [](const TriangleMesh& s) -> const Edges& { return s.edges(); }, view,
            "(N, 2) int64 interior edges (a, b), a < b, one RWG function each; read-only view.")
        .def_property_readonly("num_vertices", &TriangleMesh::num_vertices, "Number of vertices.")
        .def_property_readonly("num_triangles", &TriangleMesh::num_triangles,
                               "Number of triangles.")
        .def_property_readonly("num_edges", &TriangleMesh::num_edges,
                               "Number of interior edges N (= number of RWG functions).")
        .def_property_readonly("num_boundary_edges", &TriangleMesh::num_boundary_edges,
                               "Number of boundary edges (one adjacent triangle).")
        .def_property_readonly("num_components", &TriangleMesh::num_components,
                               "Number of edge-connected components.")
        .def("is_closed", &TriangleMesh::is_closed,
             "True if there are no boundary edges and at least one triangle.")
        .def("signed_volume", &TriangleMesh::signed_volume,
             "Signed enclosed volume [m^3], positive for outward normals (closed meshes).")
        .def("bounding_box", &TriangleMesh::bounding_box,
             "(lo, hi) corners [m] of the axis-aligned bounding box; RuntimeError if empty.")
        .def("quality", &TriangleMesh::quality,
             "Mesh statistics as a MeshQuality (counts, closedness, edge lengths [m]).")
        .def("quality_report", &TriangleMesh::quality_report,
             "Multi-line text report of quality() (same as str(mesh.quality())).")
        .def("flip_normals", &TriangleMesh::flip_normals,
             "Reverse the orientation of every triangle (negates normals and volume).")
        .def("__repr__", [](const TriangleMesh& s) {
            return "<TriangleMesh V=" + std::to_string(s.num_vertices()) +
                   " F=" + std::to_string(s.num_triangles()) +
                   " E=" + std::to_string(s.num_edges()) + ">";
        });

    m.def("make_icosphere", &geometry::make_icosphere, py::arg("radius"), py::arg("subdivisions"),
          py::arg("center") = Vec3::Zero(), py::call_guard<py::gil_scoped_release>(), R"doc(
Icosphere of the given radius [m] around center [m], refined ``subdivisions`` times.

20 * 4^n triangles, 10 * 4^n + 2 vertices, 30 * 4^n edges, normals outward. ValueError for
subdivisions outside [0, 10], radius <= 0 or non-finite input.
)doc");
    m.def("make_sphere", &geometry::make_sphere, py::arg("radius"), py::arg("target_edge_length"),
          py::arg("center") = Vec3::Zero(), py::call_guard<py::gil_scoped_release>(), R"doc(
Icosphere with the fewest subdivisions whose mean edge length is <= target_edge_length.

radius, target_edge_length and center in metres. ValueError for non-positive or non-finite
input or a target that needs more than 10 subdivisions.
)doc");

    py::class_<HeightMap>(m, "HeightMap", R"doc(
Height map on a regular grid centred at the origin.

Parameters
----------
z : array_like, shape (n_x, n_y)
    Heights [m]; z[i, j] at x_i = -(n_x - 1) dx / 2 + i dx, y_j = -(n_y - 1) dy / 2 + j dy.
dx, dy : float
    Grid spacings [m].
)doc")
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
        .def_property_readonly(
            "z", [](const HeightMap& h) { return MatrixXrRow(h.z); },
            "(n_x, n_y) float64 heights [m] (a copy).")
        .def_readonly("dx", &HeightMap::dx, "Grid spacing along x [m].")
        .def_readonly("dy", &HeightMap::dy, "Grid spacing along y [m].")
        .def("rms", &HeightMap::rms, "Root mean square of the heights about their mean [m].")
        .def("estimated_correlation_length", &HeightMap::estimated_correlation_length,
             py::call_guard<py::gil_scoped_release>(),
             "Correlation length Lc [m] estimated from the height autocorrelation.");

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
        py::call_guard<py::gil_scoped_release>(), R"doc(
Gaussian random height map (Fu et al. 2023, Eqs. 8-9) on an L x L patch.

Parameters
----------
L : float
    Patch edge length [m].
sigma : float
    RMS roughness [m] about the mean plane z = 0.
Lc : float
    Correlation length [m]: C(r) / C(0) = exp(-r^2 / Lc^2).
mesh_size : float
    Target grid spacing [m].
seed : int
    RNG seed; 0 draws a random seed that is only logged (reproducible only by reading the logged seed).
use_fft : bool
    FFT convolution (fast) instead of direct convolution (reference).

Returns
-------
HeightMap
)doc");
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
        py::arg("use_fft") = d.use_fft, py::call_guard<py::gil_scoped_release>(), R"doc(
Closed rough-surface mesh: a Gaussian height map (see generate_gaussian_height_map, same
L, sigma, Lc, mesh_size, seed, use_fft) closed by a box below it.

The object occupies z > xi(x, y); light comes from z < 0. box_depth [m] (bottom plate at
z = +box_depth), box_mesh_size [m] (coarse box spacing) and box_fine_depth [m] (fine band
below z = 0) default to the automatic rules of make_mesh_from_height_map. seed=0 draws a
random seed that is only logged.
)doc");
    m.def("make_mesh_from_height_map", &geometry::make_mesh_from_height_map, py::arg("height_map"),
          py::kw_only(), py::arg("box_depth") = py::none(), py::arg("box_mesh_size") = py::none(),
          py::arg("box_fine_depth") = py::none(), py::call_guard<py::gil_scoped_release>(),
          R"doc(
Closed mesh of a HeightMap: top face from the heights plus side walls and a bottom plate.

box_depth [m]: bottom plate at z = +box_depth (None: 2e-6 m, ADR 0006). box_mesh_size [m]:
coarse spacing of the lower walls and the bottom (None: min(depth / 2, L / 8,
10 h, h = min(dx, dy))). box_fine_depth [m]: the walls keep the top-face spacing down to this z
(None: no fine band); size it as 3 delta + 3 sigma with delta = field_decay_length.
)doc");
}

}  // namespace specklebem::python
