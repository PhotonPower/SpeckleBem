/// Bindings of the Simulation driver (simulation.hpp), its GMRES result, the system operator
/// and the post-processing (postprocessing/fields.hpp, scattering.hpp).
#include "specklebem/postprocessing/scattering.hpp"
#include "specklebem/simulation.hpp"

#include <pybind11/stl.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common.hpp"

namespace specklebem::python {

namespace {

using FieldMatrix = Eigen::Matrix<Complex, Eigen::Dynamic, 3>;
using formulation::Kind;
using solver::GmresParams;
using solver::GmresResult;
using solver::PreconditionerSide;

template <class T>
T dict_value(const py::handle& v, const std::string& key, const char* what) {
    try {
        return v.cast<T>();
    } catch (const py::cast_error&) {
        throw py::value_error(std::string(what) + "['" + key + "']: invalid value type");
    }
}

/// Dict key as str; non-str keys (int, bytes, ...) are a user error -> ValueError.
std::string dict_key(const py::handle& k, const char* what) {
    if (!py::isinstance<py::str>(k))
        throw py::value_error(std::string(what) + ": keys must be str");
    return py::cast<std::string>(k);
}

PreconditionerSide parse_side(const std::string& side) {
    const std::string s = to_lower(side);
    if (s == "left" || s == "right") {
        return s == "left" ? PreconditionerSide::Left : PreconditionerSide::Right;
    }
    throw py::value_error("side: expected \"left\" or \"right\"");
}

/// gmres=dict(tol, max_iter, restart (None = full GMRES), side, verbose) onto GmresParams.
void apply_gmres(GmresParams& p, const py::dict& d) {
    for (const auto& [k, v] : d) {
        const auto key = dict_key(k, "gmres");
        if (key == "tol") {
            p.tolerance = dict_value<Real>(v, key, "gmres");
        } else if (key == "max_iter") {
            p.max_iter = dict_value<int>(v, key, "gmres");
        } else if (key == "restart") {
            p.restart = v.is_none() ? 0 : dict_value<int>(v, key, "gmres");
        } else if (key == "side") {
            if (!py::isinstance<py::str>(v))
                throw py::value_error("gmres['side']: expected a str");
            p.side = parse_side(dict_value<std::string>(v, key, "gmres"));
        } else if (key == "verbose") {
            p.verbose = dict_value<bool>(v, key, "gmres");
        } else {
            throw py::value_error("gmres: unknown key '" + key + "'");
        }
    }
}

/// kernels=dict(...) onto kernels::OperatorOptions (keys = the C++ member names).
void apply_kernels(kernels::OperatorOptions& o, const py::dict& d) {
    for (const auto& [k, v] : d) {
        const auto key = dict_key(k, "kernels");
        if (key == "quad_degree_far") {
            o.quad_degree_far = dict_value<int>(v, key, "kernels");
        } else if (key == "quad_degree_near") {
            o.quad_degree_near = dict_value<int>(v, key, "kernels");
        } else if (key == "quad_degree_sing") {
            o.quad_degree_sing = dict_value<int>(v, key, "kernels");
        } else if (key == "outer_grading_levels") {
            o.outer_grading_levels = dict_value<int>(v, key, "kernels");
        } else if (key == "near_distance_factor") {
            o.near_distance_factor = dict_value<Real>(v, key, "kernels");
        } else if (key == "symmetrize_touching_above_kh") {
            o.symmetrize_touching_above_kh = dict_value<Real>(v, key, "kernels");
        } else if (key == "target_accuracy") {
            o.target_accuracy = dict_value<Real>(v, key, "kernels");
        } else {
            throw py::value_error("kernels: unknown key '" + key + "'");
        }
    }
}

std::optional<Kind> parse_formulation(const py::object& f) {
    if (py::isinstance<Kind>(f)) {
        return f.cast<Kind>();
    }
    if (py::isinstance<py::str>(f)) {
        const std::string s = to_lower(f.cast<std::string>());
        if (s == "auto") {
            return std::nullopt;
        }
        const std::pair<const char*, Kind> names[] = {{"pmchwt", Kind::PMCHWT},
                                                      {"ictf", Kind::ICTF},
                                                      {"mctf", Kind::MCTF},
                                                      {"jmcfie", Kind::JMCFIE}};
        for (const auto& [name, k] : names) {
            if (s == name) {
                return k;  // JMCFIE: rejected by the Simulation constructor (not implemented)
            }
        }
    }
    throw py::value_error("formulation: expected \"auto\", \"PMCHWT\", \"ICTF\" or \"MCTF\"");
}

std::unique_ptr<Simulation> make_simulation(
    const geometry::TriangleMesh& mesh, std::shared_ptr<excitation::Excitation> exc,
    const material::Material& object, const std::optional<material::Material>& exterior,
    std::optional<Real> wavelength, const py::object& form, const std::string& preconditioner,
    const std::string& solver_name, const std::string& compression,
    const std::optional<py::dict>& gmres, const std::optional<py::dict>& kernel_opts) {
    if (!exc) {
        throw py::value_error("excitation: must not be None");
    }
    SimulationConfig c;
    c.wavelength = wavelength.value_or(exc->wavelength());
    c.exterior = exterior.value_or(material::vacuum());
    c.object = object;
    c.formulation = parse_formulation(form);
    const std::string pre = to_lower(preconditioner);
    if (pre == "diagonal" || pre == "none") {
        c.diagonal_preconditioner = pre == "diagonal";
    } else if (pre != "auto") {
        throw py::value_error("preconditioner: expected \"auto\", \"none\" or \"diagonal\"");
    }
    const std::string sol = to_lower(solver_name);
    if (sol != "gmres" && sol != "direct") {
        throw py::value_error("solver: expected \"gmres\" or \"direct\"");
    }
    c.solver = sol == "direct" ? SolverKind::Direct : SolverKind::Gmres;
    c.compression = compression;
    if (gmres) {
        apply_gmres(c.gmres, *gmres);
    }
    if (kernel_opts) {
        apply_kernels(c.kernels, *kernel_opts);
    }
    geometry::TriangleMesh copy = mesh;
    py::gil_scoped_release release;
    return std::make_unique<Simulation>(std::move(copy), std::move(exc), std::move(c));
}

GmresResult solve(Simulation& sim, std::optional<Real> tol, std::optional<int> max_iter,
                  std::optional<int> restart, const std::optional<std::string>& side,
                  const py::object& callback) {
    GmresParams p = sim.config().gmres;
    p.tolerance = tol.value_or(p.tolerance);
    p.max_iter = max_iter.value_or(p.max_iter);
    p.restart = restart.value_or(p.restart);
    if (side) {
        p.side = parse_side(*side);
    }
    solver::IterationCallback cb;
    if (!callback.is_none()) {
        if (!PyCallable_Check(callback.ptr())) {
            throw py::type_error("callback: expected a callable f(iteration, residual)");
        }
        // Holds a pointer only: copies of the std::function never touch Python reference
        // counts without the GIL. A Python exception leaves gmres as py::error_already_set and
        // is restored by pybind11 once the GIL is re-acquired.
        const py::object* fn = &callback;
        cb = [fn](int iteration, Real residual) {
            py::gil_scoped_acquire gil;
            (*fn)(iteration, residual);
        };
    }
    py::gil_scoped_release release;
    return sim.solve(p, cb);
}

py::object vector_or_scalar(const VectorXr& v, bool scalar) {
    return scalar ? py::object(py::float_(v(0))) : py::cast(v);
}

py::object bistatic_rcs(const Simulation& sim, const py::object& angles, const py::object& plane) {
    Vec3 normal;
    if (py::isinstance<py::str>(plane)) {
        const std::string s = to_lower(plane.cast<std::string>());
        if (s != "xz" && s != "yz") {
            throw py::value_error("plane: expected \"xz\", \"yz\" or a (3,) normal vector");
        }
        normal = s == "xz" ? Vec3::UnitY() : Vec3(-Vec3::UnitX());
    } else {
        const auto n = real_array(plane, "plane");
        if (n.ndim() != 1 || n.shape(0) != 3) {
            throw py::value_error("plane: expected \"xz\", \"yz\" or a (3,) normal vector");
        }
        normal = Eigen::Map<const Vec3>(n.data());
    }
    const auto a = real_array(angles, "angles");
    if (a.ndim() > 1) {
        throw py::value_error("angles: expected a scalar or a 1-D array");
    }
    const VectorXr th = Eigen::Map<const VectorXr>(a.data(), static_cast<Index>(a.size()));
    require_all_finite(th, "angles");
    const post::SurfaceSolution& s = sim.solution();
    VectorXr sigma;
    {
        py::gil_scoped_release release;
        sigma = post::bistatic_rcs(s, normal, th);
    }
    return vector_or_scalar(sigma, a.ndim() == 0);
}

py::tuple field(const Simulation& sim, const py::object& points, const std::string& kind,
                int quad_degree, Real min_distance_factor) {
    bool single = false;
    const Vertices p = points_from_array(points, "points", &single);
    require_all_finite(p, "points");
    const std::string k = to_lower(kind);
    if (k != "scattered" && k != "total") {
        throw py::value_error("kind: expected \"scattered\" or \"total\"");
    }
    const post::SurfaceSolution& s = sim.solution();
    const post::FieldOptions opt{quad_degree, min_distance_factor};
    FieldMatrix E, H;
    {
        py::gil_scoped_release release;
        if (k == "total") {
            post::total_field(s, p, E, H, opt);
        } else {
            post::scattered_field(s, p, E, H, opt);
        }
    }
    return py::make_tuple(fields_to_array(E, single), fields_to_array(H, single));
}

py::object far_field(const Simulation& sim, const py::object& directions, int quad_degree) {
    bool single = false;
    const Vertices d = points_from_array(directions, "directions", &single);
    require_all_finite(d, "directions");
    const post::SurfaceSolution& s = sim.solution();
    FieldMatrix F;
    {
        py::gil_scoped_release release;
        post::far_field(s, d, F, post::FieldOptions{quad_degree, 1e-3});
    }
    return fields_to_array(F, single);
}

VectorXc matvec(const op::LinearOperator& A, const py::object& x) {
    const auto a = py::array::ensure(x);
    const char k = a ? a.dtype().kind() : '?';
    const bool shape_ok = a && ((a.ndim() == 1 && a.shape(0) == A.cols()) ||
                                (a.ndim() == 2 && a.shape(0) == A.cols() && a.shape(1) == 1));
    if ((k != 'c' && k != 'f' && k != 'i' && k != 'u') || !shape_ok) {
        throw py::value_error("x: expected a numeric array of shape (n,) or (n, 1), n = " +
                              std::to_string(A.cols()));
    }
    const auto c = py::array_t<Complex, py::array::c_style | py::array::forcecast>::ensure(a);
    const VectorXc v = Eigen::Map<const VectorXc>(c.data(), A.cols());
    require_all_finite(v, "x");
    py::gil_scoped_release release;
    return A * v;
}

std::string result_repr(const GmresResult& r) {
    return "<SolveResult iterations=" + std::to_string(r.iterations) +
           " converged=" + (r.converged ? "True" : "False") +
           " true_relative_residual=" + std::to_string(r.true_relative_residual) + ">";
}

}  // namespace

void bind_simulation(py::module_& m) {
    py::class_<GmresResult>(m, "SolveResult", R"doc(
Result of Simulation.solve() (C++ solver::GmresResult).

Attributes
----------
iterations : int
    GMRES iterations (0 for the direct solver).
converged : bool
    Monitored residual <= tolerance (always True for the direct solver).
residual_history : ndarray of float64, shape (iterations + 1,)
    Monitored relative residual per iteration (copy); entry 0 is the initial one. Left
    preconditioning monitors |M^-1 (b - Z x)| / |M^-1 b|, right preconditioning the true one.
true_relative_residual : float
    |b - Z x| / |b| of the returned solution, computed explicitly.
wall_seconds : float
    Solve time [s] (assembly excluded).
x : ndarray of complex128, shape (2N,)
    The solution [J; M] (read-only view kept alive by this result; J in A/m, M in V/m).
)doc")
        .def_readonly("iterations", &GmresResult::iterations)
        .def_readonly("converged", &GmresResult::converged)
        .def_readonly("true_relative_residual", &GmresResult::true_relative_residual)
        .def_readonly("wall_seconds", &GmresResult::wall_seconds)
        .def_readonly("x", &GmresResult::x)
        .def_property_readonly("residual_history",
                               [](const GmresResult& r) {
                                   return py::array_t<double>(
                                       static_cast<py::ssize_t>(r.residual_history.size()),
                                       r.residual_history.data());
                               })
        .def("__repr__", &result_repr);

    py::class_<op::LinearOperator, std::shared_ptr<op::LinearOperator>>(m, "LinearOperator",
                                                                        R"doc(
System operator Z of a Simulation (square, complex; never formed explicitly in general).

Obtained from Simulation.operator(); keeps its Simulation alive. ``matvec(x)`` and ``A @ x``
compute Z x with the GIL released; ``as_scipy()`` wraps it for scipy.sparse.linalg.
)doc")
        .def_property_readonly(
            "shape", [](const op::LinearOperator& A) { return py::make_tuple(A.rows(), A.cols()); })
        .def_property_readonly(
            "dtype", [](const op::LinearOperator&) { return py::dtype::of<Complex>(); },
            "numpy.complex128.")
        .def("matvec", &matvec, py::arg("x"), R"doc(
y = Z x.

Parameters
----------
x : array_like, shape (n,) or (n, 1)
    Finite real or complex vector, n = shape[1]; ValueError otherwise.

Returns
-------
ndarray of complex128, shape (n,)
)doc")
        .def("__matmul__", &matvec, py::arg("x"))
        .def(
            "as_scipy",
            [](const py::object& self) {
                const py::module_ spla = py::module_::import("scipy.sparse.linalg");
                return spla.attr("LinearOperator")(self.attr("shape"),
                                                   py::arg("matvec") = self.attr("matvec"),
                                                   py::arg("dtype") = self.attr("dtype"));
            },
            "scipy.sparse.linalg.LinearOperator calling matvec (imports SciPy lazily; "
            "ImportError if SciPy is not installed).")
        .def("describe", &op::LinearOperator::describe, "Human-readable description.")
        .def_property_readonly("memory_bytes", &op::LinearOperator::memory_bytes,
                               "Approximate memory footprint [bytes].")
        .def("__repr__",
             [](const op::LinearOperator& A) { return "<LinearOperator " + A.describe() + ">"; });

    py::class_<Simulation>(m, "Simulation", R"doc(
Dense SIE simulation of one object (C++ specklebem::Simulation): assemble, solve, post-process.

Parameters
----------
mesh : TriangleMesh
    Closed surface, normals out of the object R2 into R1 (copied).
excitation : Excitation
    Incident field in R1 (shared with the caller).
object : Material
    Material of R2 (keyword-only, as all following parameters).
exterior : Material, optional
    Material of R1; None = vacuum. Must equal ``excitation.background``.
wavelength : float, optional
    Vacuum wavelength [m]; None = ``excitation.wavelength`` (must agree to 1e-12).
formulation : str or Formulation
    "auto" (formulation.recommend(object.eps_r)), "PMCHWT", "ICTF" or "MCTF" (any case).
preconditioner : str
    "auto" (the recommendation's choice), "none" or "diagonal" (Jacobi); GMRES only.
solver : str
    "gmres" or "direct" (dense LU).
compression : str
    "dense" (the only strategy so far; others raise ValueError).
gmres : dict, optional
    Keys ``tol`` (1e-3), ``max_iter`` (2000), ``restart`` (None = full GMRES), ``side``
    ("left" / "right") and ``verbose`` (True: progress via the C++ log).
kernels : dict, optional
    Quadrature options (kernels::OperatorOptions member names): quad_degree_far,
    quad_degree_near, quad_degree_sing, outer_grading_levels, near_distance_factor,
    symmetrize_touching_above_kh, target_accuracy.

Raises
------
ValueError
    For unknown strings or dictionary keys and everything the C++ constructor rejects (open
    or inward-oriented mesh, wavelength or background mismatch, invalid options).

Notes
-----
Not thread-safe: do not call methods of one Simulation from several threads at once.
)doc")
        .def(py::init(&make_simulation), py::arg("mesh"), py::arg("excitation"), py::kw_only(),
             py::arg("object"), py::arg("exterior") = py::none(),
             py::arg("wavelength") = py::none(), py::arg("formulation") = "auto",
             py::arg("preconditioner") = "auto", py::arg("solver") = "gmres",
             py::arg("compression") = "dense", py::arg("gmres") = py::none(),
             py::arg("kernels") = py::none())
        .def("assemble", &Simulation::assemble, py::call_guard<py::gil_scoped_release>(),
             "Assemble Z, the right-hand side and (Jacobi GMRES) the diagonal; idempotent.")
        .def("solve", &solve, py::arg("tol") = py::none(), py::arg("max_iter") = py::none(),
             py::arg("restart") = py::none(), py::arg("side") = py::none(),
             py::arg("callback") = py::none(), R"doc(
Assemble if needed and solve; the solution replaces any previous one.

Parameters
----------
tol, max_iter, restart, side : optional
    Per-call GMRES overrides of the ``gmres`` dict (restart 0 = full GMRES); None keeps it.
callback : callable, optional
    ``callback(iteration, residual)`` once per GMRES iteration (iteration = 1, 2, ...),
    called with the GIL held; an exception it raises aborts the solve and propagates.

Returns
-------
SolveResult
)doc")
        .def("report", &Simulation::report, "Multi-line text report (formulation, timings, ...).")
        .def_property_readonly("formulation", &Simulation::formulation_kind,
                               "Formulation used (resolved).")
        .def_property_readonly(
            "preconditioner",
            [](const Simulation& s) { return s.diagonal_preconditioner() ? "diagonal" : "none"; },
            "Preconditioner of the GMRES solve (resolved): \"diagonal\" or \"none\".")
        .def_property_readonly(
            "solver",
            [](const Simulation& s) {
                return s.config().solver == SolverKind::Direct ? "direct" : "gmres";
            },
            "\"gmres\" or \"direct\".")
        .def_property_readonly(
            "wavelength", [](const Simulation& s) { return s.config().wavelength; },
            "Vacuum wavelength [m].")
        .def_property_readonly("num_unknowns", &Simulation::num_unknowns,
                               "Number of unknowns 2N (N interior edges).")
        .def_property_readonly(
            "currents", [](const Simulation& s) { return VectorXc(s.solution().currents); },
            "Solution [J; M] (copy; J in A/m, M in V/m). RuntimeError before solve().")
        .def(
            "operator",
            [](const Simulation& s) {
                return std::const_pointer_cast<op::LinearOperator>(s.system_operator());
            },
            py::keep_alive<0, 1>(),
            "System operator Z (LinearOperator); RuntimeError before "
            "assemble().")
        .def(
            "rhs", [](const Simulation& s) { return VectorXc(s.rhs()); },
            "Right-hand side b (copy); RuntimeError before assemble().")
        .def("field", &field, py::arg("points"), py::arg("kind") = "scattered",
             py::arg("quad_degree") = 6, py::arg("min_distance_factor") = 1e-3, R"doc(
E [V/m] and H [A/m] at points [m] from the solved currents.

Parameters
----------
points : array_like, shape (n, 3) or (3,)
    Finite points; their region (R1 or R2) is found by the winding number of the mesh.
kind : str
    "scattered" (R1: scattered field, R2: total field) or "total" (adds the incident field
    in R1).
quad_degree, min_distance_factor :
    post::FieldOptions; points closer than min_distance_factor * h to the surface raise
    ValueError.

Returns
-------
(E, H) : ndarrays of complex128, shape (n, 3) or (3,)
)doc")
        .def("far_field", &far_field, py::arg("directions"), py::arg("quad_degree") = 6,
             "Far-field amplitude F [V] with E_s ~ F exp(-jkr) / r for directions (n, 3) "
             "(normalised internally); returns (n, 3) complex128.")
        .def("bistatic_rcs", &bistatic_rcs, py::arg("angles"), py::arg("plane") = "xz", R"doc(
Bistatic radar cross section sigma(theta) [m^2] (incidence along +z).

Parameters
----------
angles : float or array_like, shape (n,)
    Finite scattering angles theta [rad] from +z (0 = forward, pi = backward).
plane : str or array_like, shape (3,)
    "xz" (normal +y: phi = 0), "yz" (normal -x: phi = pi / 2) or a plane normal
    perpendicular to z (k_hat = rotation of +z about it by theta).

Returns
-------
float or ndarray of float64, shape (n,)
)doc");

    m.def(
        "plane_grid",
        [](const Vec3& origin, const Vec3& u, const Vec3& v, int nu, int nv) {
            require_all_finite(origin, "origin");
            require_all_finite(u, "u");
            require_all_finite(v, "v");
            return post::plane_grid(origin, u, v, nu, nv);
        },
        py::arg("origin"), py::arg("u"), py::arg("v"), py::arg("nu"), py::arg("nv"),
        "(nu * nv, 3) points origin + i / (nu - 1) u + j / (nv - 1) v [m], row i * nv + j "
        "(u, v are the full extents).");
    m.def("cylinder_grid", &post::cylinder_grid, py::arg("radius"), py::arg("theta_min"),
          py::arg("theta_max"), py::arg("n_theta"), py::arg("y_min"), py::arg("y_max"),
          py::arg("n_y"),
          "(n_theta * n_y, 3) points (r sin theta, y, -r cos theta) [m] on a cylinder around "
          "the y-axis, theta [rad] from -z, row k * n_y + l.");
    m.def(
        "polarized_intensity",
        [](const py::object& E, const Vec3& co_pol) {
            const FieldMatrix f = fields_from_array(E, "E");
            const post::PolarizedIntensity p = post::polarized_intensity(f, co_pol);
            return py::make_tuple(p.co, p.cross);
        },
        py::arg("E"), py::arg("co_pol"),
        "(co, cross) intensities [V^2/m^2] of E (n, 3): co = |E . e_co|^2, cross = |E|^2 - co.");
    m.def(
        "intensity",
        [](const py::object& E) {
            bool single = false;
            const FieldMatrix f = fields_from_array(E, "E", &single);
            return vector_or_scalar(f.rowwise().squaredNorm(), single);
        },
        py::arg("E"), "|E|^2 [V^2/m^2] per row of E (n, 3) (a float for one (3,) vector).");
    m.def(
        "differential_reflection_coefficient",
        [](const py::object& E, int n_theta, int n_y) {
            return post::differential_reflection_coefficient(fields_from_array(E, "E"), n_theta,
                                                             n_y);
        },
        py::arg("E_cyl"), py::arg("n_theta"), py::arg("n_y"),
        "Relative DRC: mean |E|^2 [V^2/m^2] over the n_y axial samples per theta of fields on "
        "cylinder_grid (row k * n_y + l).");
}

}  // namespace specklebem::python
