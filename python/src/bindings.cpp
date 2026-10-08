/// pybind11 bindings. Zero-copy NumPy interop via Eigen; the Python package
/// `specklebem` wraps this module with a friendlier API (python/specklebem/).
#include "specklebem/core/config.hpp"
#include "specklebem/core/logging.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/material/material.hpp"

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace specklebem;

PYBIND11_MODULE(_specklebem, m) {
    m.doc() = "SpeckleBem core bindings";
    m.attr("__version__") = VERSION_INFO;

    m.def("has_openmp", &has_openmp);
    m.def("has_cuda", &has_cuda);
    m.def("has_hdf5", &has_hdf5);
    m.def("has_blas_lapack", &has_blas_lapack);

    py::enum_<log::Level>(m, "LogLevel")
        .value("trace", log::Level::trace)
        .value("debug", log::Level::debug)
        .value("info", log::Level::info)
        .value("warn", log::Level::warn)
        .value("error", log::Level::error)
        .value("off", log::Level::off);
    m.def("set_log_level", &log::set_level);

    py::class_<material::Material>(m, "Material")
        .def(py::init<>())
        .def(py::init([](Complex eps_r, Complex mu_r) { return material::Material{eps_r, mu_r}; }),
             py::arg("eps_r"), py::arg("mu_r") = Complex(1, 0))
        .def_readwrite("eps_r", &material::Material::eps_r)
        .def_readwrite("mu_r", &material::Material::mu_r)
        .def_property_readonly("n", &material::Material::refractive_index)
        .def("wavenumber", &material::Material::wavenumber)
        .def("wave_impedance", &material::Material::wave_impedance);
    m.def("vacuum", &material::vacuum);
    m.def("silver_500nm", &material::silver_500nm);
    m.def("silicon_500nm", &material::silicon_500nm);

    py::enum_<formulation::Kind>(m, "Formulation")
        .value("PMCHWT", formulation::Kind::PMCHWT)
        .value("ICTF", formulation::Kind::ICTF)
        .value("MCTF", formulation::Kind::MCTF)
        .value("JMCFIE", formulation::Kind::JMCFIE);
    m.def("recommend_formulation", [](Complex eps_r) {
        auto r = formulation::recommend(eps_r);
        return py::make_tuple(r.kind, r.diagonal_preconditioner);
    });

    // Mesh, excitation, Simulation, post-processing: added in Phases 1-6.
}
