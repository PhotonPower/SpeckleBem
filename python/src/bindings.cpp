/// pybind11 bindings. Zero-copy NumPy interop via Eigen; the Python package
/// `specklebem` wraps this module with a friendlier API (python/specklebem/).
///
/// Exceptions: pybind11's defaults map std::invalid_argument, std::domain_error and
/// std::length_error to ValueError, std::overflow_error to OverflowError, std::bad_alloc to
/// MemoryError and every other std::exception (std::runtime_error, std::logic_error) to
/// RuntimeError. std::out_of_range (e.g. DispersiveMaterial outside its table) is mapped to
/// ValueError instead of IndexError by a module-local translator.
#include "specklebem/core/config.hpp"
#include "specklebem/core/logging.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/material/material.hpp"

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <stdexcept>

#include "common.hpp"

namespace py = pybind11;
using namespace specklebem;

PYBIND11_MODULE(_specklebem, m) {
    m.doc() = "SpeckleBem core bindings";
    m.attr("__version__") = VERSION_INFO;
    py::register_local_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const std::out_of_range& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        }
    });

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
    m.def("field_decay_length",
          py::overload_cast<const material::Material&, Real>(&material::field_decay_length),
          py::arg("material"), py::arg("wavelength"));
    m.def("field_decay_length", py::overload_cast<Complex, Real>(&material::field_decay_length),
          py::arg("eps_r"), py::arg("wavelength"));
    py::class_<material::DispersiveMaterial>(m, "DispersiveMaterial")
        .def(py::init<VectorXr, VectorXc>(), py::arg("wavelengths"), py::arg("refractive_indices"))
        .def("at_wavelength", &material::DispersiveMaterial::at_wavelength, py::arg("wavelength"))
        .def("at", &material::DispersiveMaterial::at_wavelength, py::arg("wavelength"));

    py::enum_<formulation::Kind>(m, "Formulation")
        .value("PMCHWT", formulation::Kind::PMCHWT)
        .value("ICTF", formulation::Kind::ICTF)
        .value("MCTF", formulation::Kind::MCTF)
        .value("JMCFIE", formulation::Kind::JMCFIE);
    m.def("recommend_formulation", [](Complex eps_r) {
        auto r = formulation::recommend(eps_r);
        return py::make_tuple(r.kind, r.diagonal_preconditioner);
    });

    specklebem::python::bind_geometry(m);
    specklebem::python::bind_excitation(m);
    // Simulation, operators, post-processing, I/O: WP14b2.
}
