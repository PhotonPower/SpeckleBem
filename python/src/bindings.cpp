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

    m.def("has_openmp", &has_openmp, "True if the core was built with OpenMP.");
    m.def("has_cuda", &has_cuda, "True if the core was built with CUDA.");
    m.def("has_hdf5", &has_hdf5, "True if the core was built with HDF5.");
    m.def("has_blas_lapack", &has_blas_lapack, "True if the core links BLAS/LAPACK.");

    py::enum_<log::Level>(m, "LogLevel", "Log level of the C++ core (spdlog).")
        .value("trace", log::Level::trace)
        .value("debug", log::Level::debug)
        .value("info", log::Level::info)
        .value("warn", log::Level::warn)
        .value("error", log::Level::error)
        .value("off", log::Level::off);
    m.def("set_log_level", &log::set_level, py::arg("level"),
          "Set the log level of the C++ core (a LogLevel value).");

    py::class_<material::Material>(m, "Material", R"doc(
Homogeneous isotropic material, time convention exp(+jwt).

Parameters
----------
eps_r : complex
    Relative permittivity; passive media have Im(eps_r) <= 0 (n + ik optics data must be
    conjugated, docs/06_conventions.md).
mu_r : complex, optional
    Relative permeability (1 for optical materials).

Notes
-----
``Material()`` is vacuum. ``n`` is sqrt(eps_r mu_r) on the branch with Im(n) <= 0.
)doc")
        .def(py::init<>())
        .def(py::init([](Complex eps_r, Complex mu_r) { return material::Material{eps_r, mu_r}; }),
             py::arg("eps_r"), py::arg("mu_r") = Complex(1, 0))
        .def_readwrite("eps_r", &material::Material::eps_r, "Relative permittivity.")
        .def_readwrite("mu_r", &material::Material::mu_r, "Relative permeability.")
        .def_property_readonly("n", &material::Material::refractive_index,
                               "Refractive index sqrt(eps_r mu_r) with Im(n) <= 0.")
        .def("wavenumber", &material::Material::wavenumber, py::arg("omega"),
             "Wavenumber k = omega sqrt(mu eps) [1/m] at the angular frequency omega [rad/s].")
        .def("wave_impedance", &material::Material::wave_impedance, py::arg("omega"),
             "Wave impedance eta = sqrt(mu / eps) [Ohm] (omega [rad/s] is unused).");
    m.def("vacuum", &material::vacuum, "Vacuum (eps_r = mu_r = 1).");
    m.def("silver_500nm", &material::silver_500nm,
          "Ag at 500 nm (Johnson & Christy): eps_r = -9.794 - 0.313j (exp(+jwt)).");
    m.def("silicon_500nm", &material::silicon_500nm,
          "Si at 500 nm (Aspnes & Studna): eps_r = 18.478 - 0.606j (exp(+jwt)).");
    const char* decay_doc = R"doc(
Field decay length delta = wavelength / (2 pi |Im n|) [m] of a plane wave.

Parameters
----------
material : Material
    The medium (n = sqrt(eps_r mu_r)); alternatively ``eps_r`` (complex, mu_r = 1).
wavelength : float
    Vacuum wavelength [m].

Returns
-------
float
    Depth [m] over which the field amplitude falls by 1/e; ``inf`` for a lossless
    dielectric. Only |Im n| enters, so the sign convention of Im(eps_r) does not matter.

Raises
------
ValueError
    For non-finite eps_r / mu_r or a non-positive / non-finite wavelength.
)doc";
    m.def("field_decay_length",
          py::overload_cast<const material::Material&, Real>(&material::field_decay_length),
          py::arg("material"), py::arg("wavelength"), decay_doc);
    m.def("field_decay_length", py::overload_cast<Complex, Real>(&material::field_decay_length),
          py::arg("eps_r"), py::arg("wavelength"), decay_doc);
    py::class_<material::DispersiveMaterial>(m, "DispersiveMaterial", R"doc(
Tabulated refractive index n(wavelength), linearly interpolated (mu_r = 1).

Parameters
----------
wavelengths : array_like, shape (m,)
    Vacuum wavelengths [m]; finite, positive, strictly increasing, m >= 2.
refractive_indices : array_like, shape (m,)
    Complex indices in the exp(+jwt) form n - jk (Im(n) <= 0). Optics data in the form n + ik
    (e.g. refractiveindex.info) must be conjugated first.

Raises
------
ValueError
    For mismatched sizes, invalid wavelengths, non-finite indices or Im(n) > 0.
)doc")
        .def(py::init<VectorXr, VectorXc>(), py::arg("wavelengths"), py::arg("refractive_indices"))
        .def("at_wavelength", &material::DispersiveMaterial::at_wavelength, py::arg("wavelength"),
             R"doc(
Material with eps_r = n(wavelength)^2 at the vacuum wavelength [m].

The table endpoints are inclusive. Raises ValueError for a non-finite wavelength or one
outside the table.
)doc")
        .def("at", &material::DispersiveMaterial::at_wavelength, py::arg("wavelength"),
             "Alias of at_wavelength(wavelength) (vacuum wavelength [m]).");

    py::enum_<formulation::Kind>(m, "Formulation", "Surface integral equation formulation.")
        .value("PMCHWT", formulation::Kind::PMCHWT)
        .value("ICTF", formulation::Kind::ICTF)
        .value("MCTF", formulation::Kind::MCTF)
        .value("JMCFIE", formulation::Kind::JMCFIE);
    m.def(
        "recommend_formulation",
        [](Complex eps_r) {
            auto r = formulation::recommend(eps_r);
            return py::make_tuple(r.kind, r.diagonal_preconditioner);
        },
        py::arg("eps_r"),
        "(Formulation, diagonal_preconditioner: bool) recommended for the object eps_r "
        "(exp(+jwt), Im(eps_r) <= 0).");

    specklebem::python::bind_geometry(m);
    specklebem::python::bind_excitation(m);
    // Simulation, operators, post-processing, I/O: WP14b2.
}
