/// Bindings of excitation/ (shared_ptr holders, as Simulation takes shared_ptr<Excitation>)
/// and of the Mie reference solution.
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/reference/mie.hpp"

#include <pybind11/complex.h>

#include <cctype>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common.hpp"

namespace specklebem::python {

namespace {

using excitation::Excitation;
using excitation::GaussianBeam;
using excitation::PlaneWave;
using excitation::Polarization;
using material::Material;
using reference::MieSolution;

/// "p" / "s" (any case) or a Polarization value; ValueError otherwise.
Polarization parse_polarization(const py::object& obj) {
    if (py::isinstance<py::str>(obj)) {
        std::string s = obj.cast<std::string>();
        for (char& c : s) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        if (s == "p") {
            return Polarization::P;
        }
        if (s == "s") {
            return Polarization::S;
        }
    } else if (py::isinstance<Polarization>(obj)) {
        return obj.cast<Polarization>();
    }
    throw py::value_error("polarization: expected \"p\", \"s\" or a Polarization value");
}

/// sigma(theta, phi) [m^2], NumPy-broadcast over theta and phi [rad]; a float for scalars.
py::object bistatic_rcs(const MieSolution& s, const py::object& theta, const py::object& phi) {
    using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;
    const py::tuple b = py::module_::import("numpy").attr("broadcast_arrays")(
        real_array(theta, "theta"), real_array(phi, "phi"));
    const auto t = Arr::ensure(b[0]);
    const auto p = Arr::ensure(b[1]);
    py::array_t<double> out(std::vector<py::ssize_t>(t.shape(), t.shape() + t.ndim()));
    const double* tp = t.data();
    const double* pp = p.data();
    double* op = out.mutable_data();
    const auto n = static_cast<Index>(t.size());
    {
        py::gil_scoped_release release;
        for (Index i = 0; i < n; ++i) {
            op[i] = s.bistatic_rcs(tp[i], pp[i]);
        }
    }
    if (t.ndim() == 0) {
        return py::float_(op[0]);
    }
    return py::object(std::move(out));
}

}  // namespace

void bind_excitation(py::module_& m) {
    py::enum_<Polarization>(m, "Polarization")
        .value("P", Polarization::P)
        .value("S", Polarization::S);

    py::class_<Excitation, std::shared_ptr<Excitation>>(m, "Excitation",
                                                        "Incident field (abstract).")
        .def(
            "electric_field",
            [](const Excitation& e, const py::object& points) {
                return evaluate_at_points(points,
                                          [&e](const Vec3& r) { return e.electric_field(r); });
            },
            py::arg("points"))
        .def(
            "magnetic_field",
            [](const Excitation& e, const py::object& points) {
                return evaluate_at_points(points,
                                          [&e](const Vec3& r) { return e.magnetic_field(r); });
            },
            py::arg("points"))
        .def_property_readonly("omega", &Excitation::omega)
        .def_property_readonly("wavelength", &Excitation::wavelength)
        .def_property_readonly("background", [](const Excitation& e) { return e.background(); });

    py::class_<PlaneWave, Excitation, std::shared_ptr<PlaneWave>>(m, "PlaneWave")
        .def(py::init<Real, const Vec3&, const Vec3c&, Material>(), py::arg("wavelength"),
             py::arg("direction"), py::arg("polarization"),
             py::arg("background") = material::vacuum());

    py::class_<GaussianBeam, Excitation, std::shared_ptr<GaussianBeam>>(m, "GaussianBeam")
        .def(py::init([](Real wavelength, Real waist, const py::object& polarization,
                         Real incidence_angle, const Vec3& focus, const Material& background) {
                 GaussianBeam::Params p;
                 p.wavelength = wavelength;
                 p.waist_radius = waist;
                 p.polarization = parse_polarization(polarization);
                 p.incidence_angle = incidence_angle;
                 p.focus = focus;
                 return std::make_shared<GaussianBeam>(p, background);
             }),
             py::arg("wavelength"), py::arg("waist"), py::arg("polarization") = "p",
             py::arg("incidence_angle") = 0.0, py::arg("focus") = Vec3::Zero(),
             py::arg("background") = material::vacuum())
        .def_property_readonly("waist",
                               [](const GaussianBeam& b) { return b.params().waist_radius; })
        .def_property_readonly("incidence_angle",
                               [](const GaussianBeam& b) { return b.params().incidence_angle; })
        .def_property_readonly("focus", [](const GaussianBeam& b) { return b.params().focus; })
        .def_property_readonly("polarization",
                               [](const GaussianBeam& b) { return b.params().polarization; });

    py::class_<MieSolution>(m, "Mie", "Mie series of a sphere at the origin, x-polarised +z wave.")
        .def(py::init([](Real radius, Real wavelength, const Material& material,
                         const Material& exterior, int n_max) {
                 return MieSolution(
                     reference::MieParams{radius, wavelength, material, exterior, n_max});
             }),
             py::arg("radius"), py::arg("wavelength"), py::arg("material"),
             py::arg("exterior") = material::vacuum(), py::arg("n_max") = 0)
        .def("bistatic_rcs", &bistatic_rcs, py::arg("theta"), py::arg("phi"))
        .def(
            "scattered_E",
            [](const MieSolution& s, const py::object& points) {
                return evaluate_at_points(points, [&s](const Vec3& r) { return s.scattered_E(r); });
            },
            py::arg("points"))
        .def(
            "scattered_H",
            [](const MieSolution& s, const py::object& points) {
                return evaluate_at_points(points, [&s](const Vec3& r) { return s.scattered_H(r); });
            },
            py::arg("points"))
        .def(
            "internal_E",
            [](const MieSolution& s, const py::object& points) {
                return evaluate_at_points(points, [&s](const Vec3& r) { return s.internal_E(r); });
            },
            py::arg("points"))
        .def("scattering_cross_section", &MieSolution::scattering_cross_section)
        .def("extinction_cross_section", &MieSolution::extinction_cross_section)
        .def_property_readonly(
            "a_n", [](const MieSolution& s) -> const VectorXc& { return s.a_n(); },
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "b_n", [](const MieSolution& s) -> const VectorXc& { return s.b_n(); },
            py::return_value_policy::reference_internal);
}

}  // namespace specklebem::python
