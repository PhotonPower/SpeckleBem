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
/// ValueError for non-finite angles.
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
    require_all_finite(Eigen::Map<const VectorXr>(tp, n), "theta");
    require_all_finite(Eigen::Map<const VectorXr>(pp, n), "phi");
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
    py::enum_<Polarization>(m, "Polarization",
                            "Gaussian-beam polarisation: P (E in the xz-plane of incidence), "
                            "S (E along y).")
        .value("P", Polarization::P)
        .value("S", Polarization::S);

    py::class_<Excitation, std::shared_ptr<Excitation>>(m, "Excitation",
                                                        R"doc(
Incident field (abstract base of PlaneWave and GaussianBeam), time convention exp(+jwt).

Fields are evaluated in the background medium R1 at points [m]; E in V/m, H in A/m.
)doc")
        .def(
            "electric_field",
            [](const Excitation& e, const py::object& points) {
                return evaluate_at_points(points,
                                          [&e](const Vec3& r) { return e.electric_field(r); });
            },
            py::arg("points"),
            R"doc(
Incident electric field E [V/m] at points [m], exp(+jwt) phasor.

Parameters
----------
points : array_like, shape (n, 3) or (3,)
    Real, finite points [m]; ValueError otherwise.

Returns
-------
ndarray of complex128, shape (n, 3) or (3,)
)doc")
        .def(
            "magnetic_field",
            [](const Excitation& e, const py::object& points) {
                return evaluate_at_points(points,
                                          [&e](const Vec3& r) { return e.magnetic_field(r); });
            },
            py::arg("points"),
            R"doc(
Incident magnetic field H [A/m] at points [m], exp(+jwt) phasor.

Parameters
----------
points : array_like, shape (n, 3) or (3,)
    Real, finite points [m]; ValueError otherwise.

Returns
-------
ndarray of complex128, shape (n, 3) or (3,)
)doc")
        .def_property_readonly("omega", &Excitation::omega, "Angular frequency omega [rad/s].")
        .def_property_readonly("wavelength", &Excitation::wavelength, "Vacuum wavelength [m].")
        .def_property_readonly(
            "background", [](const Excitation& e) { return e.background(); },
            "Background Material of R1 (lossless).");

    py::class_<PlaneWave, Excitation, std::shared_ptr<PlaneWave>>(m, "PlaneWave", R"doc(
Plane wave E(r) = e0 exp(-j k k_hat . r), H = k_hat x E / eta (exp(+jwt)).

Parameters
----------
wavelength : float
    Vacuum wavelength [m]; k and eta are those of the background.
direction : array_like, shape (3,)
    Propagation direction k_hat (normalised by the constructor).
polarization : array_like, shape (3,)
    Complex amplitude e0 [V/m], transverse to k_hat.
background : Material, optional
    Lossless background medium (default vacuum).

Raises
------
ValueError
    For a non-positive wavelength, a zero direction or e0, a non-transverse e0 or a lossy
    background.
)doc")
        .def(py::init<Real, const Vec3&, const Vec3c&, Material>(), py::arg("wavelength"),
             py::arg("direction"), py::arg("polarization"),
             py::arg("background") = material::vacuum());

    py::class_<GaussianBeam, Excitation, std::shared_ptr<GaussianBeam>>(m, "GaussianBeam", R"doc(
Paraxial Gaussian beam with first-order longitudinal fields, 1 V/m on axis at the focus.

Parameters
----------
wavelength : float
    Vacuum wavelength [m].
waist : float
    1/e^2 intensity radius w0 [m] at the focus.
polarization : {"p", "s"} or Polarization, optional
    p: E in the xz-plane of incidence; s: E along y (case-insensitive strings).
incidence_angle : float, optional
    theta_in [rad], rotation of +z about y; |theta_in| < pi/2 (the beam travels towards +z).
focus : array_like, shape (3,), optional
    Waist centre [m].
background : Material, optional
    Lossless background medium (default vacuum).

Raises
------
ValueError
    For waist or wavelength <= 0, a bad polarization, |theta_in| >= pi/2, a non-finite focus
    or a lossy background.
)doc")
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
        .def_property_readonly(
            "waist", [](const GaussianBeam& b) { return b.params().waist_radius; },
            "1/e^2 intensity radius w0 [m].")
        .def_property_readonly(
            "incidence_angle", [](const GaussianBeam& b) { return b.params().incidence_angle; },
            "Incidence angle theta_in [rad].")
        .def_property_readonly(
            "focus", [](const GaussianBeam& b) { return b.params().focus; }, "Focus [m].")
        .def_property_readonly(
            "polarization", [](const GaussianBeam& b) { return b.params().polarization; },
            "Polarization.P or Polarization.S.");

    py::class_<MieSolution>(m, "Mie", R"doc(
Mie series of a sphere at the origin under an x-polarised plane wave travelling in +z.

E_inc = x_hat 1 V/m exp(-j k1 z), time convention exp(+jwt).

Parameters
----------
radius : float
    Sphere radius [m].
wavelength : float
    Vacuum wavelength [m].
material : Material
    Sphere material (Im(eps_r) <= 0, mu_r = 1).
exterior : Material, optional
    Lossless, non-magnetic background (default vacuum).
n_max : int, optional
    Number of terms; 0 selects ceil(x + 11 x^(1/3) + 1), x = k1 radius.
)doc")
        .def(py::init([](Real radius, Real wavelength, const Material& material,
                         const Material& exterior, int n_max) {
                 return MieSolution(
                     reference::MieParams{radius, wavelength, material, exterior, n_max});
             }),
             py::arg("radius"), py::arg("wavelength"), py::arg("material"),
             py::arg("exterior") = material::vacuum(), py::arg("n_max") = 0)
        .def("bistatic_rcs", &bistatic_rcs, py::arg("theta"), py::arg("phi"), R"doc(
Bistatic radar cross section sigma(theta, phi) [m^2].

theta [rad] from +z (pi = backscattering), phi [rad] from +x; finite, NumPy-broadcast.
Returns a float for scalar input, else a float64 array of the broadcast shape.
)doc")
        .def(
            "scattered_E",
            [](const MieSolution& s, const py::object& points) {
                return evaluate_at_points(points, [&s](const Vec3& r) { return s.scattered_E(r); });
            },
            py::arg("points"),
            R"doc(
Scattered electric field E_s [V/m] at points [m], exp(+jwt) phasor;
exact series (not the far field).

Parameters
----------
points : array_like, shape (n, 3) or (3,)
    Real, finite points [m], outside the sphere; ValueError otherwise.

Returns
-------
ndarray of complex128, shape (n, 3) or (3,)
)doc")
        .def(
            "scattered_H",
            [](const MieSolution& s, const py::object& points) {
                return evaluate_at_points(points, [&s](const Vec3& r) { return s.scattered_H(r); });
            },
            py::arg("points"),
            R"doc(
Scattered magnetic field H_s [A/m] at points [m], exp(+jwt) phasor;
exact series (not the far field).

Parameters
----------
points : array_like, shape (n, 3) or (3,)
    Real, finite points [m], outside the sphere; ValueError otherwise.

Returns
-------
ndarray of complex128, shape (n, 3) or (3,)
)doc")
        .def(
            "internal_E",
            [](const MieSolution& s, const py::object& points) {
                return evaluate_at_points(points, [&s](const Vec3& r) { return s.internal_E(r); });
            },
            py::arg("points"),
            R"doc(
Electric field inside the sphere [V/m] at points [m], exp(+jwt) phasor. Exact series.

Parameters
----------
points : array_like, shape (n, 3) or (3,)
    Real, finite points [m], inside the sphere; ValueError otherwise.

Returns
-------
ndarray of complex128, shape (n, 3) or (3,)
)doc")
        .def("scattering_cross_section", &MieSolution::scattering_cross_section,
             "Scattering cross section C_sca [m^2].")
        .def("extinction_cross_section", &MieSolution::extinction_cross_section,
             "Extinction cross section C_ext [m^2].")
        .def_property_readonly(
            "a_n", [](const MieSolution& s) -> const VectorXc& { return s.a_n(); },
            py::return_value_policy::reference_internal,
            "Mie coefficients a_n, n = 1..n_max (exp(+jwt): conjugated Bohren & Huffman); "
            "read-only view.")
        .def_property_readonly(
            "b_n", [](const MieSolution& s) -> const VectorXc& { return s.b_n(); },
            py::return_value_policy::reference_internal,
            "Mie coefficients b_n, n = 1..n_max (exp(+jwt): conjugated Bohren & Huffman); "
            "read-only view.");
}

}  // namespace specklebem::python
