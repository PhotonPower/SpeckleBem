/// Bindings of excitation/ (shared_ptr holders, as Simulation takes shared_ptr<Excitation>)
/// and of the Mie reference solution.
#include "specklebem/excitation/angular_spectrum_beam.hpp"
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

using excitation::AngularSpectrumBeam;
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

    py::class_<AngularSpectrumBeam, Excitation, std::shared_ptr<AngularSpectrumBeam>>(
        m, "AngularSpectrumBeam", R"doc(
Rigorous Gaussian beam: a finite sum of exact propagating plane waves (exp(+jwt)).

E(r) = C sum_i W_i exp(-k_t,i^2 w0^2 / 4) p_i exp(-j k k_hat_i . (r - focus)), H = sum of
k_hat_i x E_i / eta, over directions k_hat_i in the cone k_hat . k0_hat > 0 around
k0_hat = R_y(theta_in) z_hat (Gauss-Legendre in the polar angle x trapezoid in azimuth);
k_t = k sin(alpha) is the transverse wavenumber relative to k0_hat and
p_i = e0 - (k_hat_i . e0) k_hat_i the projected central polarisation (not renormalised).
E(focus) . e0 = 1 V/m. Evanescent components are omitted. Solves Maxwell's equations exactly;
the quadrature is refined until a 1.5x finer grid changes E by < tolerance (relative) on probe
points in the ball |r - focus| <= region_radius, outside of which fields are not controlled.

Parameters
----------
wavelength : float
    Vacuum wavelength [m]; k = 2 pi n1 / wavelength of the background.
waist : float
    w0 [m] of the spectrum (1/e^2 intensity radius of the paraxial limit).
polarization : {"p", "s"} or Polarization, optional
    p: e0 = R_y(theta_in) x_hat (xz-plane of incidence); s: e0 = y_hat.
incidence_angle : float, optional
    theta_in [rad], rotation of +z about y; |theta_in| < pi/2.
focus : array_like, shape (3,), optional
    Focus (waist centre, phase reference) [m].
background : Material, optional
    Lossless background medium (default vacuum).
tolerance : float, optional
    Grid check in (0, 1e-2] (default 1e-10).
region_radius : float, optional
    Radius R [m] of the ball around the focus where the fields are controlled; None = 4 w0.
polar_order, azimuth_order : int, optional
    Fixed grid (both >= 1; azimuth rounded up to a multiple of 4); 0 = automatic (default).
max_plane_waves : int, optional
    Cap of the automatic refinement (default 2 000 000).

Raises
------
ValueError
    For waist or wavelength <= 0, a bad polarization, |theta_in| >= pi/2, a non-finite focus,
    a lossy background or invalid quadrature controls.
RuntimeError
    When the automatic refinement does not converge within max_plane_waves.
)doc")
        .def(py::init([](Real wavelength, Real waist, const py::object& polarization,
                         Real incidence_angle, const Vec3& focus, const Material& background,
                         Real tolerance, const py::object& region_radius, int polar_order,
                         int azimuth_order, Index max_plane_waves) {
                 AngularSpectrumBeam::Params p;
                 p.wavelength = wavelength;
                 p.waist_radius = waist;
                 p.polarization = parse_polarization(polarization);
                 p.incidence_angle = incidence_angle;
                 p.focus = focus;
                 p.tolerance = tolerance;
                 if (!region_radius.is_none()) {
                     p.region_radius = region_radius.cast<Real>();
                     if (!(p.region_radius > 0)) {
                         throw py::value_error("region_radius must be > 0 (or None for 4 w0)");
                     }
                 }
                 p.polar_order = polar_order;
                 p.azimuth_order = azimuth_order;
                 p.max_plane_waves = max_plane_waves;
                 py::gil_scoped_release release;
                 return std::make_shared<AngularSpectrumBeam>(p, background);
             }),
             py::arg("wavelength"), py::arg("waist"), py::arg("polarization") = "p",
             py::arg("incidence_angle") = 0.0, py::arg("focus") = Vec3::Zero(),
             py::arg("background") = material::vacuum(), py::kw_only(),
             py::arg("tolerance") = 1e-10, py::arg("region_radius") = py::none(),
             py::arg("polar_order") = 0, py::arg("azimuth_order") = 0,
             py::arg("max_plane_waves") = Index{2'000'000})
        .def(
            "fields",
            [](const AngularSpectrumBeam& b, const py::object& points) {
                // One pass over the plane waves per point for both fields.
                bool single = false;
                const Vertices p = points_from_array(points, "points", &single);
                require_all_finite(p, "points");
                MatrixX3c e(p.rows(), 3);
                MatrixX3c h(p.rows(), 3);
                {
                    py::gil_scoped_release release;
                    for (Index i = 0; i < p.rows(); ++i) {
                        const auto [ei, hi] = b.fields(Vec3(p.row(i).transpose()));
                        e.row(i) = ei.transpose();
                        h.row(i) = hi.transpose();
                    }
                }
                if (single) {
                    return py::make_tuple(Vec3c(e.row(0).transpose()), Vec3c(h.row(0).transpose()));
                }
                return py::make_tuple(std::move(e), std::move(h));
            },
            py::arg("points"),
            R"doc(
E [V/m] and H [A/m] at points [m], exp(+jwt) phasors (same as electric_field and
magnetic_field).

Returns
-------
(E, H) : tuple of ndarray of complex128, shape (n, 3) or (3,)
)doc")
        .def_property_readonly(
            "waist", [](const AngularSpectrumBeam& b) { return b.params().waist_radius; },
            "w0 [m] of the spectrum exp(-k_t^2 w0^2 / 4).")
        .def_property_readonly(
            "incidence_angle",
            [](const AngularSpectrumBeam& b) { return b.params().incidence_angle; },
            "Incidence angle theta_in [rad].")
        .def_property_readonly(
            "focus", [](const AngularSpectrumBeam& b) { return b.params().focus; }, "Focus [m].")
        .def_property_readonly(
            "polarization", [](const AngularSpectrumBeam& b) { return b.params().polarization; },
            "Polarization.P or Polarization.S.")
        .def_property_readonly(
            "tolerance", [](const AngularSpectrumBeam& b) { return b.params().tolerance; },
            "Requested grid check (relative).")
        .def_property_readonly("num_plane_waves", &AngularSpectrumBeam::num_plane_waves,
                               "Number of plane waves in the sum.")
        .def_property_readonly("polar_order", &AngularSpectrumBeam::polar_order,
                               "Gauss-Legendre nodes in the polar angle.")
        .def_property_readonly("azimuth_order", &AngularSpectrumBeam::azimuth_order,
                               "Trapezoid nodes in azimuth (a multiple of 4).")
        .def_property_readonly("grid_change", &AngularSpectrumBeam::grid_change,
                               "Achieved relative change of E against the 1.5x finer grid.")
        .def_property_readonly("max_polar_angle", &AngularSpectrumBeam::max_polar_angle,
                               "Spectrum truncation angle alpha_max [rad] (<= pi/2).")
        .def_property_readonly("region_radius", &AngularSpectrumBeam::region_radius,
                               "Radius [m] of the controlled ball around the focus.")
        .def_property_readonly("power", &AngularSpectrumBeam::power,
                               "Power [W] through any plane normal to k0_hat (Parseval); "
                               "pi w0^2 / (4 eta) in the paraxial limit.");

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
