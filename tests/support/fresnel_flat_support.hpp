#pragma once
/// @file fresnel_flat_support.hpp
/// Flat-interface limit of the truncated rough surface (WP22c, former WP12; docs/01 Phase 2/4 DoD
/// "Flat interface limit", docs/05 row "Flat box, tapered beam, 0 deg and 45 deg | Fresnel r_p,
/// r_s | rel on |r|^2 | < 1 %"). Shared by the study executable benchmarks/fresnel_flat.cpp, the
/// fast reference checks in tests/validation/test_flat_interface_fresnel.cpp and the
/// memory-guarded case in tests/validation_large/test_flat_interface_fresnel_large.cpp, so it must
/// not depend on Catch2.
///
/// Reference (exp(+jwt), docs/06):
///  * Plane-wave Fresnel coefficients of the interface between a lossless background of index n1
///    (R1, z < 0) and the object eps_r2 (R2, z > 0, mu_r = 1). With kz2 = sqrt(eps2 - n1^2
///    sin^2 theta) (= n2 cos theta_t, branch Im kz2 <= 0: the transmitted wave decays towards +z)
///      r_s = (n1 cos theta - kz2) / (n1 cos theta + kz2),
///      r_p = (eps2 cos theta - n1 kz2) / (eps2 cos theta + n1 kz2)
///    (r_p for the p amplitude along k x s_hat; with this sign r_p = -r_s at normal incidence and
///    r_p = r_s^2 at 45 deg, the Abeles identity). Only |r|^2 enters the comparison.
///  * Beam reflectance R_beam: the rigorous AngularSpectrumBeam is a quadrature over plane waves
///    k_hat_i with real amplitudes a_i (plane_wave_amplitudes()). Each plane wave reflects at an
///    infinite interface with its own angle theta_i = acos(k_hat_i . z_hat) and its local s / p
///    split with respect to the interface normal (s_hat = k_hat x z_hat / |k_hat x z_hat|, p_hat =
///    s_hat x k_hat); reflected s and p waves are orthogonal, and plane waves of different
///    directions carry power independently (Parseval over the plane z = 0), so
///      R_beam = sum_i P_i (|r_s(theta_i)|^2 |a_i . s_hat|^2 + |r_p(theta_i)|^2 |a_i . p_hat|^2)
///               / |a_i|^2 / sum_i P_i,
///    with the power weight of wave i, P_i = W_i A_i^2 |p_i|^2 cos(alpha_i) (the summand of
///    AngularSpectrumBeam::power()), recovered from the amplitudes as P_i ~ |a_i| A_i |p_i|
///    cos(alpha_i) (a_i = W_i A_i p_i / norm; alpha_i the angle to the beam axis k0_hat, A_i =
///    exp(-k^2 sin^2(alpha_i) w0^2 / 4), p_i = e0_hat - (k_hat_i . e0_hat) k_hat_i). R_beam
///    differs from the plane-wave |r(theta_0)|^2 at the central angle by O((lambda / (pi w0))^2).
///
/// Simulation: flat patch (sigma = 0) of edge L, grid spacing L / round(L / h) (h = 50 nm), closed
/// by the box of rough_surface_box_params() (ADR 0006 with the 2026-10-10 amendment: depth
/// default_box_depth, coarse cells <= lambda_1 / 5, Si fine band 3 delta), rigorous beam
/// (AngularSpectrumBeam, waist w0 <= L / 4, focus at the origin = surface centre, incidence about
/// y, region_radius >= the farthest mesh vertex and >= 5 w0 / cos(theta) for the flux check),
/// Simulation with compression "mlfmm" (d0, automatic leaf rule and exact-part budget),
/// formulation per formulation::recommend, full GMRES.
///  * Reflected power: P_refl = (1 / 2 eta1) int_{k_z < 0} |F|^2 dOmega of the scattered far field
///    (post::far_field), Gauss-Legendre in theta (from -z) x trapezoid in phi (hemisphere_grid);
///    checked with half the nodes per direction, and there with Dunavant degree 8 instead of 6.
///    Diagnostics: the share inside the specular cone of half-angle 3 lambda / (pi w0) and at
///    grazing angles theta > 80 deg (edge diffraction of the box).
///  * Incident power: AngularSpectrumBeam::power(), checked against the numerical flux of the
///    incident Poynting vector through z = 0 (trapezoid over |x|, |y| <= X, X = R / sqrt(2));
///    the edge loss is 1 - (flux through the L x L patch) / (flux through the square).
///  * R_sim = P_refl / P_inc; diagnostics: absorbed power P_abs = 1/2 Re oint (n x M) . J* dS from
///    the currents (as tests/support/mie_dense_test_support.hpp), forward-hemisphere scattered
///    power, and the balance 1 - R_sim - A_sim.
#include "specklebem/basis/rwg.hpp"
#include "specklebem/compression/mlfmm/far_operator.hpp"
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/near_field.hpp"
#include "specklebem/compression/mlfmm/octree.hpp"
#include "specklebem/compression/mlfmm/patterns.hpp"
#include "specklebem/core/types.hpp"
#include "specklebem/excitation/angular_spectrum_beam.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/geometry/mesh.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/kernels/quadrature.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/sparse_operator.hpp"
#include "specklebem/postprocessing/fields.hpp"
#include "specklebem/simulation.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "system_memory.hpp"

namespace fresnel_flat {

using namespace specklebem;

inline constexpr Real kLambda = 500e-9;
inline constexpr Real kDeg = constants::pi / 180.0;

// ---------------------------------------------------------------------------------------------
// Reference

/// Plane-wave Fresnel reflection coefficients (see the file comment for the conventions).
struct FresnelCoefficients {
    Complex r_s;
    Complex r_p;
};

/// Fresnel coefficients for a plane wave from the lossless background (real index n1 > 0) onto
/// the object eps2 (mu_r = 1) at cos(theta) = cos_theta in [0, 1].
/// @throws std::invalid_argument for n1 not finite and > 0, cos_theta outside [0, 1], a
///         non-finite eps2 or an active object (Im eps2 > 0, docs/06).
inline FresnelCoefficients fresnel(Real n1, Complex eps2, Real cos_theta) {
    if (!(std::isfinite(n1) && n1 > 0.0))
        throw std::invalid_argument("fresnel: n1 must be finite and > 0");
    if (!(cos_theta >= 0.0 && cos_theta <= 1.0))
        throw std::invalid_argument("fresnel: cos_theta must lie in [0, 1]");
    if (!(std::isfinite(eps2.real()) && std::isfinite(eps2.imag())) || eps2.imag() > 0.0)
        throw std::invalid_argument("fresnel: eps2 must be finite with Im(eps2) <= 0 (exp(+jwt))");
    const Real sin2 = 1.0 - cos_theta * cos_theta;
    Complex kz2 = std::sqrt(eps2 - n1 * n1 * sin2);
    if (kz2.imag() > 0.0 || (kz2.imag() == 0.0 && kz2.real() < 0.0))
        kz2 = -kz2;
    const Real kz1 = n1 * cos_theta;
    FresnelCoefficients c;
    c.r_s = (kz1 - kz2) / (kz1 + kz2);
    c.r_p = (eps2 * cos_theta - n1 * kz2) / (eps2 * cos_theta + n1 * kz2);
    return c;
}

/// |r|^2 of the polarisation `pol` (docs/06: p in the plane of incidence) at incidence angle
/// theta [rad].
inline Real fresnel_reflectance(Real n1, Complex eps2, Real theta, excitation::Polarization pol) {
    const FresnelCoefficients c = fresnel(n1, eps2, std::cos(theta));
    return std::norm(pol == excitation::Polarization::P ? c.r_p : c.r_s);
}

/// Beam reflectance of an infinite flat interface z = 0 (see the file comment).
struct BeamReflectance {
    Real reflectance = 0;  ///< R_beam
    /// Power-weighted share of the s component (local s/p split w.r.t. the interface normal).
    Real s_share = 0;
    Real fresnel_central = 0;  ///< |r_pol(theta_in)|^2 of the central plane wave
    Index plane_waves = 0;
    /// Power share of the plane waves with k_z <= 0 (outer cone components of an oblique beam,
    /// docs/06): they never reach the interface and count as not reflected.
    Real away_share = 0;
};

/// R_beam of `beam` (lossless background of index n1) on the object eps2 (mu_r = 1),
/// normalised by the power of all plane waves (AngularSpectrumBeam::power()).
inline BeamReflectance beam_reflectance(const excitation::AngularSpectrumBeam& beam, Real n1,
                                        Complex eps2) {
    const excitation::AngularSpectrumBeam::Params& bp = beam.params();
    const Real theta = bp.incidence_angle;
    const Vec3 k0_hat(std::sin(theta), 0.0, std::cos(theta));
    const Vec3 u_hat(std::cos(theta), 0.0, -std::sin(theta));
    const Vec3 e0_hat =
        bp.polarization == excitation::Polarization::P ? u_hat : Vec3(Vec3::UnitY());
    const Real k = 2.0 * constants::pi * n1 / bp.wavelength;
    const Real w0 = bp.waist_radius;
    const std::vector<Vec3> dirs = beam.plane_wave_directions();
    const std::vector<Vec3> amps = beam.plane_wave_amplitudes();
    Real sum_p = 0, sum_r = 0, sum_s = 0, sum_away = 0;
    for (std::size_t i = 0; i < dirs.size(); ++i) {
        const Vec3& kh = dirs[i];
        const Vec3& a = amps[i];
        const Real a_norm = a.norm();
        if (a_norm == 0.0)
            continue;
        const Real ca = std::clamp(kh.dot(k0_hat), -1.0, 1.0);
        const Real sa2 = 1.0 - ca * ca;
        const Real amp = std::exp(-0.25 * k * k * sa2 * w0 * w0);
        const Vec3 pol = e0_hat - kh.dot(e0_hat) * kh;
        const Real power = a_norm * amp * pol.norm() * ca;  // ~ W A^2 |p|^2 cos(alpha)
        sum_p += power;
        if (!(kh.z() > 0.0)) {
            sum_away += power;
            continue;
        }
        Vec3 s_hat = kh.cross(Vec3::UnitZ());
        const Real s_norm = s_hat.norm();
        s_hat = s_norm > 1e-12 ? Vec3(s_hat / s_norm) : Vec3(Vec3::UnitY());
        const Vec3 p_hat = s_hat.cross(kh);
        const Real fs = a.dot(s_hat) * a.dot(s_hat) / (a_norm * a_norm);
        const Real fp = a.dot(p_hat) * a.dot(p_hat) / (a_norm * a_norm);
        const FresnelCoefficients c = fresnel(n1, eps2, std::min(kh.z(), 1.0));
        sum_s += power * fs;
        sum_r += power * (std::norm(c.r_s) * fs + std::norm(c.r_p) * fp);
    }
    BeamReflectance r;
    r.reflectance = sum_r / sum_p;
    r.s_share = sum_s / sum_p;
    r.fresnel_central = fresnel_reflectance(n1, eps2, theta, bp.polarization);
    r.plane_waves = static_cast<Index>(dirs.size());
    r.away_share = sum_away / sum_p;
    return r;
}

/// Time-averaged power of the incident field through the plane z = 0.
struct PlaneFlux {
    Real square = 0;  ///< through |x|, |y| <= half_extent [W]
    Real patch = 0;   ///< through |x|, |y| <= L / 2 [W]
};

/// Trapezoid sum of 1/2 Re(E x H*) . z_hat over a grid x_i = -X + i (2 X / n) (n = 2 ceil(X /
/// spacing) cells, so that the integration bounds are grid lines) on [-X, X]^2.
inline Real flux_z0_square(const excitation::Excitation& inc, Real X, Real spacing) {
    const auto half_cells = static_cast<Index>(std::ceil(X / spacing));
    const Index n = 2 * half_cells;
    const Real h = 2.0 * X / static_cast<Real>(n);
    Real sum = 0;
    for (Index i = 0; i <= n; ++i) {
        const Real x = -X + h * static_cast<Real>(i);
        const Real wx = (i == 0 || i == n) ? 0.5 : 1.0;
        for (Index j = 0; j <= n; ++j) {
            const Real y = -X + h * static_cast<Real>(j);
            const Real wy = (j == 0 || j == n) ? 0.5 : 1.0;
            const auto [e, hf] = inc.fields(Vec3(x, y, 0.0));
            // Unconjugated cross product with conj(H) (Eigen's complex cross conjugates).
            const Complex sz = e.x() * std::conj(hf.y()) - e.y() * std::conj(hf.x());
            sum += wx * wy * 0.5 * sz.real();
        }
    }
    return sum * h * h;
}

/// Incident flux through z = 0 over the square |x|, |y| <= half_extent and the L x L patch.
inline PlaneFlux incident_flux_z0(const excitation::Excitation& inc, Real L, Real half_extent,
                                  Real spacing) {
    PlaneFlux f;
    f.square = flux_z0_square(inc, half_extent, spacing);
    f.patch = flux_z0_square(inc, 0.5 * L, spacing);
    return f;
}

// ---------------------------------------------------------------------------------------------
// Case, geometry, beam

/// One case of the study.
struct Case {
    std::string material = "ag";  ///< "ag" (silver_500nm) or "si" (silicon_500nm)
    Real L = 4e-6;                ///< patch edge [m] (a multiple of mesh_size)
    Real waist = 1e-6;            ///< w0 [m] (<= L / 4, check_beam_waist)
    Real theta_deg = 0.0;         ///< incidence angle about y [deg]
    excitation::Polarization pol = excitation::Polarization::P;
    Real mesh_size = 50e-9;  ///< top-face spacing h [m]
    Real digits = 3.0;       ///< MLFMM accuracy digits d0
    Real tolerance = 1e-4;   ///< GMRES relative residual
    int max_iter = 6000;
    int restart = 0;  ///< 0: full GMRES
    /// Midpoint spacing in theta of the reflection-hemisphere grid [deg] (phi: twice that).
    Real ff_dtheta_deg = 0.5;
    int ff_degree = 6;  ///< Dunavant degree of the far-field source quadrature
    /// Empty: formulation::recommend (Ag: ICTF + left Jacobi, Si: ICTF).
    std::optional<formulation::Kind> formulation;
    std::optional<bool> jacobi;
};

/// Object material of a case. @throws std::invalid_argument for an unknown name.
inline material::Material object_material(const std::string& name) {
    if (name == "ag")
        return material::silver_500nm();
    if (name == "si")
        return material::silicon_500nm();
    throw std::invalid_argument("fresnel_flat: material must be \"ag\" or \"si\", got \"" + name +
                                "\"");
}

/// Case label, e.g. "Ag L = 4 um, w0 = 1 um, 0 deg, p".
inline std::string label(const Case& c) {
    std::ostringstream os;
    os << (c.material == "ag" ? "Ag" : (c.material == "si" ? "Si" : c.material))
       << " L = " << c.L * 1e6 << " um, w0 = " << c.waist * 1e6 << " um, " << c.theta_deg
       << " deg, " << (c.pol == excitation::Polarization::P ? "p" : "s") << ", d0 = " << c.digits
       << ", tol = " << c.tolerance;
    return os.str();
}

/// Flat closed box of a case.
struct Geometry {
    geometry::TriangleMesh mesh;
    RoughBoxParams box;
    geometry::detail::BoxGrading grading;
    Index top_triangles = 0;
    Real spacing = 0;  ///< actual grid spacing L / round(L / h)
    Real r_max = 0;    ///< largest vertex distance from the origin (the focus) [m]
};

/// Flat height map (sigma = 0) of edge L with n = round(L / h) + 1 points per axis.
inline geometry::HeightMap flat_height_map(Real L, Real h) {
    const auto cells = static_cast<Index>(std::llround(L / h));
    if (cells < 2)
        throw std::invalid_argument("flat_height_map: L must be at least 2 h");
    geometry::HeightMap m;
    m.dx = L / static_cast<Real>(cells);
    m.dy = m.dx;
    m.z = MatrixXr::Zero(cells + 1, cells + 1);
    return m;
}

/// Mesh of a case: the flat patch closed by the box of rough_surface_box_params (sigma = 0).
inline Geometry make_geometry(const Case& c) {
    const material::Material object = object_material(c.material);
    const geometry::HeightMap hm = flat_height_map(c.L, c.mesh_size);
    const RoughBoxParams box =
        rough_surface_box_params(object, material::vacuum(), kLambda, 0.0, c.mesh_size);
    Geometry g{geometry::make_mesh_from_height_map(hm, box.box_depth, box.box_mesh_size,
                                                   box.box_fine_depth, box.exterior_wavelength),
               box,
               geometry::detail::box_grading(hm, box.box_depth, box.box_mesh_size,
                                             box.box_fine_depth, box.exterior_wavelength),
               0,
               hm.dx,
               0.0};
    const Index cells = hm.z.rows() - 1;
    g.top_triangles = 2 * cells * cells;
    for (Index v = 0; v < g.mesh.num_vertices(); ++v)
        g.r_max = std::max(g.r_max, g.mesh.vertices().row(v).norm());
    return g;
}

/// Region radius of the beam: covers the mesh (Simulation requires it) and the square of the
/// flux check, half-width >= 3.5 w0 / cos(theta) (beam intensity <= exp(-24) at its edge).
inline Real beam_region_radius(const Case& c, Real r_max) {
    const Real flux = 5.0 * c.waist / std::cos(c.theta_deg * kDeg);
    return std::max(1.02 * r_max, flux);
}

inline std::shared_ptr<excitation::AngularSpectrumBeam> make_beam(const Case& c, Real r_max) {
    check_beam_waist(c.L, c.waist);
    excitation::AngularSpectrumBeam::Params bp;
    bp.wavelength = kLambda;
    bp.waist_radius = c.waist;
    bp.focus = Vec3::Zero();
    bp.incidence_angle = c.theta_deg * kDeg;
    bp.polarization = c.pol;
    bp.region_radius = beam_region_radius(c, r_max);
    return std::make_shared<excitation::AngularSpectrumBeam>(bp, material::vacuum());
}

inline SimulationConfig make_config(const Case& c) {
    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.exterior = material::vacuum();
    cfg.object = object_material(c.material);
    cfg.compression = "mlfmm";
    cfg.mlfmm.accuracy_digits = c.digits;
    cfg.formulation = c.formulation;
    cfg.diagonal_preconditioner = c.jacobi;
    cfg.gmres.tolerance = c.tolerance;
    cfg.gmres.max_iter = c.max_iter;
    cfg.gmres.restart = c.restart;
    cfg.gmres.verbose = true;
    return cfg;
}

/// Beam reference of a case (no simulation): R_beam, Fresnel at the central angle, power()
/// against the numerical flux and the edge loss.
struct Reference {
    BeamReflectance beam;
    Real power = 0;             ///< AngularSpectrumBeam::power() [W]
    PlaneFlux flux;             ///< numerical flux through z = 0
    Real flux_half_extent = 0;  ///< X of the flux square [m]
    Real edge_loss = 0;         ///< 1 - flux.patch / flux.square
    Real region_radius = 0;
    Real max_polar_angle = 0;  ///< [rad]
    Real grid_change = 0;
    Real build_s = 0;  ///< beam construction [s]
    Real flux_s = 0;   ///< flux check [s]
};

inline Reference reference(const Case& c, const excitation::AngularSpectrumBeam& beam,
                           Real build_s) {
    using Clock = std::chrono::steady_clock;
    Reference r;
    r.build_s = build_s;
    r.beam = beam_reflectance(beam, 1.0, object_material(c.material).eps_r);
    r.power = beam.power();
    r.region_radius = beam.region_radius();
    r.max_polar_angle = beam.max_polar_angle();
    r.grid_change = beam.grid_change();
    r.flux_half_extent = beam.region_radius() / std::sqrt(2.0);
    const auto t0 = Clock::now();
    r.flux = incident_flux_z0(beam, c.L, r.flux_half_extent, kLambda / 8.0);
    r.flux_s = std::chrono::duration<Real>(Clock::now() - t0).count();
    r.edge_loss = 1.0 - r.flux.patch / r.flux.square;
    return r;
}

// ---------------------------------------------------------------------------------------------
// Estimate and run

/// Mesh, octree and memory estimates of a case, without assembly.
struct Estimate {
    Index triangles = 0;
    Index top_triangles = 0;
    Index unknowns = 0;  ///< 2N
    Real depth = 0;
    Real fine_depth = 0;  ///< 0: no fine band
    Index box_levels = 0;
    Index fine_rows = 0;
    Real r_max = 0;
    Real max_support_radius = 0;
    int levels = 0;
    Real leaf_edge = 0;
    Index leaves = 0;
    std::size_t near_bytes = 0;     ///< mlfmm::estimate_near_bytes
    std::size_t pattern_bytes = 0;  ///< leaf patterns of the expanded regions
    std::array<int, 2> leaf_order{};
    /// Setup peak model of the WP22b1 scaling study (2 near + 1.1 patterns + 1 GB; the Ag exact
    /// part measured <= 0.8 GB there is covered by the constant), without the Krylov basis.
    Real peak_setup_bytes = 0;
    Real krylov_max_bytes = 0;  ///< (max_iter + 1) 2N 16 bytes (full GMRES)
};

inline Estimate estimate(const Geometry& geo, const Case& c,
                         const std::shared_ptr<excitation::Excitation>& beam) {
    const Simulation sim(geo.mesh, beam, make_config(c));
    const basis::RwgSpace& space = *sim.problem().space;
    Estimate e;
    e.triangles = geo.mesh.num_triangles();
    e.top_triangles = geo.top_triangles;
    e.unknowns = sim.num_unknowns();
    e.depth = geo.box.box_depth;
    e.fine_depth = geo.box.box_fine_depth.value_or(0.0);
    e.box_levels = geo.grading.levels;
    e.fine_rows = geo.grading.fine_rows;
    e.r_max = geo.r_max;
    e.max_support_radius = mlfmm::max_support_radius(space);
    const mlfmm::OctreeParams op = mlfmm::leaf_rule_params(space, kLambda, sim.config().mlfmm);
    const mlfmm::Octree tree(space, kLambda, op);
    e.levels = tree.levels();
    e.leaf_edge = tree.box_size(tree.leaf_level());
    e.leaves = static_cast<Index>(tree.boxes_at_level(tree.leaf_level()).size());
    e.near_bytes = mlfmm::estimate_near_bytes(tree);
    const Real omega = 2.0 * constants::pi * constants::c0 / kLambda;
    const std::array<Complex, 2> k = {sim.config().exterior.wavenumber(omega),
                                      sim.config().object.wavenumber(omega)};
    for (std::size_t i = 0; i < 2; ++i) {
        try {
            const mlfmm::LeafSampling ls = mlfmm::leaf_sampling(space, tree, k[i], c.digits);
            if (!ls.search.achievable)
                continue;
            e.leaf_order[i] = ls.sampling_order;
            const auto dirs =
                static_cast<std::size_t>(2 * (ls.sampling_order + 1) * (ls.sampling_order + 1));
            e.pattern_bytes += static_cast<std::size_t>(e.unknowns / 2) * dirs * 2 * 16;
        } catch (const std::exception&) {
            // no usable leaf expansion (Ag interior): no patterns
        }
    }
    e.peak_setup_bytes =
        2.0 * static_cast<Real>(e.near_bytes) + 1.1 * static_cast<Real>(e.pattern_bytes) + 1e9;
    e.krylov_max_bytes = static_cast<Real>(c.max_iter + 1) * static_cast<Real>(e.unknowns) *
                         static_cast<Real>(sizeof(Complex));
    return e;
}

/// Product grid on a hemisphere: n_theta Gauss-Legendre nodes in theta in (0, pi / 2) (from the
/// pole z_sign z_hat; z_sign = -1: the reflection hemisphere k_z < 0) x n_phi equispaced phi_j =
/// j 2 pi / n_phi (trapezoid), solid-angle weights w_GL sin(theta) dphi. Spectrally accurate for
/// the smooth far-field intensity (Gauss-Legendre needs no vanishing integrand at grazing
/// theta = pi / 2, where edge diffraction of the box radiates).
struct HemisphereGrid {
    Vertices dirs;
    VectorXr weights;
    VectorXr theta;  ///< polar angle from the pole per row [rad]
};

inline HemisphereGrid hemisphere_grid(Real z_sign, int n_theta, int n_phi) {
    if (n_theta < 1 || n_phi < 4)
        throw std::invalid_argument("hemisphere_grid: grid too coarse");
    const kernels::LineRule gl = kernels::gauss_legendre(n_theta);
    const Real dp = 2.0 * constants::pi / static_cast<Real>(n_phi);
    HemisphereGrid g;
    const Index rows = static_cast<Index>(n_theta) * n_phi;
    g.dirs.resize(rows, 3);
    g.weights.resize(rows);
    g.theta.resize(rows);
    for (int i = 0; i < n_theta; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        const Real t = 0.25 * constants::pi * (gl.nodes[ui] + 1.0);
        const Real wt = 0.25 * constants::pi * gl.weights[ui] * std::sin(t) * dp;
        for (int j = 0; j < n_phi; ++j) {
            const Real p = static_cast<Real>(j) * dp;
            const Index row = static_cast<Index>(i) * n_phi + j;
            g.dirs(row, 0) = std::sin(t) * std::cos(p);
            g.dirs(row, 1) = std::sin(t) * std::sin(p);
            g.dirs(row, 2) = z_sign * std::cos(t);
            g.weights(row) = wt;
            g.theta(row) = t;
        }
    }
    return g;
}

/// Grid of a case: n_theta = round(90 deg / dtheta), n_phi = round(360 deg / (2 dtheta)) (scaled
/// by 1 / coarsening).
inline HemisphereGrid hemisphere_grid(Real z_sign, Real dtheta_deg, int coarsening = 1) {
    const auto nt = static_cast<int>(std::llround(90.0 / (dtheta_deg * coarsening)));
    const auto np = static_cast<int>(std::llround(180.0 / (dtheta_deg * coarsening)));
    return hemisphere_grid(z_sign, nt, np);
}

/// Power through a hemisphere grid, (1 / 2 eta1) sum_w |F|^2 [W], with the parts in the cone of
/// half-angle cone_angle around the specular direction and at grazing angles theta > 80 deg.
struct HemispherePower {
    Real total = 0;
    Real cone = 0;
    Real grazing = 0;
};

inline HemispherePower hemisphere_power(const post::SurfaceSolution& s, const HemisphereGrid& g,
                                        int degree, Real eta1, const Vec3& specular,
                                        Real cone_angle) {
    post::FieldOptions fo;
    fo.quad_degree = degree;
    Eigen::Matrix<Complex, Eigen::Dynamic, 3> F;
    post::far_field(s, g.dirs, F, fo);
    const Real cos_cone = std::cos(cone_angle);
    HemispherePower p;
    for (Index r = 0; r < F.rows(); ++r) {
        const Real dp = g.weights(r) * F.row(r).squaredNorm() / (2.0 * eta1);
        p.total += dp;
        if (g.dirs.row(r).dot(specular.transpose()) >= cos_cone)
            p.cone += dp;
        if (g.theta(r) > 80.0 * kDeg)
            p.grazing += dp;
    }
    return p;
}

/// P_abs = 1/2 Re oint (n x M) . J* dS from the RWG currents (exact for the quadratic integrand
/// with Dunavant degree 2; see mie_dense_test_support.hpp).
inline Real absorbed_power(const post::SurfaceSolution& s) {
    const basis::RwgSpace& space = *s.problem->space;
    const geometry::TriangleMesh& mesh = space.mesh();
    const Index N = space.size();
    const kernels::TriangleRule& rule = kernels::triangle_rule(2);
    Real p_abs = 0.0;
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        const Vec3 v0 = mesh.vertices().row(mesh.triangles()(t, 0)).transpose();
        const Vec3 v1 = mesh.vertices().row(mesh.triangles()(t, 1)).transpose();
        const Vec3 v2 = mesh.vertices().row(mesh.triangles()(t, 2)).transpose();
        const Vec3 nt = mesh.normal(t);
        const basis::RwgSpace::Support sup = space.support(t);
        Real acc = 0.0;
        for (std::size_t q = 0; q < rule.weights.size(); ++q) {
            const Vec3& l = rule.barycentric[q];
            const Vec3 r = l(0) * v0 + l(1) * v1 + l(2) * v2;
            Vec3c J = Vec3c::Zero();
            Vec3c M = Vec3c::Zero();
            for (int a = 0; a < sup.count; ++a) {
                const Vec3 f = space.value(sup.n[a], t, r);
                J += s.currents(sup.n[a]) * f.cast<Complex>();
                M += s.currents(N + sup.n[a]) * f.cast<Complex>();
            }
            // n x M with real n (component form, no conjugation).
            const Vec3c nxm(nt.y() * M.z() - nt.z() * M.y(), nt.z() * M.x() - nt.x() * M.z(),
                            nt.x() * M.y() - nt.y() * M.x());
            const Complex p =
                nxm(0) * std::conj(J(0)) + nxm(1) * std::conj(J(1)) + nxm(2) * std::conj(J(2));
            acc += rule.weights[q] * p.real();
        }
        p_abs += 0.5 * mesh.area(t) * acc;
    }
    return p_abs;
}

/// Result of one solved case.
struct Result {
    Estimate est;
    Reference ref;
    std::string formulation;
    bool jacobi = false;
    int iterations = 0;
    bool converged = false;
    Real true_residual = 0;
    Real assembly_s = 0;
    Real solve_s = 0;
    Real post_s = 0;
    HemispherePower refl;  ///< reflection hemisphere, primary grid, ff_degree
    Real p_refl = 0;       ///< refl.total [W]
    Real p_refl_grid = 0;  ///< half the nodes per direction, ff_degree [W]
    Real p_refl_deg = 0;   ///< half the nodes per direction, Dunavant degree 8 [W]
    Real p_fwd = 0;        ///< forward hemisphere (half the nodes; diagnostic) [W]
    Real cone_angle = 0;   ///< 3 lambda / (pi w0) [rad]
    Real p_abs = 0;        ///< [W]
    Real near_bytes = 0;
    Real far_bytes = 0;  ///< far operator incl. exact parts, after the solve
    Real krylov_bytes = 0;
    Real peak_rss = 0;
    std::string report;

    [[nodiscard]] Real r_sim() const { return p_refl / ref.power; }
    [[nodiscard]] Real a_sim() const { return p_abs / ref.power; }
    /// Relative error against R_beam.
    [[nodiscard]] Real error_beam() const { return r_sim() / ref.beam.reflectance - 1.0; }
    /// Relative error against |r(theta_in)|^2.
    [[nodiscard]] Real error_fresnel() const { return r_sim() / ref.beam.fresnel_central - 1.0; }
};

/// Builds, assembles and solves a case and evaluates the reflected power. `geo` and `beam` from
/// make_geometry / make_beam of the same case.
inline Result run(const Case& c, const Geometry& geo,
                  const std::shared_ptr<excitation::AngularSpectrumBeam>& beam, Real beam_build_s) {
    using Clock = std::chrono::steady_clock;
    Result r;
    r.est = estimate(geo, c, beam);
    r.ref = reference(c, *beam, beam_build_s);
    Simulation sim(geo.mesh, beam, make_config(c));
    r.formulation = formulation::make_formulation(sim.formulation_kind())->name();
    r.jacobi = sim.diagonal_preconditioner();
    {
        const auto t0 = Clock::now();
        sim.assemble();
        r.assembly_s = std::chrono::duration<Real>(Clock::now() - t0).count();
    }
    const solver::GmresResult g = sim.solve();
    r.iterations = g.iterations;
    r.converged = g.converged;
    r.true_residual = g.true_relative_residual;
    r.solve_s = g.wall_seconds;
    r.krylov_bytes = static_cast<Real>(g.iterations + 1) * static_cast<Real>(sim.num_unknowns()) *
                     static_cast<Real>(sizeof(Complex));
    if (const auto op =
            std::dynamic_pointer_cast<const mlfmm::MlfmmOperator>(sim.system_operator())) {
        r.near_bytes = static_cast<Real>(op->near_operator().memory_bytes());
        r.far_bytes = static_cast<Real>(op->far_operator().memory_bytes());
    }
    const auto t0 = Clock::now();
    const Real eta1 = material::vacuum().wave_impedance(beam->omega()).real();
    const Real th = c.theta_deg * kDeg;
    const Vec3 specular(std::sin(th), 0.0, -std::cos(th));
    r.cone_angle = 3.0 * kLambda / (constants::pi * c.waist);
    r.refl = hemisphere_power(sim.solution(), hemisphere_grid(-1.0, c.ff_dtheta_deg), c.ff_degree,
                              eta1, specular, r.cone_angle);
    r.p_refl = r.refl.total;
    const HemisphereGrid coarse = hemisphere_grid(-1.0, c.ff_dtheta_deg, 2);
    r.p_refl_grid =
        hemisphere_power(sim.solution(), coarse, c.ff_degree, eta1, specular, r.cone_angle).total;
    r.p_refl_deg = hemisphere_power(sim.solution(), coarse, 8, eta1, specular, r.cone_angle).total;
    r.p_fwd = hemisphere_power(sim.solution(), hemisphere_grid(+1.0, c.ff_dtheta_deg, 2),
                               c.ff_degree, eta1, -specular, r.cone_angle)
                  .total;
    r.p_abs = absorbed_power(sim.solution());
    r.post_s = std::chrono::duration<Real>(Clock::now() - t0).count();
    r.report = sim.report();
    r.peak_rss = system_memory::peak_rss_bytes();
    return r;
}

inline Real gb(Real bytes) {
    return bytes / 1e9;
}

/// Multi-line summary of an estimate.
inline std::string summary(const Estimate& e) {
    std::ostringstream os;
    os << "triangles " << e.triangles << " (top " << e.top_triangles << ", box "
       << 100.0 * static_cast<Real>(e.triangles - e.top_triangles) /
              static_cast<Real>(e.top_triangles)
       << " %), 2N = " << e.unknowns << "; box depth " << e.depth * 1e6
       << " um, M = " << e.box_levels << ", fine band " << e.fine_depth * 1e6 << " um ("
       << e.fine_rows << " rows); r_max " << e.r_max * 1e6 << " um, support radius "
       << e.max_support_radius * 1e9 << " nm; octree " << e.levels << " levels, leaf "
       << e.leaf_edge * 1e9 << " nm, " << e.leaves << " leaves; near estimate "
       << gb(static_cast<Real>(e.near_bytes)) << " GB, leaf orders " << e.leaf_order[0] << " / "
       << e.leaf_order[1] << ", patterns " << gb(static_cast<Real>(e.pattern_bytes))
       << " GB; setup peak estimate " << gb(e.peak_setup_bytes) << " GB, Krylov at max_iter "
       << gb(e.krylov_max_bytes) << " GB";
    return os.str();
}

/// Multi-line summary of a reference.
inline std::string summary(const Reference& r) {
    std::ostringstream os;
    os.precision(8);
    os << "beam: " << r.beam.plane_waves << " plane waves, alpha_max " << r.max_polar_angle / kDeg
       << " deg, region radius " << r.region_radius * 1e6 << " um, grid change " << r.grid_change
       << ", built in " << r.build_s << " s\n"
       << "R_beam " << r.beam.reflectance << ", |r(theta_in)|^2 " << r.beam.fresnel_central
       << " (R_beam / Fresnel - 1 = " << r.beam.reflectance / r.beam.fresnel_central - 1.0
       << "), s share " << r.beam.s_share << ", k_z <= 0 share " << r.beam.away_share << "\n"
       << "power() " << r.power
       << " W, flux through z = 0 (|x|, |y| <= " << r.flux_half_extent * 1e6 << " um) "
       << r.flux.square << " W (flux / power - 1 = " << r.flux.square / r.power - 1.0
       << "), edge loss " << r.edge_loss << " (" << r.flux_s << " s)";
    return os.str();
}

/// Multi-line summary of a result.
inline std::string summary(const Result& r) {
    std::ostringstream os;
    os.precision(8);
    os << summary(r.est) << "\n"
       << summary(r.ref) << "\n"
       << "formulation " << r.formulation << (r.jacobi ? " + left Jacobi" : "") << "; GMRES "
       << r.iterations << " iterations, converged " << (r.converged ? "yes" : "no")
       << ", true residual " << r.true_residual << "; assembly " << r.assembly_s << " s, solve "
       << r.solve_s << " s, post " << r.post_s << " s\n"
       << "R_sim " << r.r_sim() << " (vs R_beam " << r.error_beam() << ", vs Fresnel "
       << r.error_fresnel() << "); hemisphere checks: half the nodes "
       << r.p_refl_grid / r.p_refl - 1.0 << ", and degree 8 " << r.p_refl_deg / r.p_refl - 1.0
       << "; specular cone (" << r.cone_angle / kDeg << " deg) " << r.refl.cone / r.p_refl
       << ", theta > 80 deg " << r.refl.grazing / r.p_refl << " of P_refl\n"
       << "A_sim " << r.a_sim() << ", 1 - R - A " << 1.0 - r.r_sim() - r.a_sim()
       << ", P_fwd / P_inc " << r.p_fwd / r.ref.power << "\n"
       << "memory: near " << gb(r.near_bytes) << " GB, far " << gb(r.far_bytes) << " GB, Krylov "
       << gb(r.krylov_bytes) << " GB, peak RSS " << gb(r.peak_rss) << " GB";
    return os.str();
}

}  // namespace fresnel_flat
