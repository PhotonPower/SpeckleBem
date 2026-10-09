#pragma once
/// @file fields_test_support.hpp
/// Shared helpers of the post-processing tests (unit: tests/unit/test_fields.cpp, validation:
/// tests/validation/test_post_mie_projection.cpp): exact Mie surface currents projected onto RWG
/// functions and the error measures of the projection test.
#include "specklebem/basis/rwg.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/geometry/mesh.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/postprocessing/fields.hpp"
#include "specklebem/postprocessing/scattering.hpp"
#include "specklebem/reference/mie.hpp"

#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace fields_test {

using namespace specklebem;
using reference::MieParams;
using reference::MieSolution;
using FieldMatrix = Eigen::Matrix<Complex, Eigen::Dynamic, 3>;

inline constexpr Real kPi = constants::pi;
inline constexpr Real kLambda = 500e-9;
inline constexpr Real kRadius = 0.5e-6;  // sphere of diameter 1 um (docs/05)
inline constexpr Complex kJ{0.0, 1.0};

inline Real omega_of(Real lambda) {
    return 2 * kPi * constants::c0 / lambda;
}

inline Real k_vacuum(Real lambda = kLambda) {
    return omega_of(lambda) / constants::c0;
}

inline material::Material lossless_n15() {
    return {Complex(1.5 * 1.5, 0), Complex(1, 0)};
}

inline Vec3 direction(Real theta, Real phi) {
    return {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};
}

/// Incident field of docs/06 in vacuum: E = x exp(-j k z), H = y exp(-j k z) / eta0, with
/// k = 2 pi / lambda (vacuum wavelength lambda, default kLambda). Analytic form of
/// MieSetup::wave (excitation::PlaneWave), used to build the projected Mie currents.
inline Vec3c incident_E(const Vec3& r, Real lambda = kLambda) {
    return {std::exp(-kJ * k_vacuum(lambda) * r.z()), 0, 0};
}
inline Vec3c incident_H(const Vec3& r, Real lambda = kLambda) {
    return {0, std::exp(-kJ * k_vacuum(lambda) * r.z()) / constants::eta0, 0};
}

/// Complex vector times real vector without conjugation.
inline Complex dot_cr(const Vec3c& a, const Vec3& b) {
    return a(0) * b(0) + a(1) * b(1) + a(2) * b(2);
}

/// a x b for real a and complex b. (Eigen's cross() conjugates the result for complex
/// operands, so it must not be used for phasors.)
inline Vec3c cross_rc(const Vec3& a, const Vec3c& b) {
    return {a.y() * b.z() - a.z() * b.y(), a.z() * b.x() - a.x() * b.z(),
            a.x() * b.y() - a.y() * b.x()};
}

/// Central-difference curl of a vector field.
inline Vec3c curl_fd(const std::function<Vec3c(const Vec3&)>& f, const Vec3& r, Real h) {
    Vec3c d[3];  // d[k] = dF/dx_k
    for (Index k = 0; k < 3; ++k) {
        Vec3 e = Vec3::Zero();
        e(k) = h;
        d[k] = (f(r + e) - f(r - e)) / (2 * h);
    }
    return {d[1](2) - d[2](1), d[2](0) - d[0](2), d[0](1) - d[1](0)};
}

/// RWG coefficients of the exact Mie surface currents on the R1 side: J = n x H_total,
/// M = -n x E_total at the edge midpoint projected radially onto the sphere (n the exact sphere
/// normal there), x_n = X . nu_n with nu_n the in-plane unit normal of edge n in the plane of
/// T_n^+, pointing from T_n^+ into T_n^- (the RWG normal component is 1 there). `lambda` is the
/// vacuum wavelength of the Mie solution (for the incident field).
inline VectorXc project_mie_currents(const basis::RwgSpace& space, const MieSolution& mie,
                                     Real lambda = kLambda) {
    const geometry::TriangleMesh& mesh = space.mesh();
    const Index N = space.size();
    VectorXc x(2 * N);
    for (Index n = 0; n < N; ++n) {
        const Vec3 va = mesh.vertices().row(mesh.edges()(n, 0)).transpose();
        const Vec3 vb = mesh.vertices().row(mesh.edges()(n, 1)).transpose();
        // T^+ traverses a -> b counter-clockwise, so (b - a) x n_plus points out of T^+.
        const Vec3 nu = (vb - va).cross(mesh.normal(space.plus_triangle(n))).normalized();
        const Vec3 rs = kRadius * (0.5 * (va + vb)).normalized();
        const Vec3 nh = rs.normalized();
        const Vec3c E = incident_E(rs, lambda) + mie.scattered_E(rs);
        const Vec3c H = incident_H(rs, lambda) + mie.scattered_H(rs);
        x(n) = dot_cr(cross_rc(nh, H), nu);
        x(N + n) = dot_cr(-cross_rc(nh, E), nu);
    }
    return x;
}

/// Sphere in vacuum, icosphere mesh, RWG space and projected Mie currents, with the plane wave
/// of docs/06 (excitation::PlaneWave, k_hat = z, e0 = x, vacuum background) attached as
/// problem.excitation and problem.omega = wave.omega(). Vacuum wavelength `lambda` (default
/// kLambda; the near-field / RCS error helpers below assume kLambda). problem.formulation is
/// not set (post-processing does not need it).
struct MieSetup {
    MieSetup(const material::Material& sphere, int subdivisions, Real lambda = kLambda)
        : mesh(geometry::make_icosphere(kRadius, subdivisions)),
          space(mesh),
          mie(MieParams{kRadius, lambda, sphere, material::vacuum(), 0}),
          wave(lambda, Vec3::UnitZ(), Vec3c(1.0, 0.0, 0.0)) {
        problem.space = &space;
        problem.exterior = material::vacuum();
        problem.object = sphere;
        problem.excitation = &wave;
        problem.omega = wave.omega();
        solution.problem = &problem;
        solution.currents = project_mie_currents(space, mie, lambda);
    }
    MieSetup(const MieSetup&) = delete;
    MieSetup& operator=(const MieSetup&) = delete;

    geometry::TriangleMesh mesh;
    basis::RwgSpace space;
    MieSolution mie;
    excitation::PlaneWave wave;
    op::Problem problem;
    post::SurfaceSolution solution;
};

/// Max over points and components of |field - reference|, normalised by |E_inc| = 1 V/m (E)
/// and |H_inc| = 1 / eta0 (H); exterior points against the Mie scattered field, interior points
/// against the Mie internal field.
struct ProjectionErrors {
    Real e_ext = 0, h_ext = 0, e_int = 0, h_int = 0;
};

inline ProjectionErrors near_field_errors(const MieSetup& s) {
    const std::vector<Vec3> dirs = {direction(0.3, 0.2), direction(1.2, 2.0), direction(2.0, 4.0),
                                    direction(2.9, 5.5)};
    const std::vector<Real> ext_radii = {1.5 * kRadius, 2.0 * kRadius, 3.0 * kRadius};
    const Real int_radius = 0.5 * kRadius;
    const auto n_dirs = static_cast<Index>(dirs.size());
    const auto n_ext = static_cast<Index>(ext_radii.size()) * n_dirs;
    Vertices pts(n_ext + n_dirs, 3);
    Index row = 0;
    for (const Real r : ext_radii) {
        for (const Vec3& d : dirs) pts.row(row++) = (r * d).transpose();
    }
    for (const Vec3& d : dirs) pts.row(row++) = (int_radius * d).transpose();

    FieldMatrix E;
    FieldMatrix H;
    post::scattered_field(s.solution, pts, E, H);

    const Real omega = omega_of(kLambda);
    const auto internal_E = [&](const Vec3& r) { return s.mie.internal_E(r); };
    ProjectionErrors err;
    for (Index p = 0; p < pts.rows(); ++p) {
        const Vec3 r = pts.row(p).transpose();
        const bool exterior = p < n_ext;
        Vec3c e_ref;
        Vec3c h_ref;
        if (exterior) {
            e_ref = s.mie.scattered_E(r);
            h_ref = s.mie.scattered_H(r);
        } else {
            e_ref = s.mie.internal_E(r);
            // curl E = -j w mu0 H (mu_r = 1)
            h_ref = curl_fd(internal_E, r, 1e-3 * kRadius) / (-kJ * omega * constants::mu0);
        }
        const Vec3c de = E.row(p).transpose() - e_ref;
        const Vec3c dh = H.row(p).transpose() - h_ref;
        const Real ee = de.cwiseAbs().maxCoeff();
        const Real eh = dh.cwiseAbs().maxCoeff() * constants::eta0;
        if (exterior) {
            err.e_ext = std::max(err.e_ext, ee);
            err.h_ext = std::max(err.h_ext, eh);
        } else {
            err.e_int = std::max(err.e_int, ee);
            err.h_int = std::max(err.h_int, eh);
        }
    }
    return err;
}

/// RCS errors against Mie in the xz-plane (plane normal +y): eps_rr of docs/05 (RMS over the
/// angles divided by max sigma_Mie) and the forward / backward errors relative to max sigma.
struct RcsErrors {
    Real rms = 0, forward = 0, backward = 0;
};

inline RcsErrors rcs_errors(const MieSetup& s) {
    const Index na = 19;  // every 10 degrees
    VectorXr angles(na);
    for (Index i = 0; i < na; ++i) angles(i) = kPi * static_cast<Real>(i) / (na - 1);
    const VectorXr sigma = post::bistatic_rcs(s.solution, Vec3::UnitY(), angles);
    VectorXr ref(na);
    for (Index i = 0; i < na; ++i) ref(i) = s.mie.bistatic_rcs(angles(i), 0.0);
    const Real max_ref = ref.maxCoeff();
    RcsErrors e;
    e.rms = std::sqrt((sigma - ref).squaredNorm() / static_cast<Real>(na)) / max_ref;
    e.forward = std::abs(sigma(0) - ref(0)) / max_ref;
    e.backward = std::abs(sigma(na - 1) - ref(na - 1)) / max_ref;
    return e;
}

inline void print_errors(const char* label, int n, const ProjectionErrors& p, const RcsErrors& r) {
    WARN(label << " n=" << n << ": E_ext " << p.e_ext << ", H_ext " << p.h_ext << ", E_int "
               << p.e_int << ", H_int " << p.h_int << ", RCS rms " << r.rms << ", fwd " << r.forward
               << ", back " << r.backward);
}

}  // namespace fields_test
