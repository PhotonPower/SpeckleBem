#pragma once
/// @file mie_dense_test_support.hpp
/// Helpers of the Mie validation of the dense solver (WP11; validation:
/// tests/validation/test_mie_sphere_dense.cpp, validation-large:
/// tests/validation_large/test_mie_sphere_dense_large.cpp): a timed dense PMCHWT + LU solve of
/// the sphere of assembler_test_support.hpp (radius 0.5 um, lambda = 500 nm, k = +z, E along x)
/// with eps_rr in the xz- and yz-planes, the power balance of docs/05 from the solved currents,
/// and a memory guard / peak-memory probe for the large cases.
#include "specklebem/kernels/quadrature.hpp"

#include <catch2/catch_message.hpp>

#include <chrono>
#include <cmath>

#include "assembler_test_support.hpp"

#if __has_include(<sys/resource.h>) && __has_include(<unistd.h>)
#include <sys/resource.h>
#include <unistd.h>
#define SPECKLEBEM_TEST_HAVE_POSIX_MEMORY 1
#endif

namespace mie_dense_test {

using namespace assembler_test;

/// RCS angles per scattering plane (theta = 0, 1, ..., 180 degrees).
inline constexpr Index kRcsAngles = 181;

/// Power balance of docs/05 from the solved currents (exp(+jwt), R1 = vacuum, |E0| = 1 V/m):
///   P_sca = (1 / 2 eta1) int |F(k_hat)|^2 dOmega over the unit sphere (Gauss-Legendre in
///           cos theta x uniform phi, post::far_field);
///   P_ext = C_ext |E0|^2 / (2 eta1) with the optical theorem
///           C_ext = -(4 pi / (k1 |E0|^2)) Im[F(z) . e0*]
///           (Bohren & Huffman: C_ext = (4 pi / k^2) Re[X . e0*] with E_s = e^{ikr} X / (-ikr),
///           i.e. F_BH = X / (-ik) and C_ext = (4 pi / k) Im[F_BH . e0*]; the exp(+jwt) phasor is
///           F = conj(F_BH), which flips the sign of the imaginary part);
///   P_abs = -1/2 Re oint (E x H*) . n dS on the R1 side, with E_tan = n x M and
///           H_tan = -n x J (M = -n x E, J = n x H): (E x H*) . n = -(n x M) . J*, so
///           P_abs = 1/2 Re oint (n x M) . J* dS, integrated exactly over the RWG expansion
///           (Dunavant degree 2 per flat triangle, the integrand is quadratic).
/// Cross sections C = P / I_inc with I_inc = |E0|^2 / (2 eta1).
struct PowerBalance {
    Real p_ext = 0, p_sca = 0, p_abs = 0;  ///< W
    Real c_ext = 0, c_sca = 0, c_abs = 0;  ///< m^2
    /// |P_ext - P_sca - P_abs| / P_ext
    [[nodiscard]] Real defect() const { return std::abs(p_ext - p_sca - p_abs) / p_ext; }
};

inline PowerBalance power_balance(const post::SurfaceSolution& s, const Vec3c& e0, Real k1,
                                  Real eta1, int n_theta = 64, int n_phi = 128) {
    const basis::RwgSpace& space = *s.problem->space;
    const geometry::TriangleMesh& mesh = space.mesh();
    const Index N = space.size();
    const Real e0_sq = e0.squaredNorm();
    const Real i_inc = e0_sq / (2.0 * eta1);

    // Scattered power: int |F|^2 dOmega, u = cos theta on Gauss-Legendre nodes.
    const kernels::LineRule gl = kernels::gauss_legendre(n_theta);
    Vertices dirs(static_cast<Index>(n_theta) * n_phi, 3);
    VectorXr w(dirs.rows());
    const Real dphi = 2.0 * kPi / static_cast<Real>(n_phi);
    Index row = 0;
    for (std::size_t i = 0; i < gl.nodes.size(); ++i) {
        const Real u = gl.nodes[i];
        const Real st = std::sqrt(1.0 - u * u);
        for (int j = 0; j < n_phi; ++j) {
            const Real phi = dphi * static_cast<Real>(j);
            dirs.row(row) = Vec3(st * std::cos(phi), st * std::sin(phi), u).transpose();
            w(row) = gl.weights[i] * dphi;
            ++row;
        }
    }
    Eigen::Matrix<Complex, Eigen::Dynamic, 3> F;
    post::far_field(s, dirs, F);
    Real int_f2 = 0.0;
    for (Index r = 0; r < dirs.rows(); ++r) int_f2 += w(r) * F.row(r).squaredNorm();

    // Extinction: optical theorem with the forward amplitude.
    Vertices fwd(1, 3);
    fwd.row(0) = Vec3::UnitZ().transpose();
    Eigen::Matrix<Complex, Eigen::Dynamic, 3> F0;
    post::far_field(s, fwd, F0);
    const Complex f_dot_e0 =
        F0(0, 0) * std::conj(e0(0)) + F0(0, 1) * std::conj(e0(1)) + F0(0, 2) * std::conj(e0(2));

    // Absorption: 1/2 Re oint (n x M) . J* dS over the RWG expansion.
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
            const Vec3c nxm = cross_rc(nt, M);
            const Complex p =
                nxm(0) * std::conj(J(0)) + nxm(1) * std::conj(J(1)) + nxm(2) * std::conj(J(2));
            acc += rule.weights[q] * p.real();
        }
        p_abs += 0.5 * mesh.area(t) * acc;
    }

    PowerBalance pb;
    pb.p_sca = int_f2 / (2.0 * eta1);
    pb.c_ext = -4.0 * kPi / (k1 * e0_sq) * f_dot_e0.imag();
    pb.p_ext = pb.c_ext * i_inc;
    pb.p_abs = p_abs;
    pb.c_sca = pb.p_sca / i_inc;
    pb.c_abs = pb.p_abs / i_inc;
    return pb;
}

/// Number of unknowns 2N of icosphere level n (20 * 4^n triangles, 30 * 4^n edges).
inline Index icosphere_unknowns(int subdivisions) {
    Index edges = 30;
    for (int i = 0; i < subdivisions; ++i) edges *= 4;
    return 2 * edges;
}

/// Estimated peak memory of a dense solve: Z plus the LU copy of solve_direct, 2 * 16 (2N)^2.
inline Real dense_solve_bytes(Index unknowns) {
    const auto n = static_cast<Real>(unknowns);
    return 2.0 * 16.0 * n * n;
}

/// Physical memory of the machine in bytes (0 if unknown).
inline Real physical_memory_bytes() {
#ifdef SPECKLEBEM_TEST_HAVE_POSIX_MEMORY
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0)
        return static_cast<Real>(pages) * static_cast<Real>(page_size);
#endif
    return 0.0;
}

/// Peak resident set size of this process in bytes (0 if unknown; ru_maxrss is in KiB on
/// Linux). catch_discover_tests runs every test case in its own process.
inline Real peak_rss_bytes() {
#ifdef SPECKLEBEM_TEST_HAVE_POSIX_MEMORY
    rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) == 0)
        return 1024.0 * static_cast<Real>(ru.ru_maxrss);
#endif
    return 0.0;
}

/// Memory guard of the validation-large cases: true if the estimated dense-solve memory of
/// `unknowns` exceeds `fraction` of the physical memory (or the latter is unknown).
inline bool exceeds_memory(Index unknowns, Real fraction = 0.6) {
    const Real phys = physical_memory_bytes();
    return !(phys > 0.0) || dense_solve_bytes(unknowns) > fraction * phys;
}

/// Result of one dense PMCHWT + LU solve of the Mie sphere.
struct DenseMieResult {
    Index triangles = 0;
    Index unknowns = 0;      ///< 2N
    Real lambda_over_h = 0;  ///< vacuum wavelength / mean edge length
    Real eps_xz = 0;         ///< eps_rr, xz-plane (phi = 0), kRcsAngles angles
    Real eps_yz = 0;         ///< eps_rr, yz-plane (phi = pi / 2)
    solver::DirectSolveInfo info;
    Real assembly_s = 0;  ///< DenseStrategy::build
    Real solve_s = 0;     ///< solve_direct (LU + refinement)
    bool has_power = false;
    PowerBalance power;  ///< filled when requested
};

/// Assembles and solves the sphere of `mat` at icosphere level `subdivisions` (PMCHWT), and
/// computes eps_rr in both planes (and, if `with_power`, the power balance). Z is released
/// before the post-processing.
inline DenseMieResult solve_mie_dense(const material::Material& mat, int subdivisions,
                                      bool with_power = false) {
    using Clock = std::chrono::steady_clock;
    SphereCase c(mat, subdivisions, formulation::Kind::PMCHWT);
    DenseMieResult res;
    res.triangles = c.setup.mesh.num_triangles();
    res.unknowns = 2 * c.size();
    {
        const geometry::TriangleMesh& mesh = c.setup.mesh;
        Real len = 0.0;
        for (Index e = 0; e < mesh.num_edges(); ++e) {
            const Vec3 a = mesh.vertices().row(mesh.edges()(e, 0)).transpose();
            const Vec3 b = mesh.vertices().row(mesh.edges()(e, 1)).transpose();
            len += (b - a).norm();
        }
        res.lambda_over_h = kLambda / (len / static_cast<Real>(mesh.num_edges()));
    }
    VectorXc x;
    {
        const auto t0 = Clock::now();
        const op::DenseOperator Z(assemble(c.problem()));
        const auto t1 = Clock::now();
        const VectorXc b = op::assemble_rhs(c.problem());
        const auto t2 = Clock::now();
        x = solver::solve_direct(Z, b, &res.info);
        const auto t3 = Clock::now();
        res.assembly_s = std::chrono::duration<Real>(t1 - t0).count();
        res.solve_s = std::chrono::duration<Real>(t3 - t2).count();
    }
    const post::SurfaceSolution sol{&c.problem(), x};
    res.eps_xz = rcs_eps_rr(sol, c.setup.mie, kRcsAngles, Vec3::UnitY(), 0.0);
    res.eps_yz = rcs_eps_rr(sol, c.setup.mie, kRcsAngles, -Vec3::UnitX(), 0.5 * kPi);
    if (with_power) {
        res.power = power_balance(sol, Vec3c(1.0, 0.0, 0.0), k_vacuum(), constants::eta0);
        res.has_power = true;
    }
    return res;
}

/// Mie cross sections of the sphere (C_abs = C_ext - C_sca).
struct MieCrossSections {
    Real c_ext = 0, c_sca = 0, c_abs = 0;
};

inline MieCrossSections mie_cross_sections(const material::Material& mat) {
    const MieSolution mie(MieParams{kRadius, kLambda, mat, material::vacuum(), 0});
    MieCrossSections c;
    c.c_ext = mie.extinction_cross_section();
    c.c_sca = mie.scattering_cross_section();
    c.c_abs = c.c_ext - c.c_sca;
    return c;
}

/// |value - ref| / |ref|
inline Real rel_error(Real value, Real ref) {
    return std::abs(value - ref) / std::abs(ref);
}

/// One-line summary of a result for WARN / the results record.
inline void report(const char* label, int subdivisions, const DenseMieResult& r) {
    WARN(label << ", icosphere n = " << subdivisions << " (" << r.triangles
               << " triangles, 2N = " << r.unknowns << ", h = lambda / " << r.lambda_over_h
               << "): eps_rr xz = " << r.eps_xz << ", yz = " << r.eps_yz << "; assembly "
               << r.assembly_s << " s, LU " << r.solve_s << " s, rcond " << r.info.rcond
               << ", LU residual " << r.info.residual << ", peak RSS " << peak_rss_bytes() / 1e9
               << " GB");
}

/// Report of the power balance: cross sections and their relative deviations from Mie (the C_abs
/// error relative to the Mie C_ext, since C_abs may vanish).
inline void report_power(const char* label, int subdivisions, const PowerBalance& p,
                         const MieCrossSections& m) {
    WARN(label << ", icosphere n = " << subdivisions << ": power balance |P_ext - P_sca - "
               << "P_abs| / P_ext = " << p.defect() << "; C_ext = " << p.c_ext << " m^2 (Mie "
               << m.c_ext << ", rel " << (p.c_ext - m.c_ext) / m.c_ext << "), C_sca = " << p.c_sca
               << " (Mie " << m.c_sca << ", rel " << (p.c_sca - m.c_sca) / m.c_sca
               << "), C_abs = " << p.c_abs << " (Mie " << m.c_abs << ", error / C_ext_Mie "
               << (p.c_abs - m.c_abs) / m.c_ext << "), C_ext - C_sca = " << p.c_ext - p.c_sca);
}

}  // namespace mie_dense_test
