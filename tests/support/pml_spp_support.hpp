#pragma once
/// @file pml_spp_support.hpp
/// Validation 7b of the open-surface PML route (ADR 0009 decision 7b, WP-PML4a; docs/05 "PML
/// route", row 7b): a surface plasmon polariton (SPP) launched on a flat Ag open surface must be
/// reflected by the PML start with |r_spp| < 1e-2. Shared by the study executable
/// benchmarks/pml_spp.cpp, the validation-large test tests/validation_large/test_pml_spp_large.cpp
/// and the unit tests of the fit (tests/unit/test_pml_spp.cpp); no Catch2.
///
/// Launch. The open surface (patch, taper, gap, PML; vacuum R1 over Ag R2, lambda = 500 nm) is
/// flat except for a small Gaussian groove xi = d exp(-rho^2 / r_g^2) at the origin (d > 0 is a
/// dimple into the metal, the object occupies z > xi), illuminated at normal incidence by the
/// rigorous angular-spectrum beam with E along x. The difference of the solutions of the grooved
/// and of the flat surface (same grid, only z differs, so the RWG numbering is the same) is the
/// field scattered by the groove: an SPP running along +-x (dipole pattern cos(phi)), radiation and
/// the quasi-cylindrical surface wave. All errors of the masked incidence at the PML start are
/// common to both solutions and cancel. A second, cheaper source uses the flat operator only:
/// the beam's right-hand side restricted to the test functions whose edge midpoint lies within
/// source_radius of the origin (an impressed source on the interface; the same LU as the flat
/// solve). Both launch the same guided mode; the PML reflection is a property of the PML.
///
/// Measurement. Along the x-axis (both half-axes, rho = |x|) between rho_min and the PML start a,
/// three quantities are sampled, each of which is, for a TM SPP with E_z ~ cos(phi) H1(k rho),
/// one exact cylindrical form on the axis (J = n x H = C grad_t E_z, M = -n x E):
///  * J_x at the midpoints of the edges perpendicular to x (the normal component of an RWG
///    expansion across an edge is continuous and equals the edge coefficient), ~ H1'(k rho);
///  * M_y at the midpoints of the edges perpendicular to y, ~ H1'(k rho);
///  * div_s J per triangle (centroid, = -j omega eps_1 E_n on the R1 side), ~ H1(k rho).
/// fit_spp() fits, by least squares, an outgoing wave w(k rho), the wave reflected at the plane
/// rho = a as a cylindrical wave from the image source, w(k (2 a - rho)), and outgoing background
/// terms exp(-j k0 rho) (k0 rho)^(-p) (radiation, quasi-cylindrical wave). The SPP wavenumber
/// k is fitted (Re k by a scan and a golden-section refinement, Im k optionally), so that the
/// numerical dispersion of the mesh does not leak into the reflected amplitude. Referred to the
/// PML start plane, r_spp = in / out (both waves have the value w(k a) there).
///
/// Consistency of the two current observables. For a TM SPP the tangential fields on the
/// interface satisfy |E_x / H_y| = |kappa| / (omega eps0) with kappa = sqrt(k_spp^2 - k0^2)
/// (vacuum R1; spp_impedance(), 127 Ohm for Ag at 500 nm), so the fitted outgoing amplitudes of a
/// clean SPP satisfy |out(M_y) / out(J_x)| = spp_impedance(); out_impedance() reports the measured
/// ratio (row_line(), the study's ZOUT output) as a diagnostic of how much non-SPP field the
/// outgoing J_x column carries.
///
/// Solve. Dense operator (op::DenseStrategy on the Simulation's Problem, owned here), LU in
/// place (LAPACK zgetrf when available, else Eigen's PartialPivLU on an Eigen::Ref of the same
/// storage; the peak is one matrix, 16 (2N)^2 bytes, not two), several right-hand sides per
/// factorisation; a few rows of Z are kept before the factorisation to report the residual
/// |b - Z x| / |b| on those rows.
#include "specklebem/core/path.hpp"

#include <Eigen/LU>
#include <Eigen/QR>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pml_fresnel_support.hpp"

#ifdef SPECKLEBEM_HAVE_BLAS_LAPACK
extern "C" {
// LAPACK (Fortran interface, LP64 integers as in OpenBLAS / reference LAPACK / MKL LP64) for the
// in-place LU below. The library's solver::DenseLu keeps its own copy of these declarations;
// pml_fresnel_support.hpp no longer provides them since it uses DenseLu (WP-LU1).
void zgetrf_(const int* m, const int* n, std::complex<double>* a, const int* lda, int* ipiv,
             int* info);
void zgetrs_(const char* trans, const int* n, const int* nrhs, const std::complex<double>* a,
             const int* lda, const int* ipiv, std::complex<double>* b, const int* ldb, int* info,
             std::size_t trans_len);
}
#endif

namespace pml_spp {

using namespace specklebem;
using fresnel_flat::kLambda;

// ---------------------------------------------------------------------------------------------
// Cylindrical Hankel functions and the SPP wavenumber
// ---------------------------------------------------------------------------------------------

/// Smallest |z| accepted by hankel(): the optimally truncated large-argument expansion is
/// accurate to about exp(-2 |z|) relative there (|z| = 6: ~1e-5, 8: ~1e-7, 10: ~2e-9).
inline constexpr Real kMinHankelArgument = 6.0;

/// Cylindrical Hankel function H_nu^(kind)(z), kind 1 or 2, integer order 0 <= nu <= 4, from the
/// large-argument expansion (DLMF 10.17.3 / 10.17.4)
///   H_nu^(1,2)(z) ~ sqrt(2 / (pi z)) exp(+-j w) sum_k (+-j)^k a_k(nu) / z^k,
///   w = z - nu pi / 2 - pi / 4, a_k = a_(k-1) (4 nu^2 - (2k - 1)^2) / (8 k),
/// summed while the terms decrease: the smallest term is included, the first growing one is not
/// (or until a term falls below 1e-17 of the sum). H^(2) ~ exp(-j z) is the outgoing wave of the
/// exp(+j w t) convention.
/// @throws std::invalid_argument for kind not 1 or 2, nu outside [0, 4], a non-finite z,
///         Re z <= 0 or |z| < kMinHankelArgument.
[[nodiscard]] inline Complex hankel(int kind, int nu, Complex z) {
    if (kind != 1 && kind != 2)
        throw std::invalid_argument("pml_spp::hankel: kind must be 1 or 2");
    if (nu < 0 || nu > 4)
        throw std::invalid_argument("pml_spp::hankel: order must be in [0, 4]");
    if (!(std::isfinite(z.real()) && std::isfinite(z.imag())) || !(z.real() > 0.0) ||
        !(std::abs(z) >= kMinHankelArgument))
        throw std::invalid_argument("pml_spp::hankel: need finite z with Re z > 0 and |z| >= " +
                                    std::to_string(kMinHankelArgument));
    const Complex j(0.0, 1.0);
    const Complex s = kind == 2 ? -j : j;
    const Real mu = 4.0 * static_cast<Real>(nu) * static_cast<Real>(nu);
    Complex sum = 1.0;
    Complex term = 1.0;
    Real previous = 1.0;
    for (int k = 1; k < 200; ++k) {
        const Real odd = 2.0 * static_cast<Real>(k) - 1.0;
        term *= s * ((mu - odd * odd) / (8.0 * static_cast<Real>(k))) / z;
        const Real mag = std::abs(term);
        if (!(mag < previous))
            break;  // asymptotic series: stop before the terms grow again
        sum += term;
        previous = mag;
        if (mag < 1e-17 * std::abs(sum))
            break;
    }
    const Complex omega = z - (static_cast<Real>(nu) * constants::pi / 2.0 + constants::pi / 4.0);
    return std::sqrt(2.0 / (constants::pi * z)) * std::exp(s * omega) * sum;
}

/// Shape of the SPP along the measurement line (see the file comment).
enum class WaveForm {
    H0,       ///< H_0^(2)(z)
    H1,       ///< H_1^(2)(z): E_z and div_s J on the axis of a cos(phi) source
    H1Prime,  ///< dH_1^(2)/dz = H_0^(2)(z) - H_1^(2)(z) / z: J_x and M_y on the axis
};

[[nodiscard]] inline const char* wave_form_name(WaveForm f) {
    switch (f) {
        case WaveForm::H0:
            return "H0";
        case WaveForm::H1:
            return "H1";
        case WaveForm::H1Prime:
            return "H1'";
    }
    return "?";
}

/// Outgoing wave of the given form at z = k rho (Hankel functions of the second kind).
[[nodiscard]] inline Complex wave(WaveForm f, Complex z) {
    switch (f) {
        case WaveForm::H0:
            return hankel(2, 0, z);
        case WaveForm::H1:
            return hankel(2, 1, z);
        case WaveForm::H1Prime:
            return hankel(2, 0, z) - hankel(2, 1, z) / z;
    }
    throw std::invalid_argument("pml_spp::wave: unknown form");
}

/// SPP wavenumber of a flat interface between eps1 (R1) and eps2 (R2) for exp(+j w t):
/// k_spp = k0 sqrt(eps1 eps2 / (eps1 + eps2)), the root with Re > 0 (then Im <= 0 for passive
/// media). Ag at 500 nm in vacuum: k_spp = (1.05526 - 0.00192 j) k0, lambda_spp = 473.8 nm,
/// amplitude decay length 1 / |Im k_spp| = 41.5 um (intensity 20.8 um).
/// @throws std::invalid_argument for k0 not finite and > 0 or eps1 + eps2 = 0.
[[nodiscard]] inline Complex spp_wavenumber(Real k0, Complex eps1, Complex eps2) {
    if (!(std::isfinite(k0) && k0 > 0.0))
        throw std::invalid_argument("pml_spp::spp_wavenumber: k0 must be finite and > 0");
    if (eps1 + eps2 == Complex(0.0, 0.0))
        throw std::invalid_argument("pml_spp::spp_wavenumber: eps1 + eps2 = 0");
    Complex q = std::sqrt(eps1 * eps2 / (eps1 + eps2));
    if (q.real() < 0.0)
        q = -q;
    return k0 * q;
}

/// |E_x / H_y| of a TM SPP with wavenumber k_spp on the interface, seen from R1 (relative
/// permittivity eps1): the field in R1 decays away from the interface with
/// kappa = sqrt(k_spp^2 - eps1 k0^2), and curl H = j omega eps0 eps1 E gives
/// |E_x / H_y| = |kappa| / (omega eps0 |eps1|) = eta0 |kappa| / (k0 |eps1|) [Ohm] (tangential E
/// and H are continuous, so the R2 side gives the same value). Ag at 500 nm in vacuum:
/// kappa = 0.337 k0, 127.0 Ohm. For the fits, |out(M_y) / out(J_x)| of a clean SPP.
/// @throws std::invalid_argument for k0 not finite and > 0 or eps1 = 0.
[[nodiscard]] inline Real spp_impedance(Real k0, Complex k_spp, Complex eps1 = Complex(1.0, 0.0)) {
    if (!(std::isfinite(k0) && k0 > 0.0))
        throw std::invalid_argument("pml_spp::spp_impedance: k0 must be finite and > 0");
    if (eps1 == Complex(0.0, 0.0))
        throw std::invalid_argument("pml_spp::spp_impedance: eps1 = 0");
    const Complex kappa = std::sqrt(k_spp * k_spp - eps1 * k0 * k0);
    return constants::eta0 * std::abs(kappa) / (k0 * std::abs(eps1));
}

// ---------------------------------------------------------------------------------------------
// Fit of outgoing + reflected cylindrical SPP waves
// ---------------------------------------------------------------------------------------------

struct FitOptions {
    WaveForm form = WaveForm::H1Prime;
    Real plane = 0.0;   ///< reflection plane a [m] (PML start); every rho must lie in (0, a]
    Real k0 = 0.0;      ///< background wavenumber [1/m] (vacuum)
    Complex k_guess{};  ///< SPP wavenumber guess [1/m] (Re > 0)
    /// Exponents p of the outgoing background terms exp(-j k0 rho) (k0 rho)^(-p); may be empty.
    std::vector<Real> background_powers = {1.0, 1.5, 2.0};
    bool fit_k = true;      ///< fit Re k (else k = k_guess)
    bool fit_imag = false;  ///< also fit Im k (after Re k); else Im k = Im k_guess
    Real k_search = 0.04;   ///< relative half range of the Re k scan about Re k_guess
};

struct FitResult {
    Complex k{};    ///< fitted SPP wavenumber [1/m]
    Complex out{};  ///< amplitude of the outgoing wave w(k rho)
    Complex in{};   ///< amplitude of the reflected wave w(k (2 a - rho))
    std::vector<Complex> background;
    Real residual = 0;  ///< |y - model| / |y|
    Index points = 0;
    /// Reflection coefficient referred to the plane rho = a.
    [[nodiscard]] Complex r() const { return in / out; }
};

namespace detail {

/// Linear least squares for a fixed k; returns the relative residual, fills *res if non-null.
inline Real fit_linear(const std::vector<Real>& rho, const std::vector<Complex>& y,
                       const FitOptions& o, Complex k, FitResult* res) {
    const auto m = static_cast<Index>(rho.size());
    const auto nb = static_cast<Index>(o.background_powers.size());
    const Index p = 2 + nb;
    const Complex j(0.0, 1.0);
    MatrixXc A(m, p);
    VectorXc b(m);
    for (Index i = 0; i < m; ++i) {
        const Real r = rho[static_cast<std::size_t>(i)];
        b(i) = y[static_cast<std::size_t>(i)];
        A(i, 0) = wave(o.form, k * r);
        A(i, 1) = wave(o.form, k * (2.0 * o.plane - r));
        for (Index q = 0; q < nb; ++q)
            A(i, 2 + q) = std::exp(-j * (o.k0 * r)) *
                          std::pow(o.k0 * r, -o.background_powers[static_cast<std::size_t>(q)]);
    }
    VectorXr scale(p);
    MatrixXc As = A;
    for (Index c = 0; c < p; ++c) {
        scale(c) = A.col(c).norm();
        As.col(c) /= scale(c);
    }
    const Eigen::ColPivHouseholderQR<MatrixXc> qr(As);
    VectorXc coef = qr.solve(b);
    for (Index c = 0; c < p; ++c) coef(c) /= scale(c);
    const Real residual = (b - A * coef).norm() / b.norm();
    if (res) {
        res->k = k;
        res->out = coef(0);
        res->in = coef(1);
        res->background.assign(coef.data() + 2, coef.data() + p);
        res->residual = residual;
        res->points = m;
    }
    return residual;
}

/// Golden-section minimisation of f on [a, b] to an absolute tolerance tol.
inline Real golden_min(const std::function<Real(Real)>& f, Real a, Real b, Real tol) {
    const Real g = (std::sqrt(5.0) - 1.0) / 2.0;
    Real c = b - g * (b - a);
    Real d = a + g * (b - a);
    Real fc = f(c);
    Real fd = f(d);
    while (b - a > tol) {
        if (fc < fd) {
            b = d;
            d = c;
            fd = fc;
            c = b - g * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + g * (b - a);
            fd = f(d);
        }
    }
    return 0.5 * (a + b);
}

/// Minimises f over [lo, hi] by a scan of n + 1 points and a golden-section refinement in the
/// bracket of the best scan point.
inline Real scan_min(const std::function<Real(Real)>& f, Real lo, Real hi, int n, Real tol) {
    Real best = lo;
    Real fbest = std::numeric_limits<Real>::infinity();
    const Real step = (hi - lo) / static_cast<Real>(n);
    for (int i = 0; i <= n; ++i) {
        const Real t = lo + step * static_cast<Real>(i);
        const Real v = f(t);
        if (v < fbest) {
            fbest = v;
            best = t;
        }
    }
    return golden_min(f, std::max(lo, best - step), std::min(hi, best + step), tol);
}

}  // namespace detail

/// Least-squares fit of y(rho) = out w(k rho) + in w(k (2 a - rho)) + sum_q c_q exp(-j k0 rho)
/// (k0 rho)^(-p_q) (see the file comment): the amplitudes by column-scaled QR for each k, Re k
/// (and optionally Im k) by minimising the relative residual (scan of +-k_search with 160 steps,
/// golden-section refinement to 1e-9 relative).
/// @throws std::invalid_argument for rho and y of different sizes, fewer than 2 + (2 + number
///         of background terms) points, rho not in (0, a] (relative slack 1e-9), a non-finite
///         sample, k0 <= 0, Re k_guess <= 0 or k_search outside (0, 0.5); hankel() throws when
///         k rho or k0 rho is below kMinHankelArgument.
[[nodiscard]] inline FitResult fit_spp(const std::vector<Real>& rho, const std::vector<Complex>& y,
                                       const FitOptions& o) {
    if (rho.size() != y.size())
        throw std::invalid_argument("pml_spp::fit_spp: rho and y differ in size");
    if (rho.size() < 4 + o.background_powers.size())
        throw std::invalid_argument("pml_spp::fit_spp: too few points for the model");
    if (!(std::isfinite(o.plane) && o.plane > 0.0) || !(std::isfinite(o.k0) && o.k0 > 0.0) ||
        !(o.k_guess.real() > 0.0) || !(o.k_search > 0.0 && o.k_search < 0.5))
        throw std::invalid_argument("pml_spp::fit_spp: invalid plane, k0, k_guess or k_search");
    for (std::size_t i = 0; i < rho.size(); ++i) {
        if (!(rho[i] > 0.0 && rho[i] <= o.plane * (1.0 + 1e-9)))
            throw std::invalid_argument("pml_spp::fit_spp: rho must lie in (0, plane]");
        if (!(std::isfinite(y[i].real()) && std::isfinite(y[i].imag())))
            throw std::invalid_argument("pml_spp::fit_spp: non-finite sample");
    }
    Complex k = o.k_guess;
    if (o.fit_k) {
        const Real kr0 = o.k_guess.real();
        const Real ki = o.k_guess.imag();
        const auto f_re = [&](Real t) {
            return detail::fit_linear(rho, y, o, Complex(kr0 * (1.0 + t), ki), nullptr);
        };
        const Real t = detail::scan_min(f_re, -o.k_search, o.k_search, 160, 1e-9);
        k = Complex(kr0 * (1.0 + t), ki);
        if (o.fit_imag) {
            const Real kr = k.real();
            const auto f_im = [&](Real u) {
                return detail::fit_linear(rho, y, o, Complex(kr, kr * u), nullptr);
            };
            const Real u = detail::scan_min(f_im, -0.02, 0.0, 80, 1e-10);
            k = Complex(kr, kr * u);
            // One more Re k pass at the fitted Im k.
            const auto f_re2 = [&](Real s) {
                return detail::fit_linear(rho, y, o, Complex(kr * (1.0 + s), k.imag()), nullptr);
            };
            const Real s = detail::golden_min(f_re2, -2e-3, 2e-3, 1e-10);
            k = Complex(kr * (1.0 + s), k.imag());
        }
    }
    FitResult res;
    detail::fit_linear(rho, y, o, k, &res);
    return res;
}

// ---------------------------------------------------------------------------------------------
// Sampling of the currents along the x-axis
// ---------------------------------------------------------------------------------------------

/// Samples of one quantity along the x-axis: signed positions x and complex values.
struct AxisSamples {
    std::vector<Real> x;
    std::vector<Complex> v;
};

/// The two vertices of interior edge n (the vertices of its plus triangle other than p_n^+).
[[nodiscard]] inline std::array<Index, 2> edge_vertices(const basis::RwgSpace& space, Index n) {
    const geometry::TriangleMesh& mesh = space.mesh();
    const Index t = space.plus_triangle(n);
    const Index free_vertex = space.plus_free_vertex(n);
    std::array<Index, 2> e{-1, -1};
    int c = 0;
    for (int a = 0; a < 3; ++a) {
        const Index v = mesh.triangles()(t, a);
        if (v != free_vertex && c < 2)
            e[static_cast<std::size_t>(c++)] = v;
    }
    if (c != 2)
        throw std::logic_error("pml_spp::edge_vertices: inconsistent RWG support");
    return e;
}

[[nodiscard]] inline Vec3 edge_midpoint(const basis::RwgSpace& space, Index n) {
    const std::array<Index, 2> e = edge_vertices(space, n);
    const Vertices& V = space.mesh().vertices();
    return 0.5 * (V.row(e[0]).transpose() + V.row(e[1]).transpose());
}

/// Current of block `block` (0: J, 1: M) of the solution x at point r on triangle t.
[[nodiscard]] inline Vec3c current_at(const basis::RwgSpace& space, const VectorXc& x, int block,
                                      Index t, const Vec3& r) {
    const Index N = space.size();
    const basis::RwgSpace::Support sup = space.support(t);
    Vec3c c = Vec3c::Zero();
    for (int a = 0; a < sup.count; ++a)
        c += x(static_cast<Index>(block) * N + sup.n[a]) *
             space.value(sup.n[a], t, r).cast<Complex>();
    return c;
}

/// Component `component` (0: x, 1: y) of the current of block `block` at the midpoints of the
/// edges perpendicular to that component (projected onto z = 0) whose midpoint lies within
/// |y| <= 0.51 h of the x-axis; edges at the same x (within 1e-3 h) are averaged (|y| = h / 2
/// above and below an axis on a grid line). The normal component across an edge is continuous
/// for an RWG expansion, so the value is the same from both support triangles.
[[nodiscard]] inline AxisSamples edge_current_on_axis(const basis::RwgSpace& space,
                                                      const VectorXc& x, int block, int component,
                                                      Real h) {
    if (block != 0 && block != 1)
        throw std::invalid_argument("pml_spp::edge_current_on_axis: block must be 0 or 1");
    if (component != 0 && component != 1)
        throw std::invalid_argument("pml_spp::edge_current_on_axis: component must be 0 or 1");
    if (x.size() != 2 * space.size())
        throw std::invalid_argument("pml_spp::edge_current_on_axis: solution size mismatch");
    const Vertices& V = space.mesh().vertices();
    const int along = 1 - component;  // an edge perpendicular to x runs along y
    std::vector<std::pair<Real, Complex>> raw;
    for (Index n = 0; n < space.size(); ++n) {
        const std::array<Index, 2> e = edge_vertices(space, n);
        const Vec3 a = V.row(e[0]).transpose();
        const Vec3 b = V.row(e[1]).transpose();
        if (std::abs(a(component) - b(component)) > 1e-6 * h)
            continue;  // not perpendicular to the component
        if (!(std::abs(a(along) - b(along)) > 0.5 * h))
            continue;
        const Vec3 mid = 0.5 * (a + b);
        if (std::abs(mid.y()) > 0.51 * h)
            continue;
        const Vec3c c = current_at(space, x, block, space.plus_triangle(n), mid);
        raw.emplace_back(mid.x(), c(component));
    }
    std::sort(raw.begin(), raw.end(),
              [](const auto& p, const auto& q) { return p.first < q.first; });
    AxisSamples s;
    std::size_t i = 0;
    while (i < raw.size()) {
        std::size_t k = i + 1;
        Complex sum = raw[i].second;
        while (k < raw.size() && raw[k].first - raw[i].first < 1e-3 * h) sum += raw[k++].second;
        s.x.push_back(raw[i].first);
        s.v.push_back(sum / static_cast<Real>(k - i));
        i = k;
    }
    return s;
}

/// Surface divergence of the current of block `block` per triangle whose centroid lies within
/// |y| <= 0.5 h of the x-axis, at the centroid's x (for J: -j omega eps_1 E_n on the R1 side).
[[nodiscard]] inline AxisSamples divergence_on_axis(const basis::RwgSpace& space, const VectorXc& x,
                                                    int block, Real h) {
    if (x.size() != 2 * space.size())
        throw std::invalid_argument("pml_spp::divergence_on_axis: solution size mismatch");
    const geometry::TriangleMesh& mesh = space.mesh();
    const Index N = space.size();
    std::vector<std::pair<Real, Complex>> raw;
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        const Vec3 c = mesh.centroid(t);
        if (std::abs(c.y()) > 0.5 * h)
            continue;
        const basis::RwgSpace::Support sup = space.support(t);
        Complex d = 0.0;
        for (int a = 0; a < sup.count; ++a)
            d += x(static_cast<Index>(block) * N + sup.n[a]) * space.divergence(sup.n[a], t);
        raw.emplace_back(c.x(), d);
    }
    std::sort(raw.begin(), raw.end(),
              [](const auto& p, const auto& q) { return p.first < q.first; });
    AxisSamples s;
    for (const auto& [px, v] : raw) {
        s.x.push_back(px);
        s.v.push_back(v);
    }
    return s;
}

/// The samples on one half-axis (side +1: x > 0, -1: x < 0) with rho = |x| in [rho_min, rho_max]
/// (relative slack 1e-9 at rho_max).
inline void select_window(const AxisSamples& s, int side, Real rho_min, Real rho_max,
                          std::vector<Real>& rho, std::vector<Complex>& v) {
    rho.clear();
    v.clear();
    for (std::size_t i = 0; i < s.x.size(); ++i) {
        const Real r = static_cast<Real>(side) * s.x[i];
        if (r >= rho_min && r <= rho_max * (1.0 + 1e-9)) {
            rho.push_back(r);
            v.push_back(s.v[i]);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The 7b case
// ---------------------------------------------------------------------------------------------

/// Which sources a run solves.
enum class Launch {
    Groove,  ///< grooved minus flat solution (two operators) and the localized source
    Source,  ///< the localized source on the flat operator only (one operator)
};

/// One run: geometry, PML profile, launch, measurement.
struct SppCase {
    Real patch = 1.0e-6;             ///< L [m]: a_0 = L / 2 (holds the groove)
    Real taper = 0.5e-6;             ///< W_t [m] (flat here; >= groove_radius = correlation length)
    Real pml_start = 2.0e-6;         ///< a_2 [m]; the gap is a_2 - L / 2 - W_t
    Real pml_width = 1.0e-6;         ///< W_pml [m] (2 lambda_1 = the current default)
    Real mesh_size = kLambda / 6.0;  ///< h [m]; must divide L / 2, W_t, the gap and W_pml
    Real sigma_max = 4.0;            ///< PmlTermination::sigma_max
    Real order = 2.0;                ///< PmlTermination::order
    Launch launch = Launch::Groove;
    Real groove_depth = 50e-9;        ///< d [m] (> 0: dimple into the metal; lambda / 10)
    Real groove_radius = 100e-9;      ///< r_g [m], 1/e radius (lambda / 5)
    Real source_radius = 150e-9;      ///< localized source: edge midpoints within this radius
    Real waist = 0.5e-6;              ///< beam w0 [m] (lambda)
    Real dark_band_tolerance = 1e-2;  ///< see make_termination()
    Real target_accuracy = 1e-5;      ///< kernels::OperatorOptions::target_accuracy
    int stretched_touching_points = 10;
    Real rho_min = 0.6e-6;  ///< fit window start [m] (k_spp rho_min = 8.0)
    std::vector<Real> background_powers = {1.0, 1.5, 2.0};
    bool fit_imag = false;
    bool use_lapack = true;
};

/// The validation 7b case (benchmarks/results/pml_validation.md, "7b SPP absorption"): the PML
/// defaults (sigma_max 4, order 2, W_pml = 2 lambda = 1 um), groove launch (d = 50 nm,
/// r_g = 100 nm, rigorous beam w0 = lambda at normal incidence), h = lambda / 5 = 100 nm, patch
/// 1 um, taper 0.5 um, PML start 1.5 um (fit window 0.6 .. 1.5 um), 2N = 14 800.
[[nodiscard]] inline SppCase seven_b_case() {
    SppCase c;
    c.patch = 1.0e-6;
    c.taper = 0.5e-6;
    c.pml_start = 1.5e-6;
    c.pml_width = 2.0 * kLambda;
    c.mesh_size = kLambda / 5.0;
    c.sigma_max = 4.0;
    c.order = 2.0;
    c.launch = Launch::Groove;
    return c;
}

/// Height map of a case: the grid of the patch plus taper with spacing h, the groove at the
/// origin (zero for the flat surface).
[[nodiscard]] inline geometry::HeightMap height_map(const SppCase& c, bool grooved) {
    geometry::HeightMap m = fresnel_flat::flat_height_map(c.patch + 2.0 * c.taper, c.mesh_size);
    if (!grooved)
        return m;
    const Real half_x = static_cast<Real>(m.z.rows() - 1) * m.dx / 2.0;
    const Real half_y = static_cast<Real>(m.z.cols() - 1) * m.dy / 2.0;
    for (Index i = 0; i < m.z.rows(); ++i) {
        for (Index jj = 0; jj < m.z.cols(); ++jj) {
            const Real x = -half_x + static_cast<Real>(i) * m.dx;
            const Real y = -half_y + static_cast<Real>(jj) * m.dy;
            m.z(i, jj) =
                c.groove_depth * std::exp(-(x * x + y * y) / (c.groove_radius * c.groove_radius));
        }
    }
    return m;
}

[[nodiscard]] inline geometry::OpenSurfaceMesh make_open_mesh(const SppCase& c, bool grooved) {
    geometry::OpenSurfaceParams op;
    op.patch_size = c.patch;
    op.taper_width = c.taper;
    op.gap_width = c.pml_start - 0.5 * c.patch - c.taper;
    op.pml_width = c.pml_width;
    if (!(op.gap_width > 0.0))
        throw std::invalid_argument("pml_spp: pml_start must exceed patch / 2 + taper");
    geometry::OpenSurfaceMesh m = geometry::make_open_surface_mesh(height_map(c, grooved), op);
    // The fit plane and the profile scan need the requested zones exactly (no grid rounding).
    if (std::abs(m.pml_start - c.pml_start) > 1e-6 * c.mesh_size ||
        std::abs(m.outer_half - m.pml_start - c.pml_width) > 1e-6 * c.mesh_size)
        throw std::invalid_argument(
            "pml_spp: pml_start or pml_width is not on the mesh grid (h must divide L / 2, W_t, "
            "the gap and W_pml)");
    return m;
}

/// PML termination of a case. The dark band defaults to 1e-2 here, not 1e-4: neither launch
/// sees the beam at the PML start (the masked-incidence errors are common to the grooved and the
/// flat solve and cancel in the difference; the localized source is zero there); the measured
/// ratio is reported.
[[nodiscard]] inline PmlTermination make_termination(const SppCase& c) {
    PmlTermination p;
    p.sigma_max = c.sigma_max;
    p.order = c.order;
    p.correlation_length = c.groove_radius;
    p.dark_band_tolerance = c.dark_band_tolerance;
    return p;
}

[[nodiscard]] inline std::shared_ptr<excitation::AngularSpectrumBeam> make_beam(const SppCase& c,
                                                                                Real pml_start) {
    excitation::AngularSpectrumBeam::Params bp;
    bp.wavelength = kLambda;
    bp.waist_radius = c.waist;
    bp.focus = Vec3::Zero();
    bp.incidence_angle = 0.0;
    bp.polarization = excitation::Polarization::P;  // E along x at normal incidence
    bp.region_radius = 1.02 * std::sqrt(2.0) * pml_start;
    return std::make_shared<excitation::AngularSpectrumBeam>(bp, material::vacuum());
}

[[nodiscard]] inline SimulationConfig make_config(const SppCase& c) {
    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.exterior = material::vacuum();
    cfg.object = material::silver_500nm();
    cfg.compression = "dense";
    cfg.solver = SolverKind::Direct;
    cfg.kernels.target_accuracy = c.target_accuracy;
    cfg.kernels.stretched_touching_points = c.stretched_touching_points;
    return cfg;
}

/// Mesh size of a case without assembly.
struct Size {
    Index triangles = 0;
    Index unknowns = 0;          ///< 2N
    std::array<Real, 4> half{};  ///< a_0 .. a_3 [m]
    Real dense_bytes = 0;        ///< 16 (2N)^2: the peak with the in-place LU
};

[[nodiscard]] inline Size size_of(const SppCase& c) {
    const geometry::OpenSurfaceMesh om = make_open_mesh(c, false);
    Size s;
    s.triangles = om.mesh.num_triangles();
    s.unknowns = 2 * om.mesh.num_edges();
    s.half = {om.patch_half, om.taper_end, om.pml_start, om.outer_half};
    s.dense_bytes = 16.0 * static_cast<Real>(s.unknowns) * static_cast<Real>(s.unknowns);
    return s;
}

/// LU factorisation in place: LAPACK zgetrf when available, else Eigen's PartialPivLU on an
/// Eigen::Ref of the moved-in storage (Eigen's in-place decomposition). Either way the peak is the
/// one matrix (Size::dense_bytes), which the validation-large memory guard budgets.
class InPlaceLu {
public:
    InPlaceLu(MatrixXc&& Z, bool use_lapack) : n_(Z.rows()) {
        if (Z.rows() != Z.cols())
            throw std::invalid_argument("InPlaceLu: matrix must be square");
#ifdef SPECKLEBEM_HAVE_BLAS_LAPACK
        lapack_ = use_lapack;
#else
        (void)use_lapack;
#endif
        lu_ = std::move(Z);
        if (lapack_) {
#ifdef SPECKLEBEM_HAVE_BLAS_LAPACK
            if (n_ > static_cast<Index>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("InPlaceLu: n too large for LP64 LAPACK");
            const int n = static_cast<int>(n_);
            ipiv_.resize(static_cast<std::size_t>(n_));
            int info = 0;
            zgetrf_(&n, &n, lu_.data(), &n, ipiv_.data(), &info);
            if (info != 0)
                throw std::runtime_error("InPlaceLu: zgetrf failed, info = " +
                                         std::to_string(info));
#endif
        } else {
            // Factorises lu_ itself (no copy); lu_ is declared before eigen_ and outlives it.
            eigen_ = std::make_unique<Eigen::PartialPivLU<Eigen::Ref<MatrixXc>>>(lu_);
        }
    }
    InPlaceLu(const InPlaceLu&) = delete;
    InPlaceLu& operator=(const InPlaceLu&) = delete;
    InPlaceLu(InPlaceLu&&) = delete;  // eigen_ refers to the storage of lu_
    InPlaceLu& operator=(InPlaceLu&&) = delete;
    ~InPlaceLu() = default;
    [[nodiscard]] VectorXc solve(const VectorXc& b) const {
        if (b.size() != n_)
            throw std::invalid_argument("InPlaceLu: right-hand side size mismatch");
        if (!lapack_)
            return eigen_->solve(b);
        VectorXc x = b;
#ifdef SPECKLEBEM_HAVE_BLAS_LAPACK
        const int n = static_cast<int>(n_);
        const int nrhs = 1;
        int info = 0;
        const char trans = 'N';
        zgetrs_(&trans, &n, &nrhs, lu_.data(), &n, ipiv_.data(), x.data(), &n, &info, 1);
        if (info != 0)
            throw std::runtime_error("InPlaceLu: zgetrs failed, info = " + std::to_string(info));
#endif
        return x;
    }
    [[nodiscard]] bool lapack() const { return lapack_; }

private:
    Index n_;
    bool lapack_ = false;
    MatrixXc lu_;
    std::vector<int> ipiv_;
    std::unique_ptr<Eigen::PartialPivLU<Eigen::Ref<MatrixXc>>> eigen_;
};

/// Fit of one quantity on one half-axis.
struct QuantityFit {
    std::string source;    ///< "groove" or "local"
    std::string quantity;  ///< "Jx", "My", "divJ"
    int side = 1;          ///< +1: x > 0, -1: x < 0
    FitResult fit;
};

/// The measured quantities of the file comment for the solution x on the space.
struct AxisData {
    AxisSamples jx, my, divj;
};

[[nodiscard]] inline AxisData axis_data(const basis::RwgSpace& space, const VectorXc& x, Real h) {
    return {edge_current_on_axis(space, x, 0, 0, h), edge_current_on_axis(space, x, 1, 1, h),
            divergence_on_axis(space, x, 0, h)};
}

/// Fits every quantity of d on both half-axes (window [rho_min, a_2]).
inline void fit_all(const SppCase& c, const std::string& source, const AxisData& d, Real plane,
                    Real k0, Complex k_spp, std::vector<QuantityFit>& out) {
    const std::array<std::pair<const char*, const AxisSamples*>, 3> q = {
        {{"Jx", &d.jx}, {"My", &d.my}, {"divJ", &d.divj}}};
    for (const auto& [name, s] : q) {
        FitOptions o;
        o.form = std::string(name) == "divJ" ? WaveForm::H1 : WaveForm::H1Prime;
        o.plane = plane;
        o.k0 = k0;
        o.k_guess = k_spp;
        o.background_powers = c.background_powers;
        o.fit_imag = c.fit_imag;
        for (const int side : {1, -1}) {
            std::vector<Real> rho;
            std::vector<Complex> v;
            select_window(*s, side, c.rho_min, plane, rho, v);
            out.push_back({source, name, side, fit_spp(rho, v, o)});
        }
    }
}

/// Result of one run.
struct RunResult {
    Size size;
    Index masked = 0;
    std::string formulation;
    bool lapack = false;
    Real dark_band = 0;
    Complex k_spp{};  ///< analytic
    Real k0 = 0;
    Real assembly_s = 0;  ///< all operators and right-hand sides
    Real lu_s = 0;
    Real solve_s = 0;
    Real post_s = 0;
    Real peak_rss = 0;
    Real residual = 0;  ///< largest |b - Z x| / |b| on the kept rows over all solves
    std::vector<QuantityFit> fits;
    /// |r_spp| of validation 7b: the largest |r| of J_x and M_y over both half-axes of the
    /// primary source (groove for Launch::Groove, else local).
    Real r_spp = 0;
    std::string report;
};

namespace detail {

/// Assembles the dense operator of sim's problem, factors it in place and solves the right-hand
/// sides. Rows kept before the factorisation give the residual.
inline std::vector<VectorXc> solve_dense(const Simulation& sim, const std::vector<VectorXc>& rhs,
                                         bool use_lapack, RunResult& r) {
    using Clock = std::chrono::steady_clock;
    const auto secs = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<Real>(b - a).count();
    };
    const auto t0 = Clock::now();
    std::shared_ptr<op::LinearOperator> lin = op::DenseStrategy{}.build(sim.problem());
    const auto dense = std::dynamic_pointer_cast<op::DenseOperator>(lin);
    if (!dense)
        throw std::logic_error("pml_spp: the dense strategy did not return a DenseOperator");
    MatrixXc Z = std::move(dense->matrix());
    lin.reset();
    const Index n = Z.rows();
    const Index stride = std::max<Index>(1, n / 64);
    std::vector<Index> rows;
    for (Index i = stride / 2; i < n; i += stride) rows.push_back(i);
    MatrixXc Zs(static_cast<Index>(rows.size()), n);
    for (std::size_t i = 0; i < rows.size(); ++i) Zs.row(static_cast<Index>(i)) = Z.row(rows[i]);
    const auto t1 = Clock::now();
    r.assembly_s += secs(t0, t1);
    std::vector<VectorXc> x;
    {
        const InPlaceLu lu(std::move(Z), use_lapack);
        r.lapack = lu.lapack();
        const auto t2 = Clock::now();
        r.lu_s += secs(t1, t2);
        for (const VectorXc& b : rhs) {
            x.push_back(lu.solve(b));
            VectorXc bs(static_cast<Index>(rows.size()));
            for (std::size_t i = 0; i < rows.size(); ++i) bs(static_cast<Index>(i)) = b(rows[i]);
            const Real nb = bs.norm();
            if (nb > 0.0)
                r.residual = std::max(r.residual, (bs - Zs * x.back()).norm() / nb);
        }
        r.solve_s += secs(t2, Clock::now());
    }
    return x;
}

}  // namespace detail

/// The localized source of a flat Simulation: its beam right-hand side b restricted to the test
/// functions whose edge midpoint lies within radius of the origin (J and M rows).
[[nodiscard]] inline VectorXc localized_rhs(const basis::RwgSpace& space, const VectorXc& b,
                                            Real radius) {
    const Index N = space.size();
    if (b.size() != 2 * N)
        throw std::invalid_argument("pml_spp::localized_rhs: size mismatch");
    VectorXc e = VectorXc::Zero(2 * N);
    Index kept = 0;
    for (Index n = 0; n < N; ++n) {
        if (edge_midpoint(space, n).norm() <= radius) {
            e(n) = b(n);
            e(N + n) = b(N + n);
            ++kept;
        }
    }
    if (kept == 0)
        throw std::invalid_argument("pml_spp::localized_rhs: no test function within radius");
    return e;
}

/// Writes the axis samples of a source as CSV lines "source,quantity,x,re,im".
inline void dump_samples(std::ofstream& f, const std::string& source, const AxisData& d) {
    const std::array<std::pair<const char*, const AxisSamples*>, 3> q = {
        {{"Jx", &d.jx}, {"My", &d.my}, {"divJ", &d.divj}}};
    char buf[200];
    for (const auto& [name, s] : q) {
        for (std::size_t i = 0; i < s->x.size(); ++i) {
            std::snprintf(buf, sizeof(buf), "%s,%s,%.9e,%.12e,%.12e\n", source.c_str(), name,
                          s->x[i], s->v[i].real(), s->v[i].imag());
            f << buf;
        }
    }
}

/// Builds, assembles, factorises and solves a case and fits r_spp. dump_path (optional) receives
/// the axis samples as CSV.
inline RunResult run(const SppCase& c, const std::string& dump_path = {}) {
    using Clock = std::chrono::steady_clock;
    RunResult r;
    r.size = size_of(c);
    const Real a2 = r.size.half[2];
    const SimulationConfig cfg = make_config(c);
    const PmlTermination term = make_termination(c);
    const auto beam = make_beam(c, a2);
    r.k0 = 2.0 * constants::pi / kLambda;
    r.k_spp = spp_wavenumber(r.k0, Complex(1.0, 0.0), cfg.object.eps_r);

    // Flat surface: beam and localized source with one LU.
    VectorXc x_flat, x_local;
    std::unique_ptr<Simulation> flat =
        std::make_unique<Simulation>(make_open_mesh(c, false), term, beam, cfg);
    r.dark_band = flat->dark_band_ratio();
    r.formulation = formulation::make_formulation(flat->formulation_kind())->name();
    {
        const std::vector<std::uint8_t> mask =
            op::pml_masked_functions(*flat->problem().space, flat->stretch());
        r.masked = static_cast<Index>(std::count(mask.begin(), mask.end(), std::uint8_t{1}));
    }
    const basis::RwgSpace& space = *flat->problem().space;
    {
        const auto t0 = Clock::now();
        const VectorXc b = op::assemble_rhs(flat->problem());
        r.assembly_s += std::chrono::duration<Real>(Clock::now() - t0).count();
        const VectorXc e = localized_rhs(space, b, c.source_radius);
        std::vector<VectorXc> x = detail::solve_dense(*flat, {b, e}, c.use_lapack, r);
        x_flat = std::move(x[0]);
        x_local = std::move(x[1]);
    }
    VectorXc x_groove;
    std::unique_ptr<Simulation> grooved;
    if (c.launch == Launch::Groove) {
        grooved = std::make_unique<Simulation>(make_open_mesh(c, true), term, beam, cfg);
        const basis::RwgSpace& gs = *grooved->problem().space;
        const Vertices& v0 = space.mesh().vertices();
        const Vertices& v1 = gs.mesh().vertices();
        if (gs.size() != space.size() || v1.rows() != v0.rows() ||
            !(v1.leftCols(2).array() == v0.leftCols(2).array()).all() ||
            !(gs.mesh().triangles().array() == space.mesh().triangles().array()).all())
            throw std::logic_error("pml_spp: grooved and flat meshes differ beyond z");
        const auto t0 = Clock::now();
        const VectorXc b = op::assemble_rhs(grooved->problem());
        r.assembly_s += std::chrono::duration<Real>(Clock::now() - t0).count();
        std::vector<VectorXc> x = detail::solve_dense(*grooved, {b}, c.use_lapack, r);
        x_groove = std::move(x[0]) - x_flat;
    }

    const auto t4 = Clock::now();
    std::ofstream dump;
    if (!dump_path.empty()) {
        dump.open(core::path_from_utf8(dump_path), std::ios::binary);
        if (!dump)
            throw std::runtime_error("pml_spp: cannot open " + dump_path);
        dump << "source,quantity,x,re,im\n";
    }
    if (c.launch == Launch::Groove) {
        // The difference lives on the flat space's numbering; sampled on the flat geometry
        // (identical outside the groove, which the window excludes).
        const AxisData d = axis_data(space, x_groove, c.mesh_size);
        fit_all(c, "groove", d, a2, r.k0, r.k_spp, r.fits);
        if (dump)
            dump_samples(dump, "groove", d);
    }
    {
        const AxisData d = axis_data(space, x_local, c.mesh_size);
        fit_all(c, "local", d, a2, r.k0, r.k_spp, r.fits);
        if (dump)
            dump_samples(dump, "local", d);
    }
    const std::string primary = c.launch == Launch::Groove ? "groove" : "local";
    for (const QuantityFit& f : r.fits)
        if (f.source == primary && (f.quantity == "Jx" || f.quantity == "My"))
            r.r_spp = std::max(r.r_spp, std::abs(f.fit.r()));
    r.post_s = std::chrono::duration<Real>(Clock::now() - t4).count();
    r.report = flat->report();
    r.peak_rss = system_memory::peak_rss_bytes();
    return r;
}

/// Reads a dump written by run() (CSV "source,quantity,x,re,im") back into AxisData per source,
/// in the order of first appearance, for refitting without a solve.
/// @throws std::runtime_error if the file cannot be opened or a line is malformed.
[[nodiscard]] inline std::vector<std::pair<std::string, AxisData>> read_dump(
    const std::string& path) {
    std::ifstream f(core::path_from_utf8(path), std::ios::binary);
    if (!f)
        throw std::runtime_error("pml_spp::read_dump: cannot open " + path);
    std::vector<std::pair<std::string, AxisData>> out;
    std::string line;
    std::getline(f, line);  // header
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty())
            continue;
        std::vector<std::string> cols;
        std::size_t start = 0;
        while (true) {
            const std::size_t comma = line.find(',', start);
            cols.push_back(line.substr(start, comma - start));
            if (comma == std::string::npos)
                break;
            start = comma + 1;
        }
        if (cols.size() != 5)
            throw std::runtime_error("pml_spp::read_dump: malformed line: " + line);
        auto it =
            std::find_if(out.begin(), out.end(), [&](const auto& p) { return p.first == cols[0]; });
        if (it == out.end()) {
            out.emplace_back(cols[0], AxisData{});
            it = out.end() - 1;
        }
        AxisSamples* s = cols[1] == "Jx"     ? &it->second.jx
                         : cols[1] == "My"   ? &it->second.my
                         : cols[1] == "divJ" ? &it->second.divj
                                             : nullptr;
        if (!s)
            throw std::runtime_error("pml_spp::read_dump: unknown quantity " + cols[1]);
        s->x.push_back(std::stod(cols[2]));
        s->v.emplace_back(std::stod(cols[3]), std::stod(cols[4]));
    }
    return out;
}

/// One fit (source, quantity, side) of a list of fits; throws if absent.
[[nodiscard]] inline const FitResult& find_fit(const std::vector<QuantityFit>& fits,
                                               const std::string& source,
                                               const std::string& quantity, int side) {
    for (const QuantityFit& f : fits)
        if (f.source == source && f.quantity == quantity && f.side == side)
            return f.fit;
    throw std::invalid_argument("pml_spp::find_fit: no fit " + source + "/" + quantity);
}

/// One fit (source, quantity, side) of a run; throws if absent.
[[nodiscard]] inline const FitResult& find_fit(const RunResult& r, const std::string& source,
                                               const std::string& quantity, int side) {
    return find_fit(r.fits, source, quantity, side);
}

/// |out(M_y)| / |out(J_x)| [Ohm] of one source and half-axis: spp_impedance() for a clean SPP
/// (file comment). A smaller value means that the outgoing J_x column carries non-SPP field, so
/// that its r = in / out is not normalised by the SPP amplitude.
/// @throws std::invalid_argument if a fit is absent or out(J_x) = 0.
[[nodiscard]] inline Real out_impedance(const std::vector<QuantityFit>& fits,
                                        const std::string& source, int side) {
    const Real jx = std::abs(find_fit(fits, source, "Jx", side).out);
    const Real my = std::abs(find_fit(fits, source, "My", side).out);
    if (!(jx > 0.0))
        throw std::invalid_argument("pml_spp::out_impedance: out(J_x) = 0");
    return my / jx;
}

/// One machine-readable line per source of a run.
[[nodiscard]] inline std::string row_line(const SppCase& c, const RunResult& r,
                                          const std::string& source) {
    const FitResult jp = find_fit(r, source, "Jx", 1);
    const FitResult jm = find_fit(r, source, "Jx", -1);
    const FitResult mp = find_fit(r, source, "My", 1);
    const FitResult mm = find_fit(r, source, "My", -1);
    const FitResult dp = find_fit(r, source, "divJ", 1);
    const FitResult dm = find_fit(r, source, "divJ", -1);
    std::string bg;
    for (std::size_t i = 0; i < c.background_powers.size(); ++i)
        bg += (i ? "," : "") + std::to_string(c.background_powers[i]).substr(0, 4);
    if (bg.empty())
        bg = "none";
    char buf[2000];
    const int len = std::snprintf(
        buf, sizeof(buf),
        "ROW | spp | %s | h %.4g | sigma %.3g | m %.3g | pml %.4g | half %.4g/%.4g/%.4g/%.4g | "
        "groove %.3g/%.3g | src %.3g | rho_min %.3g | bg %s | fit_im %d | acc %.1e | stp %d | 2N "
        "%lld | masked %lld of %lld | dark %.2e | |r| Jx %.3e %.3e | My %.3e %.3e | divJ %.3e "
        "%.3e | arg Jx+ %+.4f | k/kspp Jx+ %.5f%+.5fj | |out My/Jx|+ %.1f Ohm (SPP %.1f) | res Jx "
        "%.2e %.2e | n %lld | lin res %.1e | %s | assembly %.1f s | LU %.1f s | solve %.1f s | "
        "post %.1f s | peak %.2f GB",
        source.c_str(), c.mesh_size, c.sigma_max, c.order, c.pml_width, r.size.half[0],
        r.size.half[1], r.size.half[2], r.size.half[3], c.groove_depth, c.groove_radius,
        c.source_radius, c.rho_min, bg.c_str(), c.fit_imag ? 1 : 0, c.target_accuracy,
        c.stretched_touching_points, static_cast<long long>(r.size.unknowns),
        static_cast<long long>(r.masked), static_cast<long long>(r.size.unknowns / 2), r.dark_band,
        std::abs(jp.r()), std::abs(jm.r()), std::abs(mp.r()), std::abs(mm.r()), std::abs(dp.r()),
        std::abs(dm.r()), std::arg(jp.r()), (jp.k / r.k_spp).real(), (jp.k / r.k_spp).imag(),
        out_impedance(r.fits, source, 1), spp_impedance(r.k0, r.k_spp), jp.residual, jm.residual,
        static_cast<long long>(jp.points), r.residual, r.lapack ? "lapack" : "eigen", r.assembly_s,
        r.lu_s, r.solve_s, r.post_s, r.peak_rss / 1e9);
    if (len < 0 || static_cast<std::size_t>(len) >= sizeof(buf))
        throw std::logic_error("pml_spp::row_line: formatting failed or line truncated");
    return buf;
}

}  // namespace pml_spp
