#include "specklebem/excitation/angular_spectrum_beam.hpp"

#include "specklebem/core/logging.hpp"
#include "specklebem/kernels/fast_math.hpp"
#include "specklebem/kernels/quadrature.hpp"

#include <Eigen/Geometry>  // cross products of real vectors

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

#include "excitation_detail.hpp"

namespace specklebem::excitation {

namespace {

/// Accumulation width: independent partial sums per lane, so the wave loop has no serial
/// floating-point dependency and the compiler may vectorise it without reassociation.
constexpr std::size_t kLanes = 4;
constexpr int kMaxRounds = 40;
constexpr int kShellDirections = 32;

std::string sci(Real x) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3g", x);
    return buf;
}

/// Callers keep n <= AngularSpectrumBeam::kMaxOrder (or refine() of it), far from INT_MAX.
int round_up4(int n) {
    return 4 * ((n + 3) / 4);
}

/// n -> 1.5 n + 8 (the refinement step of the WP-V1 study); n <= kMaxOrder, so no overflow.
int refine(int n) {
    return n + n / 2 + 8;
}

/// Probe offsets in the beam frame (u, v, k0 components): the focus and two Fibonacci shells.
std::vector<Vec3> probe_offsets(Real radius) {
    std::vector<Vec3> d{Vec3::Zero()};
    const Real golden = constants::pi * (3.0 - std::sqrt(5.0));
    for (const Real rad : {0.5 * radius, radius}) {
        for (int i = 0; i < kShellDirections; ++i) {
            const Real z = 1.0 - (2.0 * i + 1.0) / kShellDirections;
            const Real s = std::sqrt(1.0 - z * z);
            const Real ph = golden * i;
            d.emplace_back(rad * s * std::cos(ph), rad * s * std::sin(ph), rad * z);
        }
    }
    return d;
}

}  // namespace

AngularSpectrumBeam::AngularSpectrumBeam(Params p, material::Material background)
    : Excitation(detail::omega_from_wavelength(p.wavelength, "AngularSpectrumBeam"), background),
      p_(std::move(p)) {
    constexpr const char* kWho = "AngularSpectrumBeam";
    detail::require_lossless(background_, kWho);
    if (!(p_.waist_radius > 0) || !std::isfinite(p_.waist_radius)) {
        throw std::invalid_argument("AngularSpectrumBeam: waist_radius must be finite and > 0");
    }
    if (!detail::all_finite(p_.focus)) {
        throw std::invalid_argument("AngularSpectrumBeam: focus must be finite");
    }
    if (!(std::abs(p_.incidence_angle) < constants::pi / 2)) {
        throw std::invalid_argument(
            "AngularSpectrumBeam: |incidence_angle| must be < pi/2 (beam travelling towards +z)");
    }
    if (!(p_.tolerance > 0) || !(p_.tolerance <= 1e-2)) {
        throw std::invalid_argument("AngularSpectrumBeam: tolerance must be in (0, 1e-2]");
    }
    if (!(p_.region_radius >= 0) || !std::isfinite(p_.region_radius)) {
        throw std::invalid_argument("AngularSpectrumBeam: region_radius must be finite and >= 0");
    }
    const bool fixed = p_.polar_order != 0 || p_.azimuth_order != 0;
    if (fixed && (p_.polar_order < 1 || p_.azimuth_order < 1)) {
        throw std::invalid_argument(
            "AngularSpectrumBeam: polar_order and azimuth_order must both be >= 1 (fixed grid) "
            "or both 0 (automatic)");
    }
    if (p_.polar_order > kMaxOrder || p_.azimuth_order > kMaxOrder) {
        throw std::invalid_argument(
            "AngularSpectrumBeam: polar_order and azimuth_order must be <= " +
            std::to_string(kMaxOrder));
    }
    if (p_.max_plane_waves < 1) {
        throw std::invalid_argument("AngularSpectrumBeam: max_plane_waves must be >= 1");
    }

    // Beam frame (as GaussianBeam): R_y(theta) = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]].
    const Real c = std::cos(p_.incidence_angle);
    const Real s = std::sin(p_.incidence_angle);
    k0_hat_ = Vec3(s, 0, c);
    u_hat_ = Vec3(c, 0, -s);
    e0_hat_ = (p_.polarization == Polarization::P) ? u_hat_ : Vec3(Vec3::UnitY());
    k_ = background_.wavenumber(omega_).real();  // real: the background is lossless
    eta_ = background_.wave_impedance(omega_).real();
    region_radius_ = p_.region_radius > 0 ? p_.region_radius : 4.0 * p_.waist_radius;
    warn_radius_ = 1.2 * region_radius_;  // before the probes are evaluated (all within R)

    // Spectrum truncation: A(k sin(alpha_max)) = 1e-3 tolerance; the field error of dropping
    // k_t > K is A(K) relative to the peak (int_K^inf exp(-k_t^2 w0^2/4) k_t dk_t / int_0^inf).
    const Real k_cut = 2.0 * std::sqrt(std::log(1e3 / p_.tolerance)) / p_.waist_radius;
    const bool clipped = k_cut >= k_;
    alpha_max_ = clipped ? 0.5 * constants::pi : std::asin(k_cut / k_);

    // Probe points in the lab frame.
    std::vector<Vec3> probes;
    for (const Vec3& d : probe_offsets(region_radius_)) {
        probes.push_back(p_.focus + d.x() * u_hat_ + d.y() * Vec3::UnitY() + d.z() * k0_hat_);
    }
    const auto evaluate_probes = [&](const Grid& g) {
        std::vector<Vec3c> e;
        e.reserve(probes.size());
        for (const Vec3& r : probes) e.push_back(probe_field(g, r));
        return e;
    };
    const auto change = [](const std::vector<Vec3c>& a, const std::vector<Vec3c>& b) {
        Real dmax = 0, emax = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            dmax = std::max(dmax, (a[i] - b[i]).norm());
            emax = std::max(emax, b[i].norm());
        }
        return dmax / emax;
    };

    const Real kr = k_ * region_radius_;
    if (fixed) {
        const int na = p_.polar_order;
        const int np = round_up4(p_.azimuth_order);
        const int na_cmp = refine(na);
        const int np_cmp = round_up4(refine(np));
        if (static_cast<Index>(na_cmp) * np_cmp > p_.max_plane_waves) {
            throw std::invalid_argument(
                "AngularSpectrumBeam: the check grid " + std::to_string(na_cmp) + " x " +
                std::to_string(np_cmp) + " of the fixed grid " + std::to_string(na) + " x " +
                std::to_string(np) +
                " exceeds max_plane_waves = " + std::to_string(p_.max_plane_waves));
        }
        grid_ = build(na, np);
        grid_change_ = change(evaluate_probes(grid_), evaluate_probes(build(na_cmp, np_cmp)));
        if (!(grid_change_ < p_.tolerance)) {
            SBEM_WARN(
                "AngularSpectrumBeam: the fixed grid {} x {} changes E by {:.2e} >= tolerance "
                "{:.2e} against the 1.5x finer grid on |r - focus| <= {:.4g} m; the fields are "
                "not converged there (use automatic orders or raise them)",
                na, np, grid_change_, p_.tolerance, region_radius_);
        }
    } else {
        const auto fail = [&](const std::string& why) {
            throw std::runtime_error("AngularSpectrumBeam: plane-wave grid not converged to " +
                                     sci(p_.tolerance) + " (" + why +
                                     "; w0 = " + sci(p_.waist_radius) +
                                     " m, region_radius = " + sci(region_radius_) + " m)");
        };
        // Start below the phase-variation estimate (k R alpha_max / 2 Gauss-Legendre nodes,
        // Bessel orders up to k R sin(alpha_max) in azimuth) and refine each direction on its
        // own in steps of ~25 %: the quadrature error falls super-exponentially once resolved,
        // so the 1.5x refinement of the study overshoots by up to 2.25x in waves. The estimates
        // are checked in floating point before the conversion to int (huge k R).
        const Real na0 = std::ceil(0.25 * kr * alpha_max_) + 4;
        const Real np0 = std::ceil(0.5 * kr * std::sin(alpha_max_)) + 4;
        if (!(na0 <= kMaxOrder) || !(np0 <= kMaxOrder)) {
            fail("the start orders " + sci(na0) + " x " + sci(np0) + " exceed " +
                 std::to_string(kMaxOrder) + " (k region_radius = " + sci(kr) + ")");
        }
        int na = static_cast<int>(na0);
        int np = round_up4(static_cast<int>(np0));
        const auto step = [](int n) { return n + std::max(n / 4, 2); };
        bool converged = false;
        for (int round = 0; round < kMaxRounds && !converged; ++round) {
            if (na > kMaxOrder || np > kMaxOrder) {
                fail("the orders " + std::to_string(na) + " x " + std::to_string(np) + " exceed " +
                     std::to_string(kMaxOrder) + ", last change " + sci(grid_change_));
            }
            const int na_cmp = refine(na);
            const int np_cmp = round_up4(refine(np));
            if (static_cast<Index>(na_cmp) * np_cmp > p_.max_plane_waves) {
                fail("the check grid " + std::to_string(na_cmp) + " x " + std::to_string(np_cmp) +
                     " exceeds max_plane_waves, last change " + sci(grid_change_));
            }
            Grid base = build(na, np);
            const std::vector<Vec3c> e_base = evaluate_probes(base);
            const Real ca = change(e_base, evaluate_probes(build(na_cmp, np)));
            const Real cp = change(e_base, evaluate_probes(build(na, np_cmp)));
            grid_change_ = std::max(ca, cp);
            if (ca < 0.5 * p_.tolerance && cp < 0.5 * p_.tolerance) {
                // Both directions resolved: confirm against the grid refined in both.
                grid_change_ = change(e_base, evaluate_probes(build(na_cmp, np_cmp)));
                if (grid_change_ < p_.tolerance) {
                    grid_ = std::move(base);
                    converged = true;
                } else {
                    na = step(na);
                    np = round_up4(step(np));
                }
                continue;
            }
            if (ca >= 0.5 * p_.tolerance)
                na = step(na);
            if (cp >= 0.5 * p_.tolerance)
                np = round_up4(step(np));
        }
        if (!converged)
            fail("refinement rounds exhausted");
    }

    SBEM_INFO(
        "AngularSpectrumBeam: w0 = {:.4g} m, lambda = {:.4g} m, n1 = {:.4g}, theta_in = {:.4g} "
        "rad, {} plane waves ({} x {}), alpha_max = {:.4g} rad, region R = {:.4g} m, grid "
        "change {:.2e}{}",
        p_.waist_radius, p_.wavelength, k_ / (omega_ / constants::c0), p_.incidence_angle, grid_.n,
        grid_.n_alpha, grid_.n_phi, alpha_max_, region_radius_, grid_change_,
        fixed ? " (fixed grid)" : "");
    if (clipped) {
        SBEM_INFO(
            "AngularSpectrumBeam: spectrum cut at k_t = k (evanescent part omitted) where it is "
            "still exp(-(k w0)^2/4) = {:.3g} of its peak; the focal profile deviates from "
            "exp(-rho^2/w0^2) at that level",
            std::exp(-0.25 * k_ * k_ * p_.waist_radius * p_.waist_radius));
    }
    if (std::abs(p_.incidence_angle) + alpha_max_ > 0.5 * constants::pi) {
        SBEM_INFO(
            "AngularSpectrumBeam: |theta_in| + alpha_max = {:.4g} rad > pi/2: the outer plane "
            "waves of the cone around k0_hat travel towards -z (the beam as a whole travels "
            "towards +z)",
            std::abs(p_.incidence_angle) + alpha_max_);
    }
}

void AngularSpectrumBeam::warn_outside(Real distance) const {
    std::atomic<bool>& fired = outside_warned_.fired;
    if (fired.load(std::memory_order_relaxed) || fired.exchange(true, std::memory_order_relaxed)) {
        return;
    }
    SBEM_WARN(
        "AngularSpectrumBeam: field evaluated at |r - focus| = {:.4g} m > 1.2 region_radius = "
        "{:.4g} m, where the plane-wave sum is not controlled (azimuthal aliasing); raise "
        "region_radius to cover every evaluation point (warned once per beam)",
        distance, warn_radius_);
}

AngularSpectrumBeam::Grid AngularSpectrumBeam::build(int n_alpha, int n_phi) const {
    const kernels::LineRule gl = kernels::gauss_legendre(n_alpha);
    Grid g;
    g.n_alpha = n_alpha;
    g.n_phi = n_phi;
    g.n = static_cast<Index>(n_alpha) * static_cast<Index>(n_phi);
    const auto n_pad = static_cast<std::size_t>((g.n + static_cast<Index>(kLanes) - 1) /
                                                static_cast<Index>(kLanes)) *
                       kLanes;
    for (auto* v : {&g.kx, &g.ky, &g.kz, &g.ex, &g.ey, &g.ez, &g.hx, &g.hy, &g.hz}) {
        v->assign(n_pad, 0.0);
    }
    const Vec3 v_hat = Vec3::UnitY();
    const Real dphi = 2.0 * constants::pi / n_phi;
    const Real w0 = p_.waist_radius;
    Real norm = 0;   // sum W A (p . e0): E(focus) . e0 before normalisation
    Real power = 0;  // sum W A^2 |p|^2 cos(alpha)
    std::size_t i = 0;
    for (int ia = 0; ia < n_alpha; ++ia) {
        const auto ua = static_cast<std::size_t>(ia);
        const Real alpha = 0.5 * alpha_max_ * (gl.nodes[ua] + 1.0);
        const Real sa = std::sin(alpha);
        const Real ca = std::cos(alpha);
        const Real kt = k_ * sa;
        // Measure d^2 k_t = k^2 sin cos d alpha d phi.
        const Real weight = 0.5 * alpha_max_ * gl.weights[ua] * dphi * k_ * k_ * sa * ca;
        const Real a = std::exp(-0.25 * kt * kt * w0 * w0);
        for (int ip = 0; ip < n_phi; ++ip, ++i) {
            const Real phi = dphi * ip;
            const Vec3 kh = sa * (std::cos(phi) * u_hat_ + std::sin(phi) * v_hat) + ca * k0_hat_;
            const Vec3 pol = e0_hat_ - kh.dot(e0_hat_) * kh;
            const Vec3 e = weight * a * pol;
            const Vec3 h = kh.cross(e) / eta_;
            g.kx[i] = k_ * kh.x();
            g.ky[i] = k_ * kh.y();
            g.kz[i] = k_ * kh.z();
            g.ex[i] = e.x();
            g.ey[i] = e.y();
            g.ez[i] = e.z();
            g.hx[i] = h.x();
            g.hy[i] = h.y();
            g.hz[i] = h.z();
            norm += weight * a * pol.dot(e0_hat_);
            power += weight * a * a * pol.squaredNorm() * ca;
        }
    }
    for (auto* v : {&g.ex, &g.ey, &g.ez, &g.hx, &g.hy, &g.hz}) {
        for (Real& x : *v) x /= norm;
    }
    // Parseval: P = (2 pi)^2 / (2 eta) int |E~|^2 cos(alpha) d^2 k_t with E~ = A p / norm.
    g.power = 4.0 * constants::pi * constants::pi / (2.0 * eta_) * power / (norm * norm);
    return g;
}

template <std::size_t M>
void AngularSpectrumBeam::sum_waves(const Grid& g, const Vec3& r, const Real* const (&amp)[M],
                                    Complex (&out)[M]) const {
    const Vec3 d = r - p_.focus;
    const Real dx = d.x(), dy = d.y(), dz = d.z();
    const Real* kx = g.kx.data();
    const Real* ky = g.ky.data();
    const Real* kz = g.kz.data();
    const std::size_t n = g.kx.size();  // padded: a multiple of kLanes
    std::array<std::array<Real, kLanes>, M> re{};
    std::array<std::array<Real, kLanes>, M> im{};
    const Real dist = d.norm();
    if (dist > warn_radius_)
        warn_outside(dist);
    // |phase| <= k |d|: the branch-free sincos is valid up to kFastSincosMax (kernels/fast_math).
    if (k_ * dist <= kernels::fastmath::kFastSincosMax) {
        for (std::size_t i = 0; i < n; i += kLanes) {
            for (std::size_t l = 0; l < kLanes; ++l) {
                const std::size_t j = i + l;
                Real sn = 0, cs = 0;
                kernels::fastmath::fast_sincos(kx[j] * dx + ky[j] * dy + kz[j] * dz, sn, cs);
                for (std::size_t m = 0; m < M; ++m) {  // exp(-j phase) = cos - j sin
                    re[m][l] += amp[m][j] * cs;
                    im[m][l] -= amp[m][j] * sn;
                }
            }
        }
    } else {
        for (std::size_t j = 0; j < n; ++j) {
            const Real ph = kx[j] * dx + ky[j] * dy + kz[j] * dz;
            const Real sn = std::sin(ph);
            const Real cs = std::cos(ph);
            for (std::size_t m = 0; m < M; ++m) {
                re[m][0] += amp[m][j] * cs;
                im[m][0] -= amp[m][j] * sn;
            }
        }
    }
    for (std::size_t m = 0; m < M; ++m) {
        Real sr = 0, si = 0;
        for (std::size_t l = 0; l < kLanes; ++l) {
            sr += re[m][l];
            si += im[m][l];
        }
        out[m] = Complex(sr, si);
    }
}

Vec3c AngularSpectrumBeam::probe_field(const Grid& g, const Vec3& r) const {
    const Real* const amp[3] = {g.ex.data(), g.ey.data(), g.ez.data()};
    Complex out[3];
    sum_waves<3>(g, r, amp, out);
    return {out[0], out[1], out[2]};
}

Vec3c AngularSpectrumBeam::electric_field(const Vec3& r) const {
    return probe_field(grid_, r);
}

Vec3c AngularSpectrumBeam::magnetic_field(const Vec3& r) const {
    const Real* const amp[3] = {grid_.hx.data(), grid_.hy.data(), grid_.hz.data()};
    Complex out[3];
    sum_waves<3>(grid_, r, amp, out);
    return {out[0], out[1], out[2]};
}

std::pair<Vec3c, Vec3c> AngularSpectrumBeam::fields(const Vec3& r) const {
    const Real* const amp[6] = {grid_.ex.data(), grid_.ey.data(), grid_.ez.data(),
                                grid_.hx.data(), grid_.hy.data(), grid_.hz.data()};
    Complex out[6];
    sum_waves<6>(grid_, r, amp, out);
    return {Vec3c(out[0], out[1], out[2]), Vec3c(out[3], out[4], out[5])};
}

std::vector<Vec3> AngularSpectrumBeam::plane_wave_directions() const {
    std::vector<Vec3> d;
    d.reserve(static_cast<std::size_t>(grid_.n));
    for (std::size_t i = 0; i < static_cast<std::size_t>(grid_.n); ++i) {
        d.emplace_back(Vec3(grid_.kx[i], grid_.ky[i], grid_.kz[i]) / k_);
    }
    return d;
}

std::vector<Vec3> AngularSpectrumBeam::plane_wave_amplitudes() const {
    std::vector<Vec3> a;
    a.reserve(static_cast<std::size_t>(grid_.n));
    for (std::size_t i = 0; i < static_cast<std::size_t>(grid_.n); ++i) {
        a.emplace_back(grid_.ex[i], grid_.ey[i], grid_.ez[i]);
    }
    return a;
}

}  // namespace specklebem::excitation
