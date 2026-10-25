// Absorbed power of the flat box (WP-A1, follow-up of WP22c): one flat-patch case of
// benchmarks/fresnel_flat.cpp (same options, same solve) with diagnostics of the absorbed power
// from the surface currents (tests/support/ag_absorption_support.hpp):
//  * P_abs = 1/2 Re oint (n x M) . J* dS per part (top face, walls, bottom) with the exact
//    Dunavant degree 2 (as the WP22c study) and with degree 6 (pointwise check);
//  * P_ext - P_sca: extinction from the currents and the incident beam minus the scattered power
//    of both hemispheres (far field), an absorbed power that does not use n x M . J*;
//  * the analytic currents of the beam on the infinite interface (every plane wave reflected
//    with its Fresnel coefficients) and their RWG interpolant, compared with the simulated
//    currents on the centre (ellipse of semi-axes w0 / (2 cos theta), w0 / 2), the inner square
//    |x|, |y| <= L / 4 and the whole top face: absorbed power (simulated, reference,
//    interpolant, cross terms) and the correlation of J and M with the reference (amplitude,
//    phase).
// Output (stdout): the fresnel_flat summary, then DIAG lines and an ABS line; the ABS line is also
// appended to --summary FILE. Record: benchmarks/results/ag_absorption.md.
//
// Usage: specklebem_ag_absorption --material ag|si [--L 2e-6] [--w0 0.4e-6] [--theta 0]
//            [--pol p|s] [--allow-wide-beam] [--mesh-size 50e-9]
//            [--compression mlfmm|dense] [--digits|--d0 3]
//            [--tol 1e-5] [--max-iter 6000] [--restart 0] [--ff-dtheta 1]
//            [--formulation pmchwt|ictf|mctf] [--jacobi 0|1] [--summary FILE] [--force]
//            [--save-currents FILE] [--refine 1e-6,1e-7] [--row-check RADIUS]
//            [--box-depth D] [--hessenberg [--save-hessenberg FILE] [--no-ritz]]
//            [--perturb REL [--perturb-seed S]] [--spectrum [--spectrum-window rect|hann]]
//            [--retro] [--interior]
//        or  ... --load-currents FILE [--row-check RADIUS] [--profile BINS]
//            [--spectrum] [--spectrum-window rect|hann] [--strip-check HALF_WIDTH]
//            [--retro] [--interior] [--perturb REL [--perturb-seed S]] [--box-depth D]
// Options of the other mode (--save-currents / --refine / --hessenberg with --load-currents;
// --profile, --strip-check without it) are rejected. --perturb works in both modes (with
// --load-currents it assembles the operator and perturbs the loaded solution).
// --allow-wide-beam accepts w0 > L cos(theta) / 4 (Ag; / 6 for Si: check_beam_waist with
// beam_waist_fraction, ADR 0006 amendments 2026-10-11 and 2026-10-12b) with a warning, as
// benchmarks/fresnel_flat.cpp: the recorded 45 deg cases use w0 = L / 5 (the WP22c parameters, edge
// loss 0.06-0.1 %).
// --save-currents writes the solved [J; M] (raw complex<double>); --load-currents evaluates the
// diagnostics of saved currents of the same case without assembly and solve. --refine adds
// iterative-refinement steps x <- x + GMRES(Z, b - Z x) (left Jacobi) down to the listed true
// relative residuals and prints a STAGE line (R, A, phase of M against J) after each.
// --row-check compares the compressed operator with the exact Galerkin rows (op::assemble_sparse)
// of the central top-face basis functions on the solution and prints the exact residual there.
// --spectrum prints the spatial spectrum along x of the J and M errors on the strip |y| <= L / 8,
// raw (area-weighted DFT over the top-triangle centroids) and relative to the reference spectrum
// at the specular wavenumber +k0 sin(theta); --spectrum-window hann tapers the strip with
// cos^2(pi x / L) (default rect, the window of the recorded spectra).
// WP-A2 (origin of the near-null mode, record section "WP-A2"):
// --box-depth D sets the depth of the closing box [m] (default: default_box_depth, 2 um for Ag);
// a MODE line (both modes) gives the normalised +-k0 sin(theta) peaks of the current error
// X_sim - X_ref on the strip |y| <= L / 8 (rect and hann windows), relative to the reference
// spectrum at +k0 sin(theta).
// --hessenberg keeps the GMRES Hessenberg matrix (GmresParams::keep_hessenberg) and prints RITZ
// lines: the harmonic Ritz values closest to 0, the Ritz values and the smallest singular values
// of the (k + 1) x k Hessenberg of the Jacobi-preconditioned operator (tests/support/
// krylov_spectrum.hpp); --save-hessenberg writes it (raw complex<double>, column-major, after
// two int64 dimensions rows, cols); --no-ritz skips the in-process analysis (Eigen, serial:
// ~25 min for k = 2757; analyse the saved matrix with numpy instead).
// --perturb REL solves Z dx = db for a broadband random db (complex Gaussian, std::mt19937_64
// with --perturb-seed, default 20261011; per block scaled to REL |b_J| and REL |b_M|), the
// response of the solution to a right-hand-side perturbation b + db (linear: x + dx), and prints
// a PERTURB line (gains, normalised +-k0 sin(theta) peaks of dx) and a STAGE line of x + dx.
// --retro prints the far-field intensity in the plane of incidence (FF lines, scattering angle
// theta_s from -z towards +x: specular +theta, retro -theta) and a RETRO summary.
// --interior prints the extinction-theorem residual E_inc + E_1 (R1 representation of the
// currents) inside the box along lines in x, y and z (INTERIOR, INTSPEC, INTZ lines).
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/solver/gmres.hpp"
#include "specklebem/solver/preconditioner.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ag_absorption_support.hpp"
#include "fresnel_flat_support.hpp"
#include "krylov_spectrum.hpp"

namespace {

using namespace fresnel_flat;
namespace aa = ag_absorption;

struct Options {
    Case c;
    std::string summary;
    bool force = false;
    std::string save;  ///< --save-currents FILE: raw [J; M] (complex<double>) after the solve
    std::string load;  ///< --load-currents FILE: diagnostics of saved currents, no solve
    /// --refine T1,T2,...: iterative refinement to the true relative residuals T1, T2, ...
    std::vector<Real> refine;
    /// --row-check RADIUS: exact system rows of the top-face basis functions within RADIUS of
    /// the centre against the compressed operator (op::assemble_sparse) [m]
    Real row_check_radius = 0;
    /// --profile BINS (with --load-currents): metrics in BINS strips along x and along y
    Index profile_bins = 0;
    /// --spectrum (with --load-currents): spatial spectrum along x of the current error
    bool spectrum = false;
    /// --spectrum-window rect|hann (with --spectrum): window of the strip along x
    bool hann = false;
    bool window_given = false;
    /// --strip-check HALF_WIDTH (with --load-currents): compressed vs exact rows on the strip
    /// |y| <= HALF_WIDTH applied to the interpolated analytic currents [m]
    Real strip_half_width = 0;
    /// --hessenberg: keep the GMRES Hessenberg and print its Ritz analysis (RITZ lines)
    bool hessenberg = false;
    std::string save_hessenberg;  ///< --save-hessenberg FILE
    /// --no-ritz: skip the in-process Ritz analysis (Eigen, serial: ~25 min for k = 2757)
    bool no_ritz = false;
    /// --perturb REL: response dx = Z^{-1} db to a broadband random db, |db_X| = REL |b_X|
    Real perturb = 0;
    /// --perturb-seed S (only with --perturb; default 20261011)
    std::optional<std::uint64_t> perturb_seed;
    [[nodiscard]] std::uint64_t seed() const { return perturb_seed.value_or(20261011); }
    bool retro = false;     ///< --retro: far field in the plane of incidence
    bool interior = false;  ///< --interior: extinction-theorem residual inside the box
};

Options parse(int argc, char** argv) {
    Options o;
    o.c.L = 2e-6;
    o.c.waist = 0.4e-6;
    o.c.tolerance = 1e-5;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::invalid_argument("missing value after " + a);
            return argv[++i];
        };
        if (a == "--material") {
            o.c.material = value();
        } else if (a == "--L") {
            o.c.L = std::stod(value());
        } else if (a == "--w0") {
            o.c.waist = std::stod(value());
        } else if (a == "--theta") {
            o.c.theta_deg = std::stod(value());
        } else if (a == "--allow-wide-beam") {
            o.c.allow_wide_beam = true;
        } else if (a == "--pol") {
            const std::string p = value();
            if (p != "p" && p != "s")
                throw std::invalid_argument("--pol must be p or s");
            o.c.pol = p == "p" ? excitation::Polarization::P : excitation::Polarization::S;
        } else if (a == "--compression") {
            o.c.compression = value();
        } else if (a == "--digits" || a == "--d0") {
            o.c.digits = std::stod(value());
        } else if (a == "--tol") {
            o.c.tolerance = std::stod(value());
        } else if (a == "--max-iter") {
            o.c.max_iter = std::stoi(value());
        } else if (a == "--restart") {
            o.c.restart = std::stoi(value());
        } else if (a == "--mesh-size") {
            o.c.mesh_size = std::stod(value());
        } else if (a == "--ff-dtheta") {
            o.c.ff_dtheta_deg = std::stod(value());
        } else if (a == "--formulation") {
            o.c.formulation = formulation::parse_kind(value());
        } else if (a == "--jacobi") {
            o.c.jacobi = std::stoi(value()) != 0;
        } else if (a == "--summary") {
            o.summary = value();
        } else if (a == "--save-currents") {
            o.save = value();
        } else if (a == "--load-currents") {
            o.load = value();
        } else if (a == "--strip-check") {
            o.strip_half_width = std::stod(value());
        } else if (a == "--spectrum") {
            o.spectrum = true;
        } else if (a == "--spectrum-window") {
            const std::string w = value();
            if (w != "rect" && w != "hann")
                throw std::invalid_argument("--spectrum-window must be rect or hann");
            o.hann = w == "hann";
            o.window_given = true;
        } else if (a == "--profile") {
            o.profile_bins = static_cast<Index>(std::stoll(value()));
            if (o.profile_bins <= 0)
                throw std::invalid_argument("--profile needs a positive number of bins");
        } else if (a == "--row-check") {
            o.row_check_radius = std::stod(value());
        } else if (a == "--refine") {
            std::stringstream ss(value());
            std::string item;
            while (std::getline(ss, item, ',')) o.refine.push_back(std::stod(item));
        } else if (a == "--box-depth") {
            o.c.box_depth = std::stod(value());
        } else if (a == "--hessenberg") {
            o.hessenberg = true;
        } else if (a == "--save-hessenberg") {
            o.save_hessenberg = value();
        } else if (a == "--no-ritz") {
            o.no_ritz = true;
        } else if (a == "--perturb") {
            o.perturb = std::stod(value());
            if (!(o.perturb > 0.0))
                throw std::invalid_argument("--perturb needs a positive relative size");
        } else if (a == "--perturb-seed") {
            o.perturb_seed = static_cast<std::uint64_t>(std::stoull(value()));
        } else if (a == "--retro") {
            o.retro = true;
        } else if (a == "--interior") {
            o.interior = true;
        } else if (a == "--force") {
            o.force = true;
        } else {
            throw std::invalid_argument("unknown option " + a);
        }
    }
    object_material(o.c.material);  // validates the name
    // Options of the other mode are errors, not silently ignored.
    if (o.load.empty()) {
        if (o.profile_bins > 0 || o.strip_half_width > 0.0)
            throw std::invalid_argument("--profile and --strip-check need --load-currents");
    } else if (!o.save.empty() || !o.refine.empty() || o.hessenberg) {
        throw std::invalid_argument(
            "--save-currents, --refine and --hessenberg need a solve (no --load-currents)");
    }
    if (o.window_given && !o.spectrum)
        throw std::invalid_argument("--spectrum-window needs --spectrum");
    if ((!o.save_hessenberg.empty() || o.no_ritz) && !o.hessenberg)
        throw std::invalid_argument("--save-hessenberg and --no-ritz need --hessenberg");
    if (o.perturb_seed.has_value() && !(o.perturb > 0.0))
        throw std::invalid_argument("--perturb-seed needs --perturb");
    o.c.keep_hessenberg = o.hessenberg;
    return o;
}

void append_summary(const std::string& path, const std::string& line) {
    if (path.empty())
        return;
    std::ofstream f(path, std::ios::app | std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    f << line << '\n';
}

void save_currents(const std::string& path, const VectorXc& x) {
    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    f.write(reinterpret_cast<const char*>(x.data()),
            static_cast<std::streamsize>(static_cast<std::size_t>(x.size()) * sizeof(Complex)));
}

VectorXc load_currents(const std::string& path, Index n) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    VectorXc x(n);
    f.read(reinterpret_cast<char*>(x.data()),
           static_cast<std::streamsize>(static_cast<std::size_t>(n) * sizeof(Complex)));
    if (!f || f.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error(path + ": size does not match 2N = " + std::to_string(n));
    return x;
}

struct Diagnostics {
    aa::AbsorbedParts deg2, deg6;
    Real p_ext = 0;
    Real p_fwd_full = 0;
    Real a_beam = 0;   ///< 1 - R_beam - away share
    Real a_patch = 0;  ///< reference absorbed flux through the L x L patch / P_inc
    aa::TopMetrics centre, inner, top;
    std::size_t ref_waves = 0;
    Real seconds = 0;
};

std::string metrics_line(const char* name, const aa::TopMetrics& m, Real p_inc) {
    char buf[900];
    std::snprintf(
        buf, sizeof(buf),
        "DIAG %s | area %.4g um2 | A_sim %.6f | A_ref %.6f | A_int %.6f | sim/ref-1 %+.4f | "
        "int/ref-1 %+.4f | A(Msim,Jref)/ref-1 %+.4f | A(Mref,Jsim)/ref-1 %+.4f | |c_J| %.6f | "
        "arg c_J %+.3f mrad | |c_M| %.6f | arg c_M %+.3f mrad | err_J %.4e | err_M %.4e | "
        "err_J_int %.4e | err_M_int %.4e | |M|/|J| %.3f Ohm",
        name, m.area * 1e12, m.p_sim / p_inc, m.p_ref / p_inc, m.p_int / p_inc,
        m.p_sim / m.p_ref - 1.0, m.p_int / m.p_ref - 1.0, m.p_ms_jr / m.p_ref - 1.0,
        m.p_mr_js / m.p_ref - 1.0, std::abs(m.c_j), 1e3 * std::arg(m.c_j), std::abs(m.c_m),
        1e3 * std::arg(m.c_m), m.err_j, m.err_m, m.err_j_int, m.err_m_int, m.m_over_j);
    return buf;
}

/// kx grid of the printed spectra: -1.5 k0 .. +1.5 k0 in steps of 0.05 k0.
std::vector<Real> spectrum_grid(Real k) {
    std::vector<Real> kx;
    for (Index i = -30; i <= 30; ++i) kx.push_back(0.05 * static_cast<Real>(i) * k);
    return kx;
}

/// Spectrum lines of the currents of s along the strip |y| <= L / 8 (X_sim - X_ref with
/// `subtract`, X_sim without), normalised by the reference spectrum at +kx0.
void print_spectrum(const char* tag, const post::SurfaceSolution& s, const aa::InterfaceFields& f,
                    bool subtract, Real spacing, Real L, bool hann, Real kx0, Real k) {
    const aa::StripSpectrum ref0 =
        aa::strip_spectrum(s, f, subtract, spacing, L, L / 8.0, hann, {kx0});
    if (!(ref0.j_ref[0] > 0.0) || !(ref0.m_ref[0] > 0.0))
        throw std::runtime_error("spectrum: vanishing reference spectrum");
    std::printf(
        "%s window %s | strip |y| <= %.3g um, %zu centroids | reference at +k0 sin(theta) = "
        "%+.4f k0: |J_ref| %.4e | |M_ref| %.4e\n",
        tag, hann ? "hann" : "rect", L / 8.0 * 1e6, ref0.centroids, kx0 / k, ref0.j_ref[0],
        ref0.m_ref[0]);
    const aa::StripSpectrum sp =
        aa::strip_spectrum(s, f, subtract, spacing, L, L / 8.0, hann, spectrum_grid(k));
    for (std::size_t i = 0; i < sp.kx.size(); ++i) {
        std::printf(
            "%s kx/k %+.2f | |J_err| %.3e | |M_err| %.3e | |J_ref| %.3e | |M_ref| %.3e | "
            "J_err/J_ref(+k0 sin) %.3e | M_err/M_ref(+k0 sin) %.3e\n",
            tag, sp.kx[i] / k, sp.j[i], sp.m[i], sp.j_ref[i], sp.m_ref[i], sp.j[i] / ref0.j_ref[0],
            sp.m[i] / ref0.m_ref[0]);
    }
    std::fflush(stdout);
}

/// MODE line: the normalised +-kx0 peaks (rect and hann windows) of X_sim - X_ref (`subtract`)
/// or of X_sim.
std::string mode_line(const char* tag, const post::SurfaceSolution& s, const aa::InterfaceFields& f,
                      bool subtract, Real spacing, Real L, Real kx0) {
    const aa::ModePeaks r = aa::mode_peaks(s, f, subtract, spacing, L, L / 8.0, false, kx0);
    const aa::ModePeaks h = aa::mode_peaks(s, f, subtract, spacing, L, L / 8.0, true, kx0);
    char buf[600];
    std::snprintf(buf, sizeof(buf),
                  "MODE %s | rect: J(-k sin) %.3e M(-k sin) %.3e J(+k sin) %.3e M(+k sin) %.3e | "
                  "hann: J(-k sin) %.3e M(-k sin) %.3e J(+k sin) %.3e M(+k sin) %.3e",
                  tag, r.j_minus, r.m_minus, r.j_plus, r.m_plus, h.j_minus, h.m_minus, h.j_plus,
                  h.m_plus);
    return buf;
}

/// Far-field intensity |F|^2 in the plane of incidence on the reflection side, scattering angle
/// theta_s from -z towards +x (direction (sin theta_s, 0, -cos theta_s)): specular at +theta,
/// retro at -theta. FF lines every 0.25 deg and a RETRO summary.
void print_far_field(const post::SurfaceSolution& s, Real theta) {
    const int n = 717;  // -89.5 .. +89.5 deg
    Vertices dirs(n, 3);
    std::vector<Real> ts(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const Real t = (-89.5 + 0.25 * i) * kDeg;
        ts[static_cast<std::size_t>(i)] = t;
        dirs.row(i) << std::sin(t), 0.0, -std::cos(t);
    }
    Eigen::Matrix<Complex, Eigen::Dynamic, 3> F;
    post::far_field(s, dirs, F);
    VectorXr I(n);
    for (int i = 0; i < n; ++i) I(i) = F.row(i).squaredNorm();
    const Real imax = I.maxCoeff();
    // Specular and retro values at the grid angles nearest to +theta and -theta (the grid is
    // 0.25 deg; any theta is accepted, the angles used are printed).
    const auto nearest = [&](Real target) {
        int best = 0;
        for (int i = 1; i < n; ++i) {
            if (std::abs(ts[static_cast<std::size_t>(i)] - target) <
                std::abs(ts[static_cast<std::size_t>(best)] - target))
                best = i;
        }
        return best;
    };
    const int i_retro = nearest(-theta);
    const int i_spec = nearest(theta);
    const Real retro_exact = I(i_retro) / imax;
    const Real spec = I(i_spec) / imax;
    std::vector<Real> back;  // theta_s < -10 deg away from the retro direction (|.| > 5 deg)
    Real retro_peak = 0;
    for (int i = 0; i < n; ++i) {
        const Real t = ts[static_cast<std::size_t>(i)];
        const Real v = I(i) / imax;
        std::printf("FF theta_s %+.2f deg | |F|^2/max %.4e | %+.2f dB\n", t / kDeg, v,
                    10.0 * std::log10(std::max(v, 1e-300)));
        if (std::abs(t + theta) <= 2.0 * kDeg)
            retro_peak = std::max(retro_peak, v);
        if (t < -10.0 * kDeg && std::abs(t + theta) > 5.0 * kDeg)
            back.push_back(v);
    }
    std::sort(back.begin(), back.end());
    const Real median = back.empty() ? 0.0 : back[back.size() / 2];
    std::printf(
        "RETRO | specular (theta_s = %+.2f deg, nearest grid angle) |F|^2/max %.4e | retro "
        "(theta_s = %+.2f deg, nearest grid angle to %+.3f) %.4e | max within 2 deg of retro "
        "%.4e | median over theta_s < -10 deg (5 deg off retro) %.4e | retro / median %.3f\n",
        ts[static_cast<std::size_t>(i_spec)] / kDeg, spec,
        ts[static_cast<std::size_t>(i_retro)] / kDeg, -theta / kDeg, retro_exact, retro_peak,
        median, median > 0.0 ? retro_peak / median : 0.0);
    std::fflush(stdout);
}

/// Points r0 + i step dir, i = 0 .. n - 1.
Vertices line_points(const Vec3& r0, const Vec3& dir, Real step, Index n) {
    Vertices p(n, 3);
    for (Index i = 0; i < n; ++i) p.row(i) = (r0 + static_cast<Real>(i) * step * dir).transpose();
    return p;
}

/// Hann-tapered DFT along a line (coordinate u of each point) of a vector field: |sum|.
Real line_dft(const Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E, const std::vector<Real>& u,
              Real u0, Real u1, Real ku) {
    Vec3c acc = Vec3c::Zero();
    for (std::size_t i = 0; i < u.size(); ++i) {
        const Real w = std::pow(std::sin(constants::pi * (u[i] - u0) / (u1 - u0)), 2);
        acc += w * std::polar(1.0, ku * u[i]) * E.row(static_cast<Index>(i)).transpose().eval();
    }
    return acc.norm();
}

Real rms(const Eigen::Matrix<Complex, Eigen::Dynamic, 3>& E) {
    return std::sqrt(E.squaredNorm() / static_cast<Real>(std::max<Index>(E.rows(), 1)));
}

/// Extinction-theorem residual inside the box (INTERIOR / INTSPEC / INTZ lines): lines along x
/// (y = 0) at several depths, one along y (x = 0) and one along z (x = y = 0), 20 nm spacing,
/// >= 150 nm from the surface.
void print_interior(const post::SurfaceSolution& s, const excitation::Excitation& inc, Real L,
                    Real depth, Real kx0, Real k) {
    const auto t0 = std::chrono::steady_clock::now();
    const Real margin = 0.15e-6;
    const Real step = 20e-9;
    const auto n_lat = static_cast<Index>(std::floor((L - 2.0 * margin) / step)) + 1;
    const Real u0 = -0.5 * L + margin;
    const Real u1 = u0 + static_cast<Real>(n_lat - 1) * step;
    std::vector<Real> u(static_cast<std::size_t>(n_lat));
    for (Index i = 0; i < n_lat; ++i)
        u[static_cast<std::size_t>(i)] = u0 + static_cast<Real>(i) * step;
    const std::vector<Real> grid = spectrum_grid(k);
    const auto lateral = [&](const char* axis, Real z, bool along_x) {
        const Vec3 r0 = along_x ? Vec3(u0, 0.0, z) : Vec3(0.0, u0, z);
        const Vec3 dir = along_x ? Vec3::UnitX() : Vec3::UnitY();
        const aa::ExtinctionResidual er =
            aa::extinction_residual(s, inc, line_points(r0, dir, step, n_lat));
        Real kpeak = 0, speak = 0;
        for (const Real kk : grid) {
            const Real v = line_dft(er.residual, u, u0, u1, kk);
            if (v > speak) {
                speak = v;
                kpeak = kk;
            }
        }
        const Real inc_ref = line_dft(er.incident, u, u0, u1, along_x ? kx0 : 0.0);
        std::printf(
            "INTERIOR %s-line z %.3f um | %lld points | rms|res| / rms|inc| %.3e | max|res| / "
            "max|inc| %.3e | rms|inc| %.3e V/m | res(+k sin)/inc %.3e | res(-k sin)/inc %.3e | "
            "res(0)/inc %.3e | peak of |res(k)| at k/k0 %+.2f (%.3e of inc)\n",
            axis, z * 1e6, static_cast<long long>(n_lat), rms(er.residual) / rms(er.incident),
            er.residual.rowwise().norm().maxCoeff() / er.incident.rowwise().norm().maxCoeff(),
            rms(er.incident), line_dft(er.residual, u, u0, u1, kx0) / inc_ref,
            line_dft(er.residual, u, u0, u1, -kx0) / inc_ref,
            line_dft(er.residual, u, u0, u1, 0.0) / inc_ref, kpeak / k, speak / inc_ref);
        for (const Real kk : grid) {
            std::printf("INTSPEC %s-line z %.3f um | k/k0 %+.2f | |res| %.3e | |inc| %.3e\n", axis,
                        z * 1e6, kk / k, line_dft(er.residual, u, u0, u1, kk) / inc_ref,
                        line_dft(er.incident, u, u0, u1, kk) / inc_ref);
        }
        std::fflush(stdout);
    };
    for (const Real z : {0.15e-6, 0.3e-6, 0.6e-6, 1.0e-6, 1.5e-6}) {
        if (z <= depth - margin)
            lateral("x", z, true);
    }
    lateral("y", 0.3e-6, false);
    // Vertical line through the centre.
    const auto n_z = static_cast<Index>(std::floor((depth - 2.0 * margin) / step)) + 1;
    const aa::ExtinctionResidual ez = aa::extinction_residual(
        s, inc, line_points(Vec3(0.0, 0.0, margin), Vec3::UnitZ(), step, n_z));
    const Real inc_max = ez.incident.rowwise().norm().maxCoeff();
    for (Index i = 0; i < n_z; ++i) {
        std::printf(
            "INTZ z %.3f um | |res| %.3e | |res_x| %.3e | |res_y| %.3e | |res_z| %.3e | "
            "|inc| %.3e (relative to max |inc| on the line)\n",
            (margin + static_cast<Real>(i) * step) * 1e6, ez.residual.row(i).norm() / inc_max,
            std::abs(ez.residual(i, 0)) / inc_max, std::abs(ez.residual(i, 1)) / inc_max,
            std::abs(ez.residual(i, 2)) / inc_max, ez.incident.row(i).norm() / inc_max);
    }
    std::printf("INTERIOR time %.1f s\n",
                std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count());
    std::fflush(stdout);
}

/// RITZ lines: harmonic Ritz values closest to 0, Ritz values, singular values of the GMRES
/// Hessenberg (Jacobi-preconditioned operator).
void print_ritz(const MatrixXc& hbar) {
    namespace ks = krylov_spectrum;
    const auto t0 = std::chrono::steady_clock::now();
    const Index k = ks::hessenberg_dimension(hbar);
    const std::vector<Complex> harm = ks::sorted_by_modulus(ks::harmonic_ritz_values(hbar));
    const std::vector<Complex> ritz = ks::sorted_by_modulus(ks::ritz_values(hbar));
    const VectorXr sv = ks::singular_values(hbar);
    const auto count_below = [&](Real t) {
        return std::count_if(harm.begin(), harm.end(),
                             [t](const Complex& z) { return std::abs(z) < t; });
    };
    std::printf(
        "RITZ | k %lld | h_{k+1,k} %.3e | sigma_max %.4e | sigma_min %.4e | sigma_max/sigma_min "
        "%.3e | max |ritz| %.4e | harmonic |theta| < 0.01: %lld, < 0.03: %lld, < 0.1: %lld, < "
        "0.3: %lld | %.1f s\n",
        static_cast<long long>(k), std::abs(hbar(k, k - 1)), sv(0), sv(sv.size() - 1),
        sv(0) / sv(sv.size() - 1), std::abs(ritz.back()), static_cast<long long>(count_below(0.01)),
        static_cast<long long>(count_below(0.03)), static_cast<long long>(count_below(0.1)),
        static_cast<long long>(count_below(0.3)),
        std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count());
    const std::size_t n_show = std::min<std::size_t>(24, harm.size());
    for (std::size_t i = 0; i < n_show; ++i) {
        std::printf(
            "RITZ harmonic %2zu | theta %+.5e %+.5ej | |theta| %.5e | ritz %+.5e %+.5ej | "
            "|ritz| %.5e | sigma_(k-%zu) %.5e\n",
            i, harm[i].real(), harm[i].imag(), std::abs(harm[i]), ritz[i].real(), ritz[i].imag(),
            std::abs(ritz[i]), i, sv(sv.size() - 1 - static_cast<Index>(i)));
    }
    std::fflush(stdout);
}

void save_hessenberg(const std::string& path, const MatrixXc& h) {
    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    const std::int64_t dims[2] = {h.rows(), h.cols()};
    f.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    f.write(reinterpret_cast<const char*>(h.data()),
            static_cast<std::streamsize>(static_cast<std::size_t>(h.size()) * sizeof(Complex)));
}

/// Broadband random perturbation of b: complex Gaussian entries, per block (J rows, M rows)
/// scaled to rel |b_block|.
VectorXc random_perturbation(const VectorXc& b, Real rel, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<Real> g(0.0, 1.0);
    const Index n = b.size() / 2;
    VectorXc db(b.size());
    for (Index i = 0; i < b.size(); ++i) db(i) = Complex(g(rng), g(rng));
    for (const Index off : {Index{0}, n}) {
        auto seg = db.segment(off, n);
        seg *= rel * b.segment(off, n).norm() / seg.norm();
    }
    return db;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        const Case& c = o.c;
        std::printf("%s\n", label(c).c_str());
        std::printf("available memory at start: %.1f GB of %.1f GB\n",
                    system_memory::available_memory_bytes() / 1e9,
                    system_memory::physical_memory_bytes() / 1e9);
        std::fflush(stdout);
        const Geometry geo = make_geometry(c);
        const auto tb = std::chrono::steady_clock::now();
        const auto beam = make_beam(c, geo.r_max);
        const Real build_s =
            std::chrono::duration<Real>(std::chrono::steady_clock::now() - tb).count();
        const Estimate e = estimate(geo, c, beam);
        std::printf("SETUP | %s\n", summary(e).c_str());
        std::fflush(stdout);
        const Real need = e.peak_setup_bytes + e.krylov_max_bytes;
        const Real avail = system_memory::available_memory_bytes();
        if (!o.force && !(avail > need + 10e9)) {
            std::fprintf(stderr,
                         "ag_absorption: estimated peak %.1f GB + 10 GB margin exceeds the "
                         "available %.1f GB (--force to run)\n",
                         need / 1e9, avail / 1e9);
            return 2;
        }
        const Complex eps2 = object_material(c.material).eps_r;
        const Real th = c.theta_deg * kDeg;
        Diagnostics d;
        const auto diagnose = [&](const post::SurfaceSolution& s) {
            const auto t0 = std::chrono::steady_clock::now();
            d.deg2 = aa::absorbed_power_parts(s, geo.spacing, 2);
            d.deg6 = aa::absorbed_power_parts(s, geo.spacing, 6);
            d.p_ext = aa::extinction_power(s, *beam, 6);
            const Real eta1 = material::vacuum().wave_impedance(beam->omega()).real();
            const Vec3 fwd(std::sin(th), 0.0, std::cos(th));
            d.p_fwd_full = hemisphere_power(s, hemisphere_grid(+1.0, c.ff_dtheta_deg), c.ff_degree,
                                            eta1, fwd, 0.1)
                               .total;
            const aa::InterfaceFields f(*beam, 1.0, eps2);
            d.ref_waves = f.plane_waves();
            const BeamReflectance br = beam_reflectance(*beam, 1.0, eps2);
            d.a_beam = 1.0 - br.reflectance - br.away_share;
            d.a_patch = aa::reference_absorbed_flux(f, 0.5 * c.L, kLambda / 16.0) / beam->power();
            const VectorXc interp = aa::interpolate_reference(*s.problem->space, f, geo.spacing);
            const Real a2 = 0.25 * c.waist * c.waist;
            const Real ct = std::cos(th);
            d.centre = aa::top_metrics(
                s, f, interp, geo.spacing,
                [&](const Vec3& p) { return p.x() * p.x() * ct * ct + p.y() * p.y() <= a2; }, 4);
            d.inner = aa::top_metrics(
                s, f, interp, geo.spacing,
                [&](const Vec3& p) {
                    return std::abs(p.x()) <= 0.25 * c.L && std::abs(p.y()) <= 0.25 * c.L;
                },
                4);
            d.top = aa::top_metrics(s, f, interp, geo.spacing, [](const Vec3&) { return true; }, 4);
            d.seconds = std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
        };
        // Exact rows of the system at the patch centre against the compressed operator.
        const auto do_row_check = [&](const Simulation& sim, const VectorXc& x) {
            if (!(o.row_check_radius > 0.0))
                return;
            const std::vector<Index> rows =
                aa::central_basis(*sim.problem().space, geo.spacing, o.row_check_radius);
            const aa::RowCheck rc =
                aa::row_check(sim.problem(), *sim.system_operator(), sim.rhs(), x, rows);
            std::printf(
                "ROWCHECK | %lld basis functions within %.3g um | op error J rows %.3e | "
                "op error M rows %.3e | residual exact %.3e | residual compressed %.3e | "
                "%.1f s\n",
                static_cast<long long>(rc.rows), o.row_check_radius * 1e6, rc.op_error_j,
                rc.op_error_m, rc.residual_exact, rc.residual_compr, rc.seconds);
            std::fflush(stdout);
        };
        // One summary line of a solution (after the solve or a refinement step).
        const auto stage = [&](const std::string& name, const post::SurfaceSolution& s,
                               Real true_res) {
            const Real eta1 = material::vacuum().wave_impedance(beam->omega()).real();
            const Vec3 specular(std::sin(th), 0.0, -std::cos(th));
            const Real p_refl = hemisphere_power(s, hemisphere_grid(-1.0, c.ff_dtheta_deg),
                                                 c.ff_degree, eta1, specular, 0.1)
                                    .total;
            diagnose(s);
            const Real P = beam->power();
            const Real r_beam = beam_reflectance(*beam, 1.0, eps2).reflectance;
            std::printf(
                "STAGE %s | true_res %.3e | R_sim/R_beam-1 %+.5f %% | A_sim %.6f | A_ext "
                "%.6f | A_beam %.6f | walls %+.2e | centre %+.4f | inner %+.4f | top "
                "%+.4f | dphi(M-J) centre %+.3f inner %+.3f top %+.3f mrad\n",
                name.c_str(), true_res, 100.0 * (p_refl / P / r_beam - 1.0), d.deg2.total() / P,
                (d.p_ext - p_refl - d.p_fwd_full) / P, d.a_beam, d.deg2.walls / P,
                d.centre.p_sim / d.centre.p_ref - 1.0, d.inner.p_sim / d.inner.p_ref - 1.0,
                d.top.p_sim / d.top.p_ref - 1.0,
                1e3 * (std::arg(d.centre.c_m) - std::arg(d.centre.c_j)),
                1e3 * (std::arg(d.inner.c_m) - std::arg(d.inner.c_j)),
                1e3 * (std::arg(d.top.c_m) - std::arg(d.top.c_j)));
            std::fflush(stdout);
        };
        // --perturb: response dx = Z^{-1} db to a broadband random db (PERTURB, PSPEC, STAGE and
        // MODE lines); needs the assembled operator of sim and its solution s.
        const auto do_perturb = [&](const Simulation& sim, const post::SurfaceSolution& s,
                                    const aa::InterfaceFields& f) {
            const Real k0 = 2.0 * constants::pi / kLambda;
            const Real kx0 = k0 * std::sin(th);
            // Response to a broadband right-hand-side perturbation: dx = Z^{-1} db (the
            // change of the solution of Z x = b + db), same operator, left Jacobi and
            // tolerance as the solve.
            const op::LinearOperator& Z = *sim.system_operator();
            const VectorXc& b = sim.rhs();
            const VectorXc db = random_perturbation(b, o.perturb, o.seed());
            const solver::DiagonalPreconditioner jac(op::assemble_diagonal(sim.problem()));
            solver::GmresParams gp;
            gp.tolerance = c.tolerance;
            gp.max_iter = c.max_iter;
            gp.verbose = true;
            const solver::GmresResult g = solver::gmres(Z, db, jac, gp);
            const VectorXc& x = s.currents;
            const Index N = x.size() / 2;
            const post::SurfaceSolution ds{&sim.problem(), g.x};
            const Real rel_b = db.norm() / b.norm();
            const Real rel_x = g.x.norm() / x.norm();
            const aa::ModePeaks pr =
                aa::mode_peaks(ds, f, false, geo.spacing, c.L, c.L / 8.0, false, kx0);
            const aa::ModePeaks ph =
                aa::mode_peaks(ds, f, false, geo.spacing, c.L, c.L / 8.0, true, kx0);
            std::printf(
                "PERTURB | seed %llu | |db|/|b| %.3e (J rows %.3e, M rows %.3e) | it %d "
                "| converged %d | true residual of dx %.3e | |dx|/|x| %.3e (J %.3e, M "
                "%.3e) | gain |dx|/|x| / (|db|/|b|) %.2f | dx peaks / X_ref(+k sin), "
                "rect: J- %.3e M- %.3e J+ %.3e M+ %.3e | hann: J- %.3e M- %.3e J+ %.3e "
                "M+ %.3e | modal gain (hann, M-) %.2f, (hann, M+) %.2f\n",
                static_cast<unsigned long long>(o.seed()), rel_b,
                db.head(N).norm() / b.head(N).norm(), db.tail(N).norm() / b.tail(N).norm(),
                g.iterations, g.converged ? 1 : 0, g.true_relative_residual, rel_x,
                g.x.head(N).norm() / x.head(N).norm(), g.x.tail(N).norm() / x.tail(N).norm(),
                rel_x / rel_b, pr.j_minus, pr.m_minus, pr.j_plus, pr.m_plus, ph.j_minus, ph.m_minus,
                ph.j_plus, ph.m_plus, ph.m_minus / rel_b, ph.m_plus / rel_b);
            std::fflush(stdout);
            print_spectrum("PSPEC", ds, f, false, geo.spacing, c.L, true, kx0, k0);
            const VectorXc xp = x + g.x;
            VectorXc zx(xp.size());
            Z.apply(xp, zx);
            stage("perturbed x + dx", post::SurfaceSolution{&sim.problem(), xp},
                  (b - zx).norm() / b.norm());
            std::printf("%s\n",
                        mode_line("perturbed x + dx", post::SurfaceSolution{&sim.problem(), xp}, f,
                                  true, geo.spacing, c.L, kx0)
                            .c_str());
        };
        if (!o.load.empty()) {
            // Diagnostics of saved currents (no solve; assembly only for --row-check).
            Simulation sim(geo.mesh, beam, make_config(c));
            const post::SurfaceSolution s{&sim.problem(),
                                          load_currents(o.load, sim.num_unknowns())};
            const Reference ref = reference(c, *beam, build_s);
            const Real eta1 = material::vacuum().wave_impedance(beam->omega()).real();
            const Vec3 specular(std::sin(th), 0.0, -std::cos(th));
            const Real p_refl = hemisphere_power(s, hemisphere_grid(-1.0, c.ff_dtheta_deg),
                                                 c.ff_degree, eta1, specular, 0.1)
                                    .total;
            diagnose(s);
            const Real P = ref.power;
            const Real p_sca = p_refl + d.p_fwd_full;
            std::printf(
                "LOADED %s | R_sim %.7f | R_beam %.7f | A_sim %.6f | A_ext %.6f | A_beam "
                "%.6f | A_patch %.6f | walls %.3e | bottom %.3e\n",
                o.load.c_str(), p_refl / P, ref.beam.reflectance, d.deg2.total() / P,
                (d.p_ext - p_sca) / P, d.a_beam, d.a_patch, d.deg2.walls / P, d.deg2.bottom / P);
            const std::string mc = metrics_line("centre", d.centre, P);
            const std::string mi = metrics_line("inner", d.inner, P);
            const std::string mt = metrics_line("top", d.top, P);
            std::printf("%s\n%s\n%s\n", mc.c_str(), mi.c_str(), mt.c_str());
            if (o.profile_bins > 0) {
                const aa::InterfaceFields f(*beam, 1.0, eps2);
                const VectorXc interp =
                    aa::interpolate_reference(*s.problem->space, f, geo.spacing);
                for (const int axis : {0, 1}) {
                    for (const aa::ProfileBin& b : aa::top_profile(
                             s, f, interp, geo.spacing, c.L, o.profile_bins, axis, c.L / 8.0)) {
                        std::printf(
                            "PROFILE %s %+.3f um | A_ref/P %.3e | sim/ref-1 %+.4f | "
                            "int/ref-1 %+.4f | |c_J| %.5f | |c_M| %.5f | dphi(M-J) "
                            "%+.3f mrad | err_J %.3e | err_M %.3e\n",
                            axis == 0 ? "x" : "y", b.centre * 1e6, b.m.p_ref / ref.power,
                            b.m.p_sim / b.m.p_ref - 1.0, b.m.p_int / b.m.p_ref - 1.0,
                            std::abs(b.m.c_j), std::abs(b.m.c_m),
                            1e3 * (std::arg(b.m.c_m) - std::arg(b.m.c_j)), b.m.err_j, b.m.err_m);
                    }
                }
                std::fflush(stdout);
            }
            const Real k0 = 2.0 * constants::pi / kLambda;
            const Real kx0 = k0 * std::sin(th);
            {
                const aa::InterfaceFields f(*beam, 1.0, eps2);
                std::printf("%s\n", mode_line("error", s, f, true, geo.spacing, c.L, kx0).c_str());
                if (o.spectrum) {
                    // Spatial spectrum along x of the current error X_sim - X_ref on the strip
                    // |y| <= L / 8 (top-triangle centroids, area weights, optional Hann taper
                    // along x): which tangential wavenumber carries the error. The raw values
                    // are area-weighted sums [A m or V m]; the relative columns divide by the
                    // reference spectrum |X_ref| at the specular wavenumber +k0 sin(theta) (the
                    // reference currents vary as exp(-j kx x), so the specular peak sits at
                    // +k0 sin(theta)).
                    print_spectrum("SPECTRUM", s, f, true, geo.spacing, c.L, o.hann, kx0, k0);
                }
            }
            if (o.retro)
                print_far_field(s, th);
            if (o.interior)
                print_interior(s, *beam, c.L, geo.box.box_depth, kx0, k0);
            // The compressed operator, built once for --strip-check and --row-check.
            if (o.strip_half_width > 0.0 || o.row_check_radius > 0.0)
                sim.assemble();
            if (o.strip_half_width > 0.0) {
                // Compressed vs exact rows along the strip |y| <= half width, applied to the RWG
                // interpolant of the analytic currents (tangential wavenumber +k sin(theta) only):
                // spatial spectrum along x of the operator error on the J and M rows.
                const basis::RwgSpace& space = *sim.problem().space;
                const geometry::TriangleMesh& mesh = space.mesh();
                const Index N = space.size();
                const aa::InterfaceFields f(*beam, 1.0, eps2);
                const VectorXc xpw = aa::interpolate_reference(space, f, geo.spacing);
                std::vector<Index> rows;
                std::vector<Real> xm;
                for (Index n = 0; n < N; ++n) {
                    if (aa::part_of(mesh, space.plus_triangle(n), geo.spacing) != aa::Part::Top ||
                        aa::part_of(mesh, space.minus_triangle(n), geo.spacing) != aa::Part::Top)
                        continue;
                    const Vec3 mid = 0.5 * (mesh.vertices().row(mesh.edges()(n, 0)).transpose() +
                                            mesh.vertices().row(mesh.edges()(n, 1)).transpose());
                    if (std::abs(mid.y()) <= o.strip_half_width) {
                        rows.push_back(n);
                        xm.push_back(mid.x());
                    }
                }
                op::BasisPattern pat;
                pat.group_of_row.assign(static_cast<std::size_t>(N), 0);
                pat.col_ptr = {0, 0, N};
                pat.cols.resize(static_cast<std::size_t>(N));
                for (Index n = 0; n < N; ++n) pat.cols[static_cast<std::size_t>(n)] = n;
                for (const Index m : rows) pat.group_of_row[static_cast<std::size_t>(m)] = 1;
                const auto Zex = op::assemble_sparse(sim.problem(), pat);
                VectorXc ye(2 * N), yc(2 * N);
                Zex->apply(xpw, ye);
                sim.system_operator()->apply(xpw, yc);
                const Real k = 2.0 * constants::pi / kLambda;
                Real nd = 0, ne = 0;
                for (const Index m : rows) {
                    for (const Index r : {m, N + m}) {
                        nd += std::norm(yc(r) - ye(r));
                        ne += std::norm(ye(r));
                    }
                }
                std::printf(
                    "STRIP | %zu basis functions with |y| <= %.3g um | relative operator "
                    "error on the interpolant %.3e\n",
                    rows.size(), o.strip_half_width * 1e6, std::sqrt(nd / ne));
                for (Index i = -30; i <= 30; ++i) {
                    const Real kx = 0.05 * static_cast<Real>(i) * k;
                    Complex dj = 0, dm = 0, ej = 0, em = 0;
                    for (std::size_t q = 0; q < rows.size(); ++q) {
                        const Complex ph = std::polar(1.0, kx * xm[q]);
                        const Index m = rows[q];
                        dj += ph * (yc(m) - ye(m));
                        dm += ph * (yc(N + m) - ye(N + m));
                        ej += ph * ye(m);
                        em += ph * ye(N + m);
                    }
                    std::printf(
                        "STRIPSPEC kx/k %+.2f | J rows: |dZx| %.3e |Zx| %.3e | M rows: "
                        "|dZx| %.3e |Zx| %.3e\n",
                        kx / k, std::abs(dj), std::abs(ej), std::abs(dm), std::abs(em));
                }
                std::fflush(stdout);
            }
            do_row_check(sim, s.currents);  // no-op without --row-check
            if (o.perturb > 0.0) {
                sim.assemble();  // idempotent
                do_perturb(sim, s, aa::InterfaceFields(*beam, 1.0, eps2));
            }
            return 0;
        }
        const Result r = run(c, geo, beam, build_s, [&](const Simulation& sim) {
            if (!o.save.empty())
                save_currents(o.save, sim.solution().currents);
            do_row_check(sim, sim.solution().currents);
            if (!o.refine.empty()) {
                // Iterative refinement: x <- x + GMRES(Z, b - Z x) to the true residuals listed.
                const op::LinearOperator& Z = *sim.system_operator();
                const VectorXc& b = sim.rhs();
                const solver::DiagonalPreconditioner jac(op::assemble_diagonal(sim.problem()));
                VectorXc x = sim.solution().currents;
                VectorXc zx(x.size());
                Z.apply(x, zx);
                stage("solve", sim.solution(), (b - zx).norm() / b.norm());
                for (const Real target : o.refine) {
                    const VectorXc res = b - zx;
                    solver::GmresParams gp;
                    gp.tolerance = target * b.norm() / res.norm();
                    gp.max_iter = c.max_iter;
                    gp.verbose = true;
                    const solver::GmresResult g = solver::gmres(Z, res, jac, gp);
                    x += g.x;
                    Z.apply(x, zx);
                    const Real tr = (b - zx).norm() / b.norm();
                    char name[64];
                    std::snprintf(name, sizeof(name), "refine %.0e (%d it)", target, g.iterations);
                    stage(name, post::SurfaceSolution{&sim.problem(), x}, tr);
                }
            }
            {
                // WP-A2 diagnostics of the solution (before the final diagnose(): stage() of the
                // perturbed solution overwrites d).
                const Real k0 = 2.0 * constants::pi / kLambda;
                const Real kx0 = k0 * std::sin(th);
                const aa::InterfaceFields f(*beam, 1.0, eps2);
                const post::SurfaceSolution& s = sim.solution();
                const std::string mode = mode_line("error", s, f, true, geo.spacing, c.L, kx0);
                std::printf("%s\n", mode.c_str());
                if (o.spectrum)
                    print_spectrum("SPECTRUM", s, f, true, geo.spacing, c.L, o.hann, kx0, k0);
                if (o.retro)
                    print_far_field(s, th);
                if (o.interior)
                    print_interior(s, *beam, c.L, geo.box.box_depth, kx0, k0);
                if (o.perturb > 0.0)
                    do_perturb(sim, s, f);
            }
            diagnose(sim.solution());
        });
        if (o.hessenberg) {
            if (!o.save_hessenberg.empty())
                save_hessenberg(o.save_hessenberg, r.hessenberg);
            if (!o.no_ritz)
                print_ritz(r.hessenberg);
        }
        const Real P = r.ref.power;
        std::printf("%s\n\n%s\n\n", summary(r).c_str(), r.report.c_str());
        std::printf("DIAG parts (degree 2) | top %.6f | walls %.3e | bottom %.3e | total %.6f\n",
                    d.deg2.top / P, d.deg2.walls / P, d.deg2.bottom / P, d.deg2.total() / P);
        std::printf("DIAG parts (degree 6) | top %.6f | walls %.3e | bottom %.3e | total %.6f\n",
                    d.deg6.top / P, d.deg6.walls / P, d.deg6.bottom / P, d.deg6.total() / P);
        const Real p_sca = r.p_refl + d.p_fwd_full;
        std::printf(
            "DIAG extinction | P_ext/P_inc %.7f | P_refl/P_inc %.7f | P_fwd/P_inc %.7f | "
            "(P_ext - P_sca)/P_inc %.6f\n",
            d.p_ext / P, r.p_refl / P, d.p_fwd_full / P, (d.p_ext - p_sca) / P);
        std::printf(
            "DIAG reference | %zu plane waves | A_beam = 1 - R_beam - away %.6f | "
            "A_ref over the L x L patch %.6f | Fresnel 1 - |r|^2 %.6f\n",
            d.ref_waves, d.a_beam, d.a_patch, 1.0 - r.ref.beam.fresnel_central);
        const std::string mc = metrics_line("centre", d.centre, P);
        const std::string mi = metrics_line("inner", d.inner, P);
        const std::string mt = metrics_line("top", d.top, P);
        std::printf("%s\n%s\n%s\nDIAG time %.1f s\n", mc.c_str(), mi.c_str(), mt.c_str(),
                    d.seconds);
        char buf[1200];
        std::snprintf(
            buf, sizeof(buf),
            "ABS | %s | L %.4g | w0 %.4g | theta %.4g | pol %s | h %.4g | %s%s | %s | tol %.1e "
            "| 2N %lld | it %d | R_sim %.7f | R_beam %.7f | err_R %+.4f %% | A_sim %.6f | A_beam "
            "%.6f | A_sim/A_beam-1 %+.4f | A_ext %.6f | A_top %.6f | A_walls %.2e | A_bottom "
            "%.2e | centre sim/ref-1 %+.4f int/ref-1 %+.4f dphi(M-J) %+.3f mrad | inner sim/ref-1 "
            "%+.4f | 1-R-A %+.3e | solve %.0f s",
            c.material.c_str(), c.L, c.waist, c.theta_deg,
            c.pol == excitation::Polarization::P ? "p" : "s", c.mesh_size, r.formulation.c_str(),
            r.preconditioner == "diagonal" ? "+J" : (r.preconditioner == "none" ? "" : "+B"),
            c.compression.c_str(), c.tolerance, static_cast<long long>(r.est.unknowns),
            r.iterations, r.r_sim(), r.ref.beam.reflectance, 100.0 * r.error_beam(),
            d.deg2.total() / P, d.a_beam, d.deg2.total() / P / d.a_beam - 1.0,
            (d.p_ext - p_sca) / P, d.deg2.top / P, d.deg2.walls / P, d.deg2.bottom / P,
            d.centre.p_sim / d.centre.p_ref - 1.0, d.centre.p_int / d.centre.p_ref - 1.0,
            1e3 * (std::arg(d.centre.c_m) - std::arg(d.centre.c_j)),
            d.inner.p_sim / d.inner.p_ref - 1.0, 1.0 - r.r_sim() - d.deg2.total() / P, r.solve_s);
        std::printf("%s\n", buf);
        std::fflush(stdout);
        append_summary(o.summary, buf);
        return r.converged ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "ag_absorption: %s\n", ex.what());
        return 1;
    }
}
