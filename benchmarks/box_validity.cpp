// Validity of the graded closing box under beam illumination (WP-V1, ADR 0006): far field in the
// reflection hemisphere of a Gaussian rough surface (Si or Ag at 500 nm) for one configuration of
// the closing box, dense operator, GMRES (ICTF, no preconditioner, tight tolerance), so that runs
// differing in a single parameter (box depth, coarse spacing, fine band, patch size L at fixed
// waist) can be compared.
//
// One process = one configuration (Z of up to ~60 GB is assembled once, solved once). The far
// field |F|^2 is written as .npy (io::open_npy_directory) on fixed grids:
//   * reflection hemisphere k_z < 0: theta_i = (i + 1/2) deg, i = 0..89 (from -z), phi_j = 3 j
//     deg, j = 0..119: k_hat = (sin t cos p, sin t sin p, -cos t), "refl_intensity" (90 x 120);
//   * forward hemisphere (same grid, +cos t), "fwd_intensity";
//   * cuts in the xz- and yz-planes, theta = -89.5 .. 89.5 deg in 0.5 deg steps (from -z):
//     k_hat = (sin t, 0, -cos t) and (0, sin t, -cos t), "cut_xz_intensity", "cut_yz_intensity"
//     (and the complex amplitudes "cut_xz_F", "cut_yz_F").
// `--reference DIR` (repeatable) compares the run with earlier output directories; `--compare A
// B ...` compares stored directories only (B, C, ... against A), without a solve.
//
// Metrics of a comparison (test T against reference R, intensities I = |F|^2):
//   l2_refl   = sqrt(sum w (I_T - I_R)^2 / sum w I_R^2) over the reflection hemisphere,
//               w = sin(theta) (solid-angle weight of the grid),
//   l2_off30  = the same restricted to theta >= 30 deg (off-specular part),
//   l2_fwd    = the same over the forward hemisphere (shadow side; diagnostic only),
//   dP_refl   = relative change of the power scattered into the reflection hemisphere,
//   eps_xz/yz = docs/05 eps_rr = sqrt(mean((I_T - I_R)^2)) / max I_R along the cuts.
//
// Surface: sigma = 50 nm, Lc = 500 nm, mesh 50 nm, fixed seed. The height map is generated on an
// L_gen x L_gen patch (`--L-gen`, default L) and cropped centrally to L, so runs with different L
// share the surface on the common area (the edge-effect check L x 1.2 compares the same surface).
//
// Beams: "spectrum" (default) is a rigorous angular-spectrum Gaussian beam built here (exact
// Maxwell solution in vacuum: a quadrature over propagating plane waves with the Gaussian
// spectrum exp(-k_t^2 w0^2 / 4), E-vector x_hat - (k_hat . x_hat) k_hat, normalised to E_x = 1 V/m
// at the focus); "paraxial" is excitation::GaussianBeam (first-order paraxial, Maxwell residual
// O((lambda / (pi w0))^2)). Both: normal incidence, travelling +z, focus at the origin, E along x.
//
// Unconverged runs (GMRES stopped at --max-iter) are neither written nor compared: the process
// prints the RESULT line ("converged no") and exits with status 1. `--allow-unconverged` writes
// and compares them anyway (attribute "converged": "no", COMPARE lines marked UNCONVERGED). Runs
// with `--eps-imag` are marked DIAGNOSTIC (SETUP, RESULT, COMPARE, attribute "diagnostic"), since
// they do not use the tabulated material.
//
// Usage: specklebem_box_validity --material ag|si [--L 2e-6] [--L-gen 2.4e-6] [--seed 1]
//            [--waist W | --waist-factor 3] [--depth-factor 1 | --depth D]
//            [--box-mesh-size 400e-9|auto|uniform] [--fine-band] [--beam spectrum|paraxial]
//            [--eps-imag X (diagnostic: Im eps_r of the object, < 0)]
//            [--tol 1e-6] [--max-iter 6000] [--max-gb 60] [--ff-degree 10] [--tag NAME]
//            [--out DIR] [--summary FILE] [--reference DIR]... [--mesh-only]
//            [--allow-unconverged]
//            [--compression dense|mlfmm] [--d0 3] [--leaf-quantile 1] [--jacobi auto|0|1]
//            [--estimate-only]
//            [--force] [--parts 0|1] [--history FILE]
//            [--sigma 50e-9] [--save-currents] [--mechanism] [--ritz K1,K2,...]
//        specklebem_box_validity <same configuration options> --load-currents DIR
//            [--mechanism] [--diff DIR_B] [--summary FILE]
//        specklebem_box_validity --compare DIR_A DIR_B [DIR_C ...] [--allow-unconverged]
//        specklebem_box_validity --depth-fit DIR_A DIR_B DIR_C [...]
//
// MLFMM (WP-V2, `--compression mlfmm`): Simulation with compression "mlfmm" (accuracy digits
// `--d0`, global leaf rule, automatic exact-part budget), ICTF, Jacobi per formulation::recommend
// unless `--jacobi` (Ag: left Jacobi, so the GMRES tolerance applies to the preconditioned
// residual; the true residual is printed). The memory guard of the MLFMM scaling study
// (tests/support/mlfmm_scaling_support.hpp: near + far + exact part setup peak model) plus the
// full-GMRES Krylov basis at --max-iter must stay below --max-gb, and the setup peak plus 10 GB
// below the available memory (else refused unless --force); `--estimate-only` prints the estimate
// and stops. The dense path (default) is unchanged from WP-V1. `--parts 0` skips the PARTS
// diagnostic, `--history FILE` writes the GMRES residual history (one value per line).
//
// WP-V3 (Ag box mechanism, record benchmarks/results/ag_box_mechanism.md; helpers in
// tests/support/ag_box_mechanism_support.hpp). Every stored run also holds the complex
// reflection-hemisphere far field "refl_F" (rows as refl_intensity, columns x, y, z) and the
// attribute "rms_roughness". `--sigma` sets the rms roughness (0: flat patch). `--save-currents`
// stores [J; M] as currents.npy in --out. `--mechanism` prints, after the solve (or for stored
// currents), the radially binned 2D spatial spectra of J and M on the top face, the four walls
// (combined) and the bottom plate (SPECSUM / SPEC lines: power fractions per 0.025 k0 bin, the
// fraction above k0 and near Re k_spp), and the extinction-theorem residual E_inc + E_1 inside the
// box on the WP-A2 probe lines (INTERIOR / INTZ lines), with a MECH summary line. `--ritz K1,...`
// keeps the GMRES Hessenberg (stored as "hessenberg" in --out) and prints the harmonic Ritz values
// of its leading (K + 1) x K blocks (RITZ / RITZV lines). `--load-currents DIR` runs the
// diagnostics on a stored run without assembly or solve: the configuration options must describe
// the stored run, and its attributes (material, beam, eps_r, rms_roughness, seed, L, map_L,
// waist_radius, box_depth, box_mesh_size, fine_band, wavelength, diagnostic) are compared with
// the options (mismatch: error). `--diff DIR_B` adds the spectrum of the top-face current
// difference against another stored run on their common top-face triangles (DIFF lines; e.g.
// depth 4 vs 2 um, or L 4.8 vs 4 um); DIR_B must match the same attributes except L and
// box_depth. `--depth-fit` fits F_i = F_inf + A exp(-j beta D_i) to the refl_F of stored runs at
// several depths D_i (DEPTHFIT / DFCURVE lines); the runs must have the same configuration
// attributes apart from box_depth (ag_box_mechanism::configuration_keys). `--ritz K` needs
// K <= --max-iter; a solve that converges in fewer iterations uses its full Hessenberg (warning).
// Every face diagnostic checks that the normal-based face classification finds the generator's
// 2 (n_x - 1)(n_y - 1) top-face triangles (60-degree slope limit, ag_box_mechanism::classify).
// The SETUP line includes sigma (the WP-V3 logs committed before the review round predate it).
#include "specklebem/basis/rwg.hpp"
#include "specklebem/core/logging.hpp"
#include "specklebem/core/path.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/io/result_writer.hpp"
#include "specklebem/kernels/fast_math.hpp"
#include "specklebem/kernels/quadrature.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/postprocessing/fields.hpp"
#include "specklebem/simulation.hpp"
#include "specklebem/solver/gmres.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ag_box_mechanism_support.hpp"
#include "box_validity_support.hpp"
#include "krylov_spectrum.hpp"
#include "mlfmm_scaling_support.hpp"
#include "system_memory.hpp"

using namespace specklebem;
using namespace box_validity;

namespace {

constexpr Real kLambda = 500e-9;
constexpr Real kCorrelationLength = 500e-9;
constexpr Real kMeshSize = 50e-9;
// Far-field grids, metrics and options: tests/support/box_validity_support.hpp.

material::Material material_by_name(const std::string& name) {
    if (name == "si")
        return material::silicon_500nm();
    if (name == "ag")
        return material::silver_500nm();
    throw std::invalid_argument("unknown material " + name + " (si or ag)");
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// ---------------------------------------------------------------------------------------------
// Rigorous Gaussian beam: angular spectrum of propagating plane waves.

/// E(r) = sum_i c_i e_i exp(-j k k_hat_i . r), H = sum_i c_i (k_hat_i x e_i) / eta0 exp(...),
/// k_hat_i = (sin a cos p, sin a sin p, cos a) on a Gauss-Legendre (a in [0, a_max]) x trapezoid
/// (p) grid, c_i = weight * exp(-(k sin a)^2 w0^2 / 4) k^2 sin a cos a (polar measure of the
/// transverse wavevector), e_i = x_hat - (k_hat_i . x_hat) k_hat_i, scaled so that E_x(0) = 1.
/// Every plane wave solves Maxwell's equations in vacuum, so the beam does exactly.
class SpectrumGaussianBeam final : public excitation::Excitation {
public:
    SpectrumGaussianBeam(Real wavelength, Real w0)
        : Excitation(2.0 * constants::pi * constants::c0 / wavelength, material::vacuum()),
          k_(2.0 * constants::pi / wavelength),
          w0_(w0) {
        // Spectrum below 1e-17 of its peak beyond a_max.
        const Real s = std::sqrt(4.0 * 39.0) / (k_ * w0_);
        a_max_ = s >= 1.0 ? 0.5 * constants::pi : std::asin(s);
    }

    /// Builds the plane-wave grid with n_alpha GL nodes and n_phi azimuths.
    void build(int n_alpha, int n_phi) {
        const kernels::LineRule gl = kernels::gauss_legendre(n_alpha);
        const std::size_t n = static_cast<std::size_t>(n_alpha) * static_cast<std::size_t>(n_phi);
        for (auto* v : {&kx_, &ky_, &kz_, &ex_, &ey_, &ez_, &hx_, &hy_, &hz_}) v->assign(n, 0.0);
        const Real dphi = 2.0 * constants::pi / n_phi;
        Real norm = 0.0;
        std::size_t idx = 0;
        for (int ia = 0; ia < n_alpha; ++ia) {
            const Real a = 0.5 * a_max_ * (gl.nodes[static_cast<std::size_t>(ia)] + 1.0);
            const Real wa = 0.5 * a_max_ * gl.weights[static_cast<std::size_t>(ia)];
            const Real kt = k_ * std::sin(a);
            const Real c = wa * dphi * std::exp(-0.25 * kt * kt * w0_ * w0_) * k_ * k_ *
                           std::sin(a) * std::cos(a);
            for (int ip = 0; ip < n_phi; ++ip, ++idx) {
                const Real p = dphi * ip;
                const Vec3 kh(std::sin(a) * std::cos(p), std::sin(a) * std::sin(p), std::cos(a));
                const Vec3 e = Vec3::UnitX() - kh.x() * kh;
                const Vec3 h = kh.cross(e) / constants::eta0;
                kx_[idx] = k_ * kh.x();
                ky_[idx] = k_ * kh.y();
                kz_[idx] = k_ * kh.z();
                ex_[idx] = c * e.x();
                ey_[idx] = c * e.y();
                ez_[idx] = c * e.z();
                hx_[idx] = c * h.x();
                hy_[idx] = c * h.y();
                hz_[idx] = c * h.z();
                norm += c * e.x();
            }
        }
        for (auto* v : {&ex_, &ey_, &ez_, &hx_, &hy_, &hz_})
            for (Real& x : *v) x /= norm;
        n_alpha_ = n_alpha;
        n_phi_ = n_phi;
        // Process-wide unique key: a beam rebuilt (or a new beam at the same address) never hits
        // a stale thread_local cache entry.
        static std::atomic<std::uint64_t> next_generation{0};
        generation_ = ++next_generation;
    }

    /// Chooses the grid for points within |r| <= r_max (rho <= rho_max from the axis): estimate
    /// from the phase variation, then refine until a 1.5x finer grid changes E by < 1e-10 of
    /// max |E| on `probe` points. Returns the achieved check value.
    Real auto_build(const Vertices& probe, Real r_max, Real rho_max) {
        int na = static_cast<int>(std::ceil(0.5 * k_ * r_max * a_max_)) + 16;
        int np = static_cast<int>(std::ceil(k_ * rho_max * std::sin(a_max_))) + 16;
        Real check = 1.0;
        for (int round = 0; round < 6; ++round) {
            np = 4 * ((np + 3) / 4);
            const int na2 = na + na / 2 + 8;
            const int np2 = 4 * ((np + np / 2 + 8 + 3) / 4);
            build(na2, np2);
            std::vector<Vec3c> fine(static_cast<std::size_t>(probe.rows()));
            Real emax = 0.0;
            for (Index i = 0; i < probe.rows(); ++i) {
                Vec3c H;
                evaluate(probe.row(i).transpose(), fine[static_cast<std::size_t>(i)], H);
                emax = std::max(emax, fine[static_cast<std::size_t>(i)].norm());
            }
            build(na, np);
            Real dmax = 0.0;
            for (Index i = 0; i < probe.rows(); ++i) {
                Vec3c E, H;
                evaluate(probe.row(i).transpose(), E, H);
                dmax = std::max(dmax, (E - fine[static_cast<std::size_t>(i)]).norm());
            }
            check = dmax / emax;
            if (check < 1e-10)
                return check;
            na = na2;
            np = np2;
        }
        return check;
    }

    [[nodiscard]] Vec3c electric_field(const Vec3& r) const override {
        Cache& c = cache();
        if (c.owner != this || c.generation != generation_ || c.r != r) {
            evaluate(r, c.E, c.H);
            c.owner = this;
            c.generation = generation_;
            c.r = r;
        }
        return c.E;
    }
    [[nodiscard]] Vec3c magnetic_field(const Vec3& r) const override {
        Cache& c = cache();
        if (c.owner != this || c.generation != generation_ || c.r != r) {
            evaluate(r, c.E, c.H);
            c.owner = this;
            c.generation = generation_;
            c.r = r;
        }
        return c.H;
    }
    [[nodiscard]] std::size_t num_waves() const { return kx_.size(); }
    [[nodiscard]] int n_alpha() const { return n_alpha_; }
    [[nodiscard]] int n_phi() const { return n_phi_; }
    [[nodiscard]] Real alpha_max() const { return a_max_; }

private:
    struct Cache {
        const SpectrumGaussianBeam* owner = nullptr;
        std::uint64_t generation = 0;
        Vec3 r = Vec3::Zero();
        Vec3c E = Vec3c::Zero();
        Vec3c H = Vec3c::Zero();
    };
    static Cache& cache() {
        // The RHS assembly calls electric_field(r) and magnetic_field(r) for the same point in
        // turn (one thread per triangle): one evaluation serves both.
        static thread_local Cache c;
        return c;
    }

    void evaluate(const Vec3& r, Vec3c& E, Vec3c& H) const {
        Real erx = 0, ery = 0, erz = 0, eix = 0, eiy = 0, eiz = 0;
        Real hrx = 0, hry = 0, hrz = 0, hix = 0, hiy = 0, hiz = 0;
        const std::size_t n = kx_.size();
        for (std::size_t i = 0; i < n; ++i) {
            Real s = 0, c = 0;
            kernels::fastmath::fast_sincos(kx_[i] * r.x() + ky_[i] * r.y() + kz_[i] * r.z(), s, c);
            // exp(-j phase) = c - j s
            erx += ex_[i] * c;
            ery += ey_[i] * c;
            erz += ez_[i] * c;
            eix -= ex_[i] * s;
            eiy -= ey_[i] * s;
            eiz -= ez_[i] * s;
            hrx += hx_[i] * c;
            hry += hy_[i] * c;
            hrz += hz_[i] * c;
            hix -= hx_[i] * s;
            hiy -= hy_[i] * s;
            hiz -= hz_[i] * s;
        }
        E = Vec3c(Complex(erx, eix), Complex(ery, eiy), Complex(erz, eiz));
        H = Vec3c(Complex(hrx, hix), Complex(hry, hiy), Complex(hrz, hiz));
    }

    Real k_;
    Real w0_;
    Real a_max_ = 0;
    int n_alpha_ = 0;
    int n_phi_ = 0;
    std::uint64_t generation_ = 0;
    std::vector<Real> kx_, ky_, kz_, ex_, ey_, ez_, hx_, hy_, hz_;
};

// ---------------------------------------------------------------------------------------------
// Far-field grids, .npy storage and comparison (readers:
// tests/support/ag_box_mechanism_support.hpp).

using ag_box_mechanism::read_attributes;
using ag_box_mechanism::read_npy;

MatrixXr read_npy_real(const std::filesystem::path& path) {
    return read_npy<Real>(path);
}

/// Tolerance of the runs stored before the "converged" attribute existed (all used GMRES tol
/// <= 1e-6 and recorded the true residual).
constexpr Real kLegacyTolerance = 1e-6;

struct StoredRun {
    FarField ff;
    bool converged = false;
    bool diagnostic = false;
    std::string residual;  // as stored, for messages
};

/// Loads a stored run. Throws unless it converged (attribute "converged" == "yes"; for runs
/// stored before that attribute: true residual <= kLegacyTolerance), or allow_unconverged.
StoredRun load_run(const std::string& dir, bool allow_unconverged) {
    const std::filesystem::path d = core::path_from_utf8(dir);
    StoredRun s;
    std::optional<std::string> converged, residual;
    for (const auto& [key, value] : read_attributes(d / "attributes.json")) {
        if (key == "converged")
            converged = value;
        else if (key == "true_residual")
            residual = value;
        else if (key == "diagnostic")
            s.diagnostic = !value.empty() && value != "no";
    }
    s.residual = residual.value_or("not recorded");
    if (converged)
        s.converged = *converged == "yes";
    else if (residual)
        s.converged = std::stod(*residual) <= kLegacyTolerance;
    if (!s.converged && !allow_unconverged) {
        const std::string why = converged
                                    ? "did not converge (true residual " + s.residual + ")"
                                    : "has no \"converged\" attribute and a true residual of " +
                                          s.residual + " (legacy rule: <= 1e-6)";
        throw std::runtime_error("stored run " + dir + " " + why +
                                 "; pass --allow-unconverged to compare it anyway");
    }
    s.ff.refl = read_npy_real(d / "refl_intensity.npy");
    s.ff.fwd = read_npy_real(d / "fwd_intensity.npy");
    s.ff.cut_xz = read_npy_real(d / "cut_xz_intensity.npy");
    s.ff.cut_yz = read_npy_real(d / "cut_yz_intensity.npy");
    return s;
}

/// One COMPARE line; `flags` marks UNCONVERGED / DIAGNOSTIC inputs (empty for a plain comparison).
void print_compare(const std::string& test, const std::string& ref, const Metrics& m,
                   bool unconverged, bool diagnostic) {
    std::string flags;
    if (unconverged)
        flags += " | UNCONVERGED";
    if (diagnostic)
        flags += " | DIAGNOSTIC";
    std::printf(
        "COMPARE | %s vs %s | l2_refl %.3e | l2_off30 %.3e | l2_fwd %.3e | dP_refl %+.3e | "
        "eps_xz %.3e | eps_yz %.3e%s\n",
        test.c_str(), ref.c_str(), m.l2_refl, m.l2_off30, m.l2_fwd, m.dP_refl, m.eps_xz, m.eps_yz,
        flags.c_str());
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------------------------

/// Power radiated into the reflection hemisphere by the currents of one part of the closed box,
/// relative to the total (diagnostic; partial far fields interfere, so the shares need not add
/// up to 1). Triangle classes: top face (n_z < -0.5), side walls (|n_z| <= 0.5), bottom plate
/// (n_z > 0.5); an RWG function belongs to a class if both its triangles do, to "rim" (top-wall)
/// or "foot" (wall-bottom) if it straddles two.
std::string part_shares(const post::SurfaceSolution& s, const geometry::TriangleMesh& mesh,
                        const post::FieldOptions& fo, Real p_total) {
    const basis::RwgSpace& space = *s.problem->space;
    const Index N = space.size();
    std::vector<int> ca(static_cast<std::size_t>(N), -1), cb(static_cast<std::size_t>(N), -1);
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        const Real nz = mesh.normal(t).z();
        const int c = nz < -0.5 ? 0 : (nz > 0.5 ? 2 : 1);
        const basis::RwgSpace::Support sup = space.support(t);
        for (int a = 0; a < sup.count; ++a) {
            const auto n = static_cast<std::size_t>(sup.n[a]);
            (ca[n] < 0 ? ca[n] : cb[n]) = c;
        }
    }
    std::vector<int> part(static_cast<std::size_t>(N));
    for (std::size_t n = 0; n < part.size(); ++n) {
        const int lo = std::min(ca[n], cb[n]);
        const int hi = std::max(ca[n], cb[n]);
        part[n] = lo == hi ? lo : (lo == 0 && hi == 1 ? 3 : (lo == 1 && hi == 2 ? 4 : 5));
    }
    static const char* const names[] = {"top", "walls", "bottom", "rim", "foot", "other"};
    std::string out;
    const Vertices dirs = hemisphere_directions(-1.0);
    for (int c = 0; c < 6; ++c) {
        post::SurfaceSolution sp{s.problem, s.currents};
        Index count = 0;
        for (Index n = 0; n < N; ++n) {
            if (part[static_cast<std::size_t>(n)] == c) {
                ++count;
                continue;
            }
            sp.currents(n) = 0.0;
            sp.currents(N + n) = 0.0;
        }
        if (count == 0)
            continue;
        FieldMatrix F;
        post::far_field(sp, dirs, F, fo);
        char buf[96];
        std::snprintf(buf, sizeof(buf), " | %s %.3e (%lld)", names[c],
                      hemi_power(intensity(F, kHemiTheta, kHemiPhi)) / p_total,
                      static_cast<long long>(count));
        out += buf;
    }
    return out;
}

/// Central crop of a height map to an L x L patch (same grid spacing).
geometry::HeightMap crop(const geometry::HeightMap& big, Real L) {
    const Index nb = big.z.rows();
    const Index ns = static_cast<Index>(std::llround(L / big.dx)) + 1;
    if (ns > nb || (nb - ns) % 2 != 0 || big.z.cols() != nb ||
        std::abs(static_cast<Real>(ns - 1) * big.dx - L) > 1e-9 * L)
        throw std::invalid_argument("L must be a centred sub-grid of the generated patch");
    const Index m = (nb - ns) / 2;
    geometry::HeightMap h;
    h.dx = big.dx;
    h.dy = big.dy;
    h.z = big.z.block(m, m, ns, ns);
    return h;
}

void append_summary(const std::string& path, const std::string& line) {
    if (path.empty())
        return;
    std::ofstream f(path, std::ios::app | std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    f << line << '\n';
}

/// Height map, closing box and mesh of one configuration.
struct BoxGeometry {
    geometry::HeightMap hm;
    std::optional<Real> fine_depth;
    geometry::detail::BoxGrading grading;
    geometry::TriangleMesh mesh;
    Index top = 0;  ///< top-face triangles
};

BoxGeometry build_geometry(const material::Material& object, Real L, Real L_gen, std::uint64_t seed,
                           Real sigma, Real depth, const std::string& box_mesh_size_option,
                           bool fine_band) {
    geometry::RoughSurfaceParams rp;
    rp.edge_length_L = L_gen;
    rp.rms_roughness = sigma;
    rp.correlation_length = kCorrelationLength;
    rp.mesh_size = kMeshSize;
    rp.seed = seed;
    BoxGeometry b;
    b.hm = crop(geometry::generate_gaussian_height_map(rp), L);
    std::optional<Real> box_mesh_size;
    if (box_mesh_size_option == "uniform")
        box_mesh_size = kMeshSize;
    else if (box_mesh_size_option != "auto")
        box_mesh_size = std::stod(box_mesh_size_option);
    if (fine_band)
        b.fine_depth = default_box_fine_depth(object, kLambda, sigma);
    b.grading = geometry::detail::box_grading(b.hm, depth, box_mesh_size, b.fine_depth);
    b.mesh = geometry::make_mesh_from_height_map(b.hm, depth, box_mesh_size, b.fine_depth);
    b.top = 2 * (b.hm.z.rows() - 1) * (b.hm.z.cols() - 1);
    return b;
}

// ---------------------------------------------------------------------------------------------
// WP-V3 diagnostics (tests/support/ag_box_mechanism_support.hpp).

namespace abm = ag_box_mechanism;

// Spectrum grid: |k_u|, |k_v| <= 2.5 k0 in steps of k0 / 32, bins of 0.025 k0.
constexpr Real kSpecKMax = 2.5;
constexpr Real kSpecDk = 1.0 / 32.0;
constexpr Real kSpecBin = 0.025;

/// Combined spectrum of several faces: bin powers weighted by the raw totals.
abm::RadialSpectrum combine(const std::vector<abm::RadialSpectrum>& parts) {
    abm::RadialSpectrum c = parts.front();
    std::fill(c.j.begin(), c.j.end(), 0.0);
    std::fill(c.m.begin(), c.m.end(), 0.0);
    Real tj = 0, tm = 0, aj = 0, am = 0, sj = 0, sm = 0;
    for (const abm::RadialSpectrum& p : parts) {
        if (p.j.size() != c.j.size())
            throw std::runtime_error("combine: different bin counts");
        for (std::size_t b = 0; b < c.j.size(); ++b) {
            c.j[b] += p.j[b] * p.j_total;
            c.m[b] += p.m[b] * p.m_total;
        }
        tj += p.j_total;
        tm += p.m_total;
        aj += p.j_above * p.j_total;
        am += p.m_above * p.m_total;
        sj += p.rms_j * p.rms_j;
        sm += p.rms_m * p.rms_m;
    }
    for (Real& x : c.j) x /= tj;
    for (Real& x : c.m) x /= tm;
    c.j_total = tj;
    c.m_total = tm;
    c.j_above = aj / tj;
    c.m_above = am / tm;
    c.rms_j = std::sqrt(sj / static_cast<Real>(parts.size()));
    c.rms_m = std::sqrt(sm / static_cast<Real>(parts.size()));
    return c;
}

/// SPECSUM line (and SPEC lines with the binned spectrum); returns the summary fragment.
std::string print_spectrum(const std::string& tag, const std::string& face, std::size_t samples,
                           const abm::RadialSpectrum& sp, Real k_spp) {
    const abm::BandPeak pj = abm::band_peak(sp.bin_centre, sp.j, 0.8, 1.4);
    const abm::BandPeak pm = abm::band_peak(sp.bin_centre, sp.m, 0.8, 1.4);
    const abm::BandPeak lj = abm::band_peak(sp.bin_centre, sp.j, 0.3, 0.95);
    const abm::BandPeak lm = abm::band_peak(sp.bin_centre, sp.m, 0.3, 0.95);
    const abm::BandPeak sj = abm::band_peak(sp.bin_centre, sp.j, k_spp - 0.05, k_spp + 0.05);
    const abm::BandPeak sm = abm::band_peak(sp.bin_centre, sp.m, k_spp - 0.05, k_spp + 0.05);
    std::printf(
        "SPECSUM | %s | %s | samples %zu | rms J %.4e A/m | rms M %.4e V/m | J>k0 %.4e | M>k0 "
        "%.4e | J [0.3,0.95] %.4e | M [0.3,0.95] %.4e | J at k_spp+-0.05 %.4e | M at k_spp+-0.05 "
        "%.4e | J [0.8,1.4] peak %.3f centroid %.3f frac %.4e | M [0.8,1.4] peak %.3f centroid "
        "%.3f frac %.4e\n",
        tag.c_str(), face.c_str(), samples, sp.rms_j, sp.rms_m, sp.j_above, sp.m_above, lj.fraction,
        lm.fraction, sj.fraction, sm.fraction, pj.peak, pj.centroid, pj.fraction, pm.peak,
        pm.centroid, pm.fraction);
    for (std::size_t b = 0; b < sp.bin_centre.size(); ++b)
        std::printf("SPEC | %s | %s | k/k0 %.4f | J %.4e | M %.4e\n", tag.c_str(), face.c_str(),
                    sp.bin_centre[b], sp.j[b], sp.m[b]);
    std::fflush(stdout);
    char buf[160];
    std::snprintf(buf, sizeof(buf), " | %s J>k0 %.3e M>k0 %.3e J@spp %.3e M@spp %.3e", face.c_str(),
                  sp.j_above, sp.m_above, sj.fraction, sm.fraction);
    return buf;
}

/// Interior probe lines of the extinction-theorem residual (WP-A2 placement rule, 20 nm spacing,
/// extra x-lines 0.5 and 1.0 um below z_p). @throws std::invalid_argument for a box too small.
std::vector<abm::ProbeLine> mechanism_probe_lines(Real L, Real depth, Real max_height) {
    return abm::interior_probe_lines(L, depth, max_height, kMeshSize, 20e-9, {0.5e-6, 1.0e-6});
}

/// Spectra of the top face, the four walls (combined) and the bottom plate; extinction-theorem
/// residual on the interior probe lines. MECH summary line to `summary`. `top_count` is the
/// generator's number of top-face triangles (checked against the normal-based classification).
void print_mechanism(const std::string& tag, const post::SurfaceSolution& s,
                     const excitation::Excitation& inc, Real L, Real depth, Real max_height,
                     Index top_count, Complex eps, const std::string& summary) {
    const auto t0 = std::chrono::steady_clock::now();
    const Real k0 = 2.0 * constants::pi / kLambda;
    const Complex spp = abm::spp_ratio(eps);
    std::printf("MECH | %s | k_spp / k0 = %.5f %+.5fj | SPP amplitude decay %.3g m\n", tag.c_str(),
                spp.real(), spp.imag(), 1.0 / (-spp.imag() * k0));
    std::string line = "MECH | " + tag;
    {
        const abm::FaceSamples top = abm::sample_face(s, abm::Face::Top, top_count);
        line += print_spectrum(tag, "top", top.size(),
                               abm::radial_spectrum(top, k0, kSpecKMax, kSpecDk, kSpecBin, true),
                               spp.real());
    }
    {
        std::vector<abm::RadialSpectrum> walls;
        std::size_t n = 0;
        for (const abm::Face f :
             {abm::Face::WallXm, abm::Face::WallXp, abm::Face::WallYm, abm::Face::WallYp}) {
            const abm::FaceSamples w = abm::sample_face(s, f);
            n += w.size();
            walls.push_back(abm::radial_spectrum(w, k0, kSpecKMax, kSpecDk, kSpecBin, true));
        }
        line += print_spectrum(tag, "walls", n, combine(walls), spp.real());
    }
    {
        const abm::FaceSamples bottom = abm::sample_face(s, abm::Face::Bottom);
        line += print_spectrum(tag, "bottom", bottom.size(),
                               abm::radial_spectrum(bottom, k0, kSpecKMax, kSpecDk, kSpecBin, true),
                               spp.real());
    }
    // Extinction-theorem residual inside the box (WP-A2 placement rule, 20 nm spacing).
    const std::vector<abm::ProbeLine> lines = mechanism_probe_lines(L, depth, max_height);
    std::vector<Real> ks;
    for (int i = -100; i <= 100; ++i) ks.push_back(0.02 * i * k0);
    Real inc_top = 0;
    for (const abm::ProbeLine& pl : lines) {
        const ag_absorption::ExtinctionResidual er =
            ag_absorption::extinction_residual(s, inc, pl.points);
        const Real r = abm::rms(er.residual);
        const Real e = abm::rms(er.incident);
        if (inc_top == 0)
            inc_top = e;  // the first line is the x-line at z_p
        if (pl.name != "z") {
            const auto [kp, rel] = abm::line_peak(er.residual, pl.coordinate, ks);
            std::printf(
                "INTERIOR | %s | %s-line z %.3f um | %lld points | rms|res|/rms|inc| %.3e | "
                "rms|res|/rms|inc(z_p)| %.3e | rms|inc| %.3e V/m | residual peak k/k0 %+.2f "
                "(%.2f of the tapered sum)\n",
                tag.c_str(), pl.name.c_str(), pl.level * 1e6,
                static_cast<long long>(pl.points.rows()), r / e, r / inc_top, e, kp / k0, rel);
            char buf[96];
            std::snprintf(buf, sizeof(buf), " | int %s z=%.2f %.3e", pl.name.c_str(),
                          pl.level * 1e6, r / inc_top);
            line += buf;
        } else {
            for (Index i = 0; i < pl.points.rows(); ++i)
                std::printf(
                    "INTZ | %s | z %.3f um | |res| %.3e | |res_x| %.3e | |res_y| %.3e | |res_z| "
                    "%.3e | |inc| %.3e (relative to rms|inc(z_p)|)\n",
                    tag.c_str(), pl.coordinate[static_cast<std::size_t>(i)] * 1e6,
                    er.residual.row(i).norm() / inc_top, std::abs(er.residual(i, 0)) / inc_top,
                    std::abs(er.residual(i, 1)) / inc_top, std::abs(er.residual(i, 2)) / inc_top,
                    er.incident.row(i).norm() / inc_top);
            std::printf(
                "INTERIOR | %s | z-line x = y = 0, z %.3f .. %.3f um | %lld points | "
                "rms|res|/rms|inc(z_p)| %.3e\n",
                tag.c_str(), pl.coordinate.front() * 1e6, pl.coordinate.back() * 1e6,
                static_cast<long long>(pl.points.rows()), r / inc_top);
            char buf[64];
            std::snprintf(buf, sizeof(buf), " | int z-line %.3e", r / inc_top);
            line += buf;
        }
        std::fflush(stdout);
    }
    std::printf("MECH | %s | time %.1f s\n", tag.c_str(), seconds_since(t0));
    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
    append_summary(summary, line);
}

/// RITZ lines at the leading Krylov dimensions `ks` (clamped to the Hessenberg): harmonic Ritz
/// values closest to 0 and the smallest singular values (tests/support/krylov_spectrum.hpp).
void print_ritz(const std::string& tag, const MatrixXc& hbar, const std::vector<Index>& ks,
                const std::string& summary) {
    namespace ks_ = krylov_spectrum;
    for (const Index k_req : ks) {
        // parse_options refuses K > --max-iter; a solve that converged earlier has a smaller
        // Hessenberg, whose full block is used instead (the RITZ line shows the actual k).
        Index k = k_req;
        if (k_req > hbar.cols()) {
            SBEM_WARN("--ritz {}: GMRES stopped after {} iterations; using k = {} (converged)",
                      static_cast<long long>(k_req), static_cast<long long>(hbar.cols()),
                      static_cast<long long>(hbar.cols()));
            k = hbar.cols();
        }
        if (k < 2) {
            SBEM_WARN("--ritz {}: Krylov dimension {} < 2, no Ritz values",
                      static_cast<long long>(k_req), static_cast<long long>(k));
            continue;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const MatrixXc h = abm::leading_hessenberg(hbar, k);
        const std::vector<Complex> harm = ks_::sorted_by_modulus(ks_::harmonic_ritz_values(h));
        const VectorXr sv = ks_::singular_values(h);
        const auto below = [&](Real t) {
            return static_cast<long long>(std::count_if(
                harm.begin(), harm.end(), [t](const Complex& z) { return std::abs(z) < t; }));
        };
        char line[512];
        std::snprintf(line, sizeof(line),
                      "RITZ | %s | k %lld | sigma_min %.4e | sigma_max %.4e | harmonic |theta| < "
                      "0.01: %lld, < 0.03: %lld, < 0.1: %lld, < 0.3: %lld | smallest |theta| %.4e "
                      "%.4e %.4e | %.1f s",
                      tag.c_str(), static_cast<long long>(k), sv(sv.size() - 1), sv(0), below(0.01),
                      below(0.03), below(0.1), below(0.3), std::abs(harm[0]),
                      std::abs(harm[std::min<std::size_t>(1, harm.size() - 1)]),
                      std::abs(harm[std::min<std::size_t>(2, harm.size() - 1)]), seconds_since(t0));
        std::printf("%s\n", line);
        append_summary(summary, line);
        const std::size_t n_show = std::min<std::size_t>(24, harm.size());
        for (std::size_t i = 0; i < n_show; ++i)
            std::printf(
                "RITZV | %s | k %lld | %2zu | theta %+.5e %+.5ej | |theta| %.5e | "
                "sigma_(k-%zu) %.5e\n",
                tag.c_str(), static_cast<long long>(k), i, harm[i].real(), harm[i].imag(),
                std::abs(harm[i]), i, sv(sv.size() - 1 - static_cast<Index>(i)));
        std::fflush(stdout);
    }
}

/// Value of a key of a stored run's attributes. @throws std::runtime_error if missing.
std::string attribute(const abm::Attributes& attrs, const std::string& key) {
    if (const std::optional<std::string> v = abm::find_attribute(attrs, key))
        return *v;
    throw std::runtime_error("stored run has no attribute " + key);
}

/// A stored solution rebuilt from its run directory (mesh from the attributes, currents.npy).
struct LoadedRun {
    std::unique_ptr<BoxGeometry> geo;
    std::unique_ptr<basis::RwgSpace> space;
    std::unique_ptr<formulation::Formulation> form;
    op::Problem problem;
    VectorXc currents;
    Real depth = 0, L = 0;
};

/// Loads a stored run after checking its attributes against `expected` (the configuration of the
/// command line, ag_box_mechanism::require_attributes; `context` names the option).
std::unique_ptr<LoadedRun> load_solution(const std::string& dir, const material::Material& object,
                                         const excitation::Excitation& inc, bool fine_band,
                                         const abm::Attributes& expected,
                                         const std::string& context) {
    const std::filesystem::path d = core::path_from_utf8(dir);
    const auto attrs = read_attributes(d / "attributes.json");
    abm::require_attributes(attrs, expected, context + " " + dir);
    auto r = std::make_unique<LoadedRun>();
    r->L = std::stod(attribute(attrs, "edge_length_L"));
    r->depth = std::stod(attribute(attrs, "box_depth"));
    const Real sigma = std::stod(attribute(attrs, "rms_roughness"));
    r->geo = std::make_unique<BoxGeometry>(build_geometry(
        object, r->L, std::stod(attribute(attrs, "map_L")),
        static_cast<std::uint64_t>(std::llround(std::stod(attribute(attrs, "seed")))), sigma,
        r->depth, attribute(attrs, "box_mesh_size"), fine_band));
    r->space = std::make_unique<basis::RwgSpace>(r->geo->mesh);
    r->form = formulation::make_formulation(formulation::Kind::ICTF);
    r->problem.space = r->space.get();
    r->problem.exterior = material::vacuum();
    r->problem.object = object;
    r->problem.formulation = r->form.get();
    r->problem.excitation = &inc;
    r->problem.omega = inc.omega();
    const MatrixXc x = read_npy<Complex>(d / "currents.npy");
    if (x.size() != 2 * r->space->size())
        throw std::runtime_error("currents.npy of " + dir + " does not match its mesh (2N = " +
                                 std::to_string(2 * r->space->size()) + ")");
    r->currents = Eigen::Map<const VectorXc>(x.data(), x.size());
    return r;
}

/// Spectrum of the top-face current difference a - b on their common triangles (DIFF lines).
/// top_a, top_b: the generator's top-face triangle counts of the two meshes.
void print_difference(const std::string& tag, const std::string& other,
                      const post::SurfaceSolution& a, const post::SurfaceSolution& b, Index top_a,
                      Index top_b, Complex eps, const std::string& summary) {
    const Real k0 = 2.0 * constants::pi / kLambda;
    const abm::FaceSamples ta = abm::sample_face(a, abm::Face::Top, top_a);
    const abm::FaceSamples tb = abm::sample_face(b, abm::Face::Top, top_b);
    const auto pairs = abm::match_points(ta.points, tb.points, 1e-3 * kMeshSize);
    if (pairs.size() < 4)
        throw std::runtime_error("--diff: the top faces have no common triangles");
    const abm::FaceSamples d = abm::difference(ta, tb, pairs);
    const abm::FaceSamples ca = abm::subset(ta, pairs);
    const abm::RadialSpectrum sd = abm::radial_spectrum(d, k0, kSpecKMax, kSpecDk, kSpecBin, true);
    const abm::RadialSpectrum sa = abm::radial_spectrum(ca, k0, kSpecKMax, kSpecDk, kSpecBin, true);
    const Real spp = abm::spp_ratio(eps).real();
    std::printf(
        "DIFF | %s - %s | top-face triangles matched %zu of %zu / %zu | rms|dJ|/rms|J| "
        "%.4e | rms|dM|/rms|M| %.4e | tapered spectral power |dJ|^2/|J|^2 %.4e, "
        "|dM|^2/|M|^2 %.4e\n",
        tag.c_str(), other.c_str(), pairs.size(), ta.size(), tb.size(), sd.rms_j / sa.rms_j,
        sd.rms_m / sa.rms_m, sd.j_total / sa.j_total, sd.m_total / sa.m_total);
    const std::string frag = print_spectrum(tag + " - " + other, "top-diff", d.size(), sd, spp);
    print_spectrum(tag + " (common)", "top", ca.size(), sa, spp);
    // Which part radiates the far-field change? Far field (reflection hemisphere, Dunavant 6) of
    // the full solutions and of the top-face basis-function difference alone (placed on a's mesh;
    // the rim functions that straddle top and wall are not included).
    const auto [xb_top, nb] = abm::transfer_face_currents(
        *b.problem->space, b.currents, *a.problem->space, abm::Face::Top, 1e-3 * kMeshSize);
    const auto [xa_top, na] = abm::transfer_face_currents(
        *a.problem->space, a.currents, *a.problem->space, abm::Face::Top, 1e-3 * kMeshSize);
    post::FieldOptions fo;
    fo.quad_degree = 6;
    const Vertices dirs = hemisphere_directions(-1.0);
    FieldMatrix Fa, Fb, Ft;
    post::far_field(a, dirs, Fa, fo);
    post::far_field(b, dirs, Fb, fo);
    post::far_field(post::SurfaceSolution{a.problem, VectorXc(xa_top - xb_top)}, dirs, Ft, fo);
    const FieldMatrix dF = Fa - Fb;
    const auto wdot = [](const FieldMatrix& x, const FieldMatrix& y) {
        Complex s = 0;
        for (Index r = 0; r < x.rows(); ++r) {
            const Real w = std::sin((static_cast<Real>(r / kHemiPhi) + 0.5) * kHemiDTheta * kDeg);
            s += w * x.row(r).dot(y.row(r));  // Eigen's dot conjugates the first argument
        }
        return s;
    };
    const Real n_a = std::sqrt(wdot(Fa, Fa).real());
    const Real n_d = std::sqrt(wdot(dF, dF).real());
    const Real n_t = std::sqrt(wdot(Ft, Ft).real());
    const FieldMatrix rest = dF - Ft;
    const Real n_r = std::sqrt(wdot(rest, rest).real());
    char ffb[512];
    std::snprintf(ffb, sizeof(ffb),
                  " | far field: |F_a - F_b|/|F_a| %.4e | top-face part |dF_top|/|dF| %.4e, "
                  "corr %.4f | rest (walls, bottom, rim) |dF - dF_top|/|dF| %.4e | top basis "
                  "functions %zu / %zu",
                  n_d / n_a, n_t / n_d, std::abs(wdot(Ft, dF)) / (n_t * n_d), n_r / n_d, na, nb);
    std::printf("DIFFFF | %s - %s%s\n", tag.c_str(), other.c_str(), ffb);
    std::fflush(stdout);
    char buf[128];
    std::snprintf(buf, sizeof(buf), " | rms dJ/J %.3e dM/M %.3e", sd.rms_j / sa.rms_j,
                  sd.rms_m / sa.rms_m);
    append_summary(summary, "DIFF | " + tag + " - " + other + buf + frag + ffb);
}

/// Travelling depth-wave fit of the complex reflection-hemisphere far fields (refl_F.npy) of
/// stored runs at several box depths (DEPTHFIT lines), over the whole hemisphere and in three
/// theta bands, samples weighted by sqrt(sin theta).
int depth_fit(const std::vector<std::string>& dirs) {
    const Real k0 = 2.0 * constants::pi / kLambda;
    std::vector<MatrixXc> F;
    std::vector<Real> depth;
    std::vector<abm::Attributes> attrs;
    for (const std::string& dir : dirs)
        attrs.push_back(read_attributes(core::path_from_utf8(dir) / "attributes.json"));
    // The runs must differ in the box depth only.
    abm::require_same_configuration(attrs, dirs, {"box_depth"});
    for (std::size_t i = 0; i < dirs.size(); ++i) {
        const std::string& dir = dirs[i];
        const std::filesystem::path d = core::path_from_utf8(dir);
        depth.push_back(std::stod(attribute(attrs[i], "box_depth")));
        F.push_back(read_npy<Complex>(d / "refl_F.npy"));
        if (F.back().rows() != kHemiTheta * kHemiPhi || F.back().cols() != 3)
            throw std::runtime_error("refl_F.npy of " + dir + " has the wrong shape");
    }
    std::vector<Real> betas;
    for (int i = 0; i <= 600; ++i) betas.push_back(0.005 * i * k0);
    const struct {
        const char* name;
        Real lo, hi;
    } bands[] = {{"all", 0, 90},    {"theta<30", 0, 30}, {"30-60", 30, 60}, {"theta>60", 60, 90},
                 {"0-10", 0, 10},   {"10-20", 10, 20},   {"20-30", 20, 30}, {"30-40", 30, 40},
                 {"40-50", 40, 50}, {"50-60", 50, 60},   {"60-70", 60, 70}, {"70-80", 70, 80},
                 {"80-90", 80, 90}};
    for (const auto& band : bands) {
        std::vector<Index> rows;
        for (int i = 0; i < kHemiTheta; ++i) {
            const Real th = (i + 0.5) * kHemiDTheta;
            if (th < band.lo || th >= band.hi)
                continue;
            for (int j = 0; j < kHemiPhi; ++j) rows.push_back(static_cast<Index>(i) * kHemiPhi + j);
        }
        MatrixXc G(static_cast<Index>(dirs.size()), 3 * static_cast<Index>(rows.size()));
        for (std::size_t r = 0; r < dirs.size(); ++r)
            for (std::size_t q = 0; q < rows.size(); ++q) {
                const Real th = (static_cast<Real>(rows[q] / kHemiPhi) + 0.5) * kHemiDTheta * kDeg;
                for (int c = 0; c < 3; ++c)
                    G(static_cast<Index>(r), 3 * static_cast<Index>(q) + c) =
                        std::sqrt(std::sin(th)) * F[r](rows[q], c);
            }
        const auto [b, res] = abm::fit_depth_wave(G, depth, betas);
        const MatrixXc C = G.rowwise() - G.colwise().mean();
        // A wave that runs down the walls with wavenumber k_w and radiates from depth D into the
        // direction theta has beta = k_w + k0 cos(theta): print beta/k0 - cos(band centre).
        const Real centre = 0.5 * (band.lo + band.hi) * kDeg;
        std::printf(
            "DEPTHFIT | %s | runs %zu | rms depth-dependent part / rms mean %.4e | best "
            "beta/k0 %.3f | residual %.4e | beta/k0 - cos(theta_centre) %.3f\n",
            band.name, dirs.size(),
            std::sqrt(C.squaredNorm() / G.colwise().mean().squaredNorm() /
                      static_cast<Real>(dirs.size())),
            b / k0, res, b / k0 - std::cos(centre));
        for (std::size_t i = 0; i < betas.size(); i += 10)
            std::printf("DFCURVE | %s | beta/k0 %.3f | residual %.4e\n", band.name, betas[i] / k0,
                        abm::depth_wave_residual(G, depth, betas[i]));
        std::fflush(stdout);
    }
    return 0;
}

int run(const Options& o) {
    material::Material object = material_by_name(o.material);
    if (o.eps_imag)
        object.eps_r = Complex(object.eps_r.real(), *o.eps_imag);
    const Real delta = material::field_decay_length(object, kLambda);
    const Real depth = o.depth ? *o.depth : o.depth_factor * default_box_depth(object, kLambda);
    const Real L_gen = o.L_gen ? *o.L_gen : o.L;
    const Real w0 = o.waist ? *o.waist : o.L / o.waist_factor;
    // eps_r of the object as used (the --eps-imag override makes the run a diagnostic one).
    char eps_buf[96];
    std::snprintf(eps_buf, sizeof(eps_buf), "eps_r %.5g%+.5gj%s", object.eps_r.real(),
                  object.eps_r.imag(), o.eps_imag ? " (DIAGNOSTIC: Im eps_r overridden)" : "");

    const BoxGeometry bg =
        build_geometry(object, o.L, L_gen, o.seed, o.sigma, depth, o.box_mesh_size, o.fine_band);
    const geometry::HeightMap& hm = bg.hm;
    const std::optional<Real>& fine_depth = bg.fine_depth;
    const geometry::detail::BoxGrading& g = bg.grading;
    const geometry::TriangleMesh& mesh = bg.mesh;
    const Index top = bg.top;
    const Index tris = mesh.num_triangles();
    const Index n_edges = mesh.num_edges();
    const Real z_gb = 16.0 * std::pow(2.0 * static_cast<double>(n_edges), 2) / 1e9;
    const std::string tag = o.tag.empty() ? o.material : o.tag;

    // Incident field: rigorous spectrum beam or the paraxial library beam.
    std::shared_ptr<excitation::Excitation> beam;
    std::string beam_info;
    Real r_max = 0, rho_max = 0;
    for (Index v = 0; v < mesh.num_vertices(); ++v) {
        const Vec3 p = mesh.vertices().row(v).transpose();
        r_max = std::max(r_max, p.norm());
        rho_max = std::max(rho_max, std::hypot(p.x(), p.y()));
    }
    if (o.beam == "spectrum") {
        auto sb = std::make_shared<SpectrumGaussianBeam>(kLambda, w0);
        // Probe: about 400 mesh vertices, every (num_vertices / 400)-th in vertex order (spread
        // over the top face, the walls and the bottom by the stride walk).
        const Index stride = std::max<Index>(1, mesh.num_vertices() / 400);
        Vertices probe((mesh.num_vertices() + stride - 1) / stride, 3);
        for (Index v = 0, k = 0; v < mesh.num_vertices(); v += stride, ++k)
            probe.row(k) = mesh.vertices().row(v);
        const auto tb = std::chrono::steady_clock::now();
        const Real check = sb->auto_build(probe, r_max, rho_max);
        char buf[200];
        std::snprintf(buf, sizeof(buf),
                      "spectrum (%zu plane waves, %d x %d, alpha_max %.1f deg, grid check %.1e, "
                      "%.1f s)",
                      sb->num_waves(), sb->n_alpha(), sb->n_phi(), sb->alpha_max() / kDeg, check,
                      seconds_since(tb));
        beam_info = buf;
        if (!(check < 1e-8))
            throw std::runtime_error("spectrum beam grid did not converge: " + beam_info);
        beam = sb;
    } else {
        excitation::GaussianBeam::Params bp;
        bp.wavelength = kLambda;
        bp.waist_radius = w0;
        bp.focus = Vec3::Zero();
        bp.incidence_angle = 0.0;
        bp.polarization = excitation::Polarization::P;
        beam = std::make_shared<excitation::GaussianBeam>(bp);
        beam_info = "paraxial";
    }
    const Real half = 0.5 * o.L;
    const Real e_bottom = beam->electric_field(Vec3(0, 0, depth)).norm();
    const Real e_rim = beam->electric_field(Vec3(half, 0, 0)).norm();
    const Real e_rim_y = beam->electric_field(Vec3(0, half, 0)).norm();
    const Real e_corner = beam->electric_field(Vec3(half, half, 0)).norm();
    const Real e_wall_bottom = beam->electric_field(Vec3(half, 0, depth)).norm();

    char fine_buf[32] = "none";
    if (fine_depth)
        std::snprintf(fine_buf, sizeof(fine_buf), "%.4g m", *fine_depth);
    std::printf(
        "SETUP | %s | %s | %s | delta %.4g m | L %.4g m (map %.4g m, seed %llu, sigma %.4g m) | "
        "depth %.4g m | box_mesh_size %s | M %lld | fine band %s (%lld rows) | triangles %lld "
        "(top %lld, box %lld = %.1f %%) | 2N %lld | Z %.2f GB | w0 %.4g m (L/%.3g) | beam %s | "
        "|E_inc| bottom centre %.3e, rim x %.3e, rim y %.3e, corner %.3e, wall foot x %.3e\n",
        tag.c_str(), o.material.c_str(), eps_buf, delta, o.L, L_gen,
        static_cast<unsigned long long>(o.seed), o.sigma, depth, o.box_mesh_size.c_str(),
        static_cast<long long>(g.levels), fine_buf, static_cast<long long>(g.fine_rows),
        static_cast<long long>(tris), static_cast<long long>(top),
        static_cast<long long>(tris - top),
        100.0 * static_cast<double>(tris - top) / static_cast<double>(top),
        static_cast<long long>(2 * n_edges), z_gb, w0, o.L / w0, beam_info.c_str(), e_bottom, e_rim,
        e_rim_y, e_corner, e_wall_bottom);
    std::fflush(stdout);
    // The interior probe lines of --mechanism must fit into the box: refuse before any assembly.
    if (o.mechanism)
        (void)mechanism_probe_lines(o.L, depth, hm.z.maxCoeff());
    if (o.mesh_only)
        return 0;
    if (!o.load_currents.empty()) {
        // WP-V3: diagnostics of stored currents. The mesh is rebuilt from the run directory's
        // attributes, which must match the command-line configuration (the --diff run: all but
        // L and the depth).
        const auto num = [](Real x) {
            char b[40];
            std::snprintf(b, sizeof(b), "%.17g", x);
            return std::string(b);
        };
        abm::Attributes common = {{"material", o.material},
                                  {"beam", o.beam},
                                  {"wavelength", num(kLambda)},
                                  {"eps_r_real", num(object.eps_r.real())},
                                  {"eps_r_imag", num(object.eps_r.imag())},
                                  {"diagnostic", o.eps_imag ? "eps_imag override" : "no"},
                                  {"rms_roughness", num(o.sigma)},
                                  {"seed", num(static_cast<Real>(o.seed))},
                                  {"map_L", num(L_gen)},
                                  {"waist_radius", num(w0)},
                                  {"box_mesh_size", o.box_mesh_size},
                                  {"fine_band", fine_buf}};
        abm::Attributes own = common;
        own.emplace_back("edge_length_L", num(o.L));
        own.emplace_back("box_depth", num(depth));
        const std::unique_ptr<LoadedRun> a =
            load_solution(o.load_currents, object, *beam, o.fine_band, own, "--load-currents");
        if (a->geo->mesh.num_triangles() != tris)
            throw std::runtime_error("--load-currents: the run directory describes another mesh");
        const post::SurfaceSolution sa{&a->problem, a->currents};
        if (o.mechanism)
            print_mechanism(tag, sa, *beam, o.L, depth, hm.z.maxCoeff(), top, object.eps_r,
                            o.summary);
        if (!o.diff.empty()) {
            const std::unique_ptr<LoadedRun> b =
                load_solution(o.diff, object, *beam, o.fine_band, common, "--diff");
            const post::SurfaceSolution sb{&b->problem, b->currents};
            print_difference(tag, o.diff, sa, sb, top, b->geo->top, object.eps_r, o.summary);
        }
        return 0;
    }
    const bool mlfmm = o.compression == "mlfmm";
    Real peak_est_gb = 0.0;  // MLFMM setup peak model + Krylov basis (+ Hessenberg) at --max-iter
    // --ritz keeps the GMRES Hessenberg, (max_iter + 1) x max_iter complex entries.
    const Real hess_gb = o.ritz.empty() ? 0.0 : hessenberg_gb(o.max_iter);
    if (mlfmm) {
        // Memory guard of the MLFMM scaling study (same octree / leaf rule / budget as the solve).
        mlfmm_scaling::Case c;
        c.material = o.material;
        c.L = o.L;
        c.digits = o.digits;
        c.leaf_radius_quantile = o.leaf_quantile;
        c.tolerance = o.tol;
        c.max_iter = o.max_iter;
        const mlfmm_scaling::Geometry geo{mesh, depth, fine_depth, g, top};
        const mlfmm_scaling::Estimate e = mlfmm_scaling::estimate(geo, c);
        const Real krylov_gb = krylov_basis_gb(o.max_iter, 2 * n_edges) + hess_gb;
        const Real setup_gb = mlfmm_scaling::gb(e.peak_setup_bytes);
        peak_est_gb = setup_gb + krylov_gb;
        const Real available_gb = system_memory::available_memory_bytes() / 1e9;
        std::printf(
            "ESTIMATE | %s | d0 %g | leaf quantile %g | %s | Krylov at max-iter %d: %.2f GB "
            "(Hessenberg %.2f GB) | peak estimate %.2f GB | available %.1f GB\n",
            tag.c_str(), o.digits, o.leaf_quantile, mlfmm_scaling::summary(e).c_str(), o.max_iter,
            krylov_gb, hess_gb, peak_est_gb, available_gb);
        std::fflush(stdout);
        if (o.estimate_only)
            return 0;
        check_memory_guard(setup_gb, krylov_gb, available_gb, o.max_gb, o.force);
    } else {
        if (o.estimate_only)
            return 0;
        if (z_gb + hess_gb > o.max_gb)
            throw std::runtime_error("dense Z of " + std::to_string(z_gb) + " GB (+ Hessenberg " +
                                     std::to_string(hess_gb) + " GB) exceeds --max-gb");
    }

    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.exterior = material::vacuum();
    cfg.object = object;
    cfg.formulation = formulation::Kind::ICTF;
    if (mlfmm) {
        cfg.compression = "mlfmm";
        cfg.mlfmm.accuracy_digits = o.digits;
        cfg.mlfmm.leaf_radius_quantile = o.leaf_quantile;
        cfg.diagonal_preconditioner = o.jacobi;  // empty: formulation::recommend
    } else {
        cfg.diagonal_preconditioner = o.jacobi.value_or(false);
    }
    cfg.gmres.tolerance = o.tol;
    cfg.gmres.max_iter = o.max_iter;
    cfg.gmres.restart = 0;
    cfg.gmres.verbose = true;
    cfg.gmres.keep_hessenberg = !o.ritz.empty();
    Simulation sim(mesh, beam, cfg);
    const auto t0 = std::chrono::steady_clock::now();
    sim.assemble();
    const double t_asm = seconds_since(t0);
    const Real peak_setup_gb = system_memory::peak_rss_bytes() / 1e9;
    const Real operator_gb = static_cast<Real>(sim.system_operator()->memory_bytes()) / 1e9;
    const solver::GmresResult res = sim.solve();
    const bool jacobi = sim.diagonal_preconditioner();
    if (!o.history.empty()) {
        std::ofstream h(core::path_from_utf8(o.history), std::ios::binary);
        if (!h)
            throw std::runtime_error("cannot open " + o.history);
        h << "# " << tag << " GMRES residual history (" << (jacobi ? "left Jacobi, " : "")
          << "monitored relative residual per iteration; true residual "
          << res.true_relative_residual << ")\n";
        char buf[32];
        for (const Real r : res.residual_history) {
            std::snprintf(buf, sizeof(buf), "%.6e\n", r);
            h << buf;
        }
    }

    const auto tf = std::chrono::steady_clock::now();
    post::FieldOptions fo;
    fo.quad_degree = o.ff_degree;
    FieldMatrix F;
    FarField ff;
    post::far_field(sim.solution(), hemisphere_directions(-1.0), F, fo);
    ff.refl = intensity(F, kHemiTheta, kHemiPhi);
    const MatrixXc refl_F = F;  // complex amplitudes (WP-V3 depth-wave fit)
    post::far_field(sim.solution(), hemisphere_directions(+1.0), F, fo);
    ff.fwd = intensity(F, kHemiTheta, kHemiPhi);
    FieldMatrix Fxz, Fyz;
    post::far_field(sim.solution(), cut_directions(false), Fxz, fo);
    post::far_field(sim.solution(), cut_directions(true), Fyz, fo);
    ff.cut_xz = intensity(Fxz, kCut, 1);
    ff.cut_yz = intensity(Fyz, kCut, 1);
    // Far-field quadrature check: degree 8 against the chosen degree (reflection hemisphere).
    post::FieldOptions fo_low;
    fo_low.quad_degree = 8;
    post::far_field(sim.solution(), hemisphere_directions(-1.0), F, fo_low);
    const Real ff_check = hemi_l2(intensity(F, kHemiTheta, kHemiPhi), ff.refl, 0.0);
    const double t_ff = seconds_since(tf);
    const Real p_refl = hemi_power(ff.refl);
    const Real p_fwd = hemi_power(ff.fwd);

    char line[1024];
    std::snprintf(
        line, sizeof(line),
        "RESULT | %s | %s | %s | L %.4g | w0 %.4g | depth %.4g | box_mesh_size %s | fine %s | beam "
        "%s | 2N %lld | iterations %d | converged %s | true residual %.3e | assembly %.1f s | "
        "solve "
        "%.1f s | far field %.1f s | ff degree check %.2e | P_refl %.6e | P_fwd %.6e",
        tag.c_str(), o.material.c_str(), eps_buf, o.L, w0, depth, o.box_mesh_size.c_str(), fine_buf,
        o.beam.c_str(), static_cast<long long>(2 * n_edges), res.iterations,
        res.converged ? "yes" : "no", res.true_relative_residual, t_asm, res.wall_seconds, t_ff,
        ff_check, p_refl, p_fwd);
    char extra[512];
    std::snprintf(
        extra, sizeof(extra),
        " | compression %s | d0 %g | lq %g | jacobi %s | tol %.1e | operator %.2f GB | peak "
        "after setup %.2f GB | peak %.2f GB | peak estimate %.2f GB",
        o.compression.c_str(), mlfmm ? o.digits : 0.0, o.leaf_quantile, jacobi ? "yes" : "no",
        o.tol, operator_gb, peak_setup_gb, system_memory::peak_rss_bytes() / 1e9, peak_est_gb);
    std::strncat(line, extra, sizeof(line) - std::strlen(line) - 1);
    std::printf("%s\n", line);
    std::fflush(stdout);
    append_summary(o.summary, line);
    // An unconverged solution is not a result of the configuration: refuse to store or compare it.
    if (!res.converged && !o.allow_unconverged) {
        throw std::runtime_error(
            "GMRES did not converge (" + std::to_string(res.iterations) +
            " iterations); nothing written or compared (--allow-unconverged to override)");
    }
    if (o.parts) {
        const std::string parts = part_shares(sim.solution(), mesh, fo, p_refl);
        std::printf("PARTS | %s | P_part / P_refl%s\n", tag.c_str(), parts.c_str());
        std::fflush(stdout);
        append_summary(o.summary, "PARTS | " + tag + " | P_part / P_refl" + parts);
    }

    if (!o.out_dir.empty()) {
        auto w = io::open_npy_directory(o.out_dir);
        w->write_matrix("refl_intensity", ff.refl);
        w->write_matrix("fwd_intensity", ff.fwd);
        w->write_matrix("cut_xz_intensity", ff.cut_xz);
        w->write_matrix("cut_yz_intensity", ff.cut_yz);
        w->write_matrix("cut_xz_F", MatrixXc(Fxz));
        w->write_matrix("cut_yz_F", MatrixXc(Fyz));
        w->write_matrix("refl_F", refl_F);
        if (o.save_currents)
            w->write_vector("currents", sim.solution().currents);
        if (!o.ritz.empty())
            w->write_matrix("hessenberg", res.hessenberg);
        MatrixXr hist(static_cast<Index>(res.residual_history.size()), 1);
        for (std::size_t k = 0; k < res.residual_history.size(); ++k)
            hist(static_cast<Index>(k), 0) = res.residual_history[k];
        w->write_matrix("residual_history", hist);
        w->write_attribute("tag", tag);
        w->write_attribute("material", o.material);
        w->write_attribute("beam", o.beam);
        w->write_attribute("box_mesh_size", o.box_mesh_size);
        w->write_attribute("fine_band", fine_buf);
        w->write_attribute("wavelength", kLambda);
        w->write_attribute("edge_length_L", o.L);
        w->write_attribute("map_L", L_gen);
        w->write_attribute("seed", static_cast<Real>(o.seed));
        w->write_attribute("rms_roughness", o.sigma);
        w->write_attribute("waist_radius", w0);
        w->write_attribute("box_depth", depth);
        w->write_attribute("unknowns", static_cast<Real>(2 * n_edges));
        w->write_attribute("triangles", static_cast<Real>(tris));
        w->write_attribute("top_triangles", static_cast<Real>(top));
        w->write_attribute("eps_r_real", object.eps_r.real());
        w->write_attribute("eps_r_imag", object.eps_r.imag());
        w->write_attribute("diagnostic", o.eps_imag ? "eps_imag override" : "no");
        w->write_attribute("iterations", static_cast<Real>(res.iterations));
        w->write_attribute("converged", res.converged ? "yes" : "no");
        w->write_attribute("gmres_tolerance", o.tol);
        w->write_attribute("compression", o.compression);
        w->write_attribute("accuracy_digits", mlfmm ? o.digits : 0.0);
        w->write_attribute("jacobi", jacobi ? "yes" : "no");
        w->write_attribute("leaf_radius_quantile", o.leaf_quantile);
        w->write_attribute("true_residual", res.true_relative_residual);
        w->write_attribute("assembly_seconds", t_asm);
        w->write_attribute("solve_seconds", res.wall_seconds);
        w->write_attribute("ff_degree", static_cast<Real>(o.ff_degree));
        w->write_attribute("hemisphere_dtheta_deg", kHemiDTheta);
        w->write_attribute("hemisphere_dphi_deg", kHemiDPhi);
        w->write_attribute("cut_step_deg", kCutStep);
    }
    for (const std::string& ref : o.references) {
        const StoredRun r = load_run(ref, o.allow_unconverged);
        print_compare(tag, ref, compare(ff, r.ff), !res.converged || !r.converged,
                      o.eps_imag.has_value() || r.diagnostic);
    }
    if (o.mechanism)
        print_mechanism(tag, sim.solution(), *beam, o.L, depth, hm.z.maxCoeff(), top, object.eps_r,
                        o.summary);
    if (!o.ritz.empty())
        print_ritz(tag, res.hessenberg, o.ritz, o.summary);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse_options(std::vector<std::string>(argv + 1, argv + argc));
        if (!o.compare.empty()) {
            const StoredRun ref = load_run(o.compare[0], o.allow_unconverged);
            for (std::size_t i = 1; i < o.compare.size(); ++i) {
                const StoredRun t = load_run(o.compare[i], o.allow_unconverged);
                print_compare(o.compare[i], o.compare[0], compare(t.ff, ref.ff),
                              !t.converged || !ref.converged, t.diagnostic || ref.diagnostic);
            }
            return 0;
        }
        if (!o.depth_fit.empty())
            return depth_fit(o.depth_fit);
        return run(o);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "box_validity: %s\n", e.what());
        return 1;
    }
}
