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
//        specklebem_box_validity --compare DIR_A DIR_B [DIR_C ...] [--allow-unconverged]
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
#include "specklebem/basis/rwg.hpp"
#include "specklebem/core/logging.hpp"
#include "specklebem/core/path.hpp"
#include "specklebem/excitation/excitation.hpp"
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
#include <utility>
#include <vector>

#include "box_validity_support.hpp"
#include "mlfmm_scaling_support.hpp"
#include "system_memory.hpp"

using namespace specklebem;
using namespace box_validity;

namespace {

constexpr Real kLambda = 500e-9;
constexpr Real kSigma = 50e-9;
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
// Far-field grids, .npy storage and comparison.

/// Minimal reader for the '<f8' C-order .npy files written by io::open_npy_directory.
MatrixXr read_npy_real(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + core::path_to_utf8(path));
    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("not an .npy file: " + core::path_to_utf8(path));
    std::size_t header_len = 0;
    if (magic[6] == 1) {
        unsigned char b[2];
        f.read(reinterpret_cast<char*>(b), 2);
        header_len = static_cast<std::size_t>(b[0]) | (static_cast<std::size_t>(b[1]) << 8);
    } else {
        unsigned char b[4];
        f.read(reinterpret_cast<char*>(b), 4);
        header_len = static_cast<std::size_t>(b[0]) | (static_cast<std::size_t>(b[1]) << 8) |
                     (static_cast<std::size_t>(b[2]) << 16) |
                     (static_cast<std::size_t>(b[3]) << 24);
    }
    std::string header(header_len, ' ');
    f.read(header.data(), static_cast<std::streamsize>(header_len));
    if (header.find("'<f8'") == std::string::npos ||
        header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error("unsupported .npy layout in " + core::path_to_utf8(path));
    const std::size_t s0 = header.find("'shape': (");
    if (s0 == std::string::npos)
        throw std::runtime_error("no shape in " + core::path_to_utf8(path));
    const std::size_t s1 = header.find(')', s0);
    std::string shape = header.substr(s0 + 10, s1 - s0 - 10);
    long long rows = 0, cols = 1;
    const std::size_t comma = shape.find(',');
    rows = std::stoll(shape.substr(0, comma));
    if (comma != std::string::npos && shape.find_first_of("0123456789", comma) != std::string::npos)
        cols = std::stoll(shape.substr(comma + 1));
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m(rows, cols);
    f.read(reinterpret_cast<char*>(m.data()),
           static_cast<std::streamsize>(sizeof(Real) * static_cast<std::size_t>(rows * cols)));
    if (!f)
        throw std::runtime_error("truncated " + core::path_to_utf8(path));
    return MatrixXr(m);
}

/// Minimal reader for the flat attributes.json written by io::open_npy_directory (one
/// `"key": value` pair per line, string or number values). String values are returned without
/// the quotes (the study writes no escaped characters).
std::vector<std::pair<std::string, std::string>> read_attributes(
    const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + core::path_to_utf8(path));
    std::vector<std::pair<std::string, std::string>> attrs;
    std::string line;
    while (std::getline(f, line)) {
        const std::size_t k0 = line.find('"');
        if (k0 == std::string::npos)
            continue;
        const std::size_t k1 = line.find('"', k0 + 1);
        const std::size_t colon = line.find(':', k1 == std::string::npos ? k0 : k1);
        if (k1 == std::string::npos || colon == std::string::npos)
            throw std::runtime_error("malformed line in " + core::path_to_utf8(path));
        std::string value = line.substr(colon + 1);
        while (!value.empty() &&
               (value.back() == ',' || value.back() == '\r' || value.back() == ' '))
            value.pop_back();
        const std::size_t v0 = value.find_first_not_of(' ');
        value = v0 == std::string::npos ? std::string() : value.substr(v0);
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
            value = value.substr(1, value.size() - 2);
        attrs.emplace_back(line.substr(k0 + 1, k1 - k0 - 1), value);
    }
    return attrs;
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

    geometry::RoughSurfaceParams rp;
    rp.edge_length_L = L_gen;
    rp.rms_roughness = kSigma;
    rp.correlation_length = kCorrelationLength;
    rp.mesh_size = kMeshSize;
    rp.seed = o.seed;
    const geometry::HeightMap hm = crop(geometry::generate_gaussian_height_map(rp), o.L);

    std::optional<Real> box_mesh_size;
    if (o.box_mesh_size == "uniform")
        box_mesh_size = kMeshSize;
    else if (o.box_mesh_size != "auto")
        box_mesh_size = std::stod(o.box_mesh_size);
    std::optional<Real> fine_depth;
    if (o.fine_band)
        fine_depth = default_box_fine_depth(object, kLambda, kSigma);

    const geometry::detail::BoxGrading g =
        geometry::detail::box_grading(hm, depth, box_mesh_size, fine_depth);
    geometry::TriangleMesh mesh =
        geometry::make_mesh_from_height_map(hm, depth, box_mesh_size, fine_depth);
    const Index top = 2 * (hm.z.rows() - 1) * (hm.z.cols() - 1);
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
        "SETUP | %s | %s | %s | delta %.4g m | L %.4g m (map %.4g m, seed %llu) | depth %.4g m | "
        "box_mesh_size %s | M %lld | fine band %s (%lld rows) | triangles %lld (top %lld, box %lld "
        "= %.1f %%) | 2N %lld | Z %.2f GB | w0 %.4g m (L/%.3g) | beam %s | |E_inc| bottom centre "
        "%.3e, rim x %.3e, rim y %.3e, corner %.3e, wall foot x %.3e\n",
        tag.c_str(), o.material.c_str(), eps_buf, delta, o.L, L_gen,
        static_cast<unsigned long long>(o.seed), depth, o.box_mesh_size.c_str(),
        static_cast<long long>(g.levels), fine_buf, static_cast<long long>(g.fine_rows),
        static_cast<long long>(tris), static_cast<long long>(top),
        static_cast<long long>(tris - top),
        100.0 * static_cast<double>(tris - top) / static_cast<double>(top),
        static_cast<long long>(2 * n_edges), z_gb, w0, o.L / w0, beam_info.c_str(), e_bottom, e_rim,
        e_rim_y, e_corner, e_wall_bottom);
    std::fflush(stdout);
    if (o.mesh_only)
        return 0;
    const bool mlfmm = o.compression == "mlfmm";
    Real peak_est_gb = 0.0;  // MLFMM setup peak model + Krylov basis at --max-iter
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
        const Real krylov_gb = krylov_basis_gb(o.max_iter, 2 * n_edges);
        const Real setup_gb = mlfmm_scaling::gb(e.peak_setup_bytes);
        peak_est_gb = setup_gb + krylov_gb;
        const Real available_gb = system_memory::available_memory_bytes() / 1e9;
        std::printf(
            "ESTIMATE | %s | d0 %g | leaf quantile %g | %s | Krylov at max-iter %d: %.2f GB | peak "
            "estimate %.2f GB "
            "| available %.1f GB\n",
            tag.c_str(), o.digits, o.leaf_quantile, mlfmm_scaling::summary(e).c_str(), o.max_iter,
            krylov_gb, peak_est_gb, available_gb);
        std::fflush(stdout);
        if (o.estimate_only)
            return 0;
        check_memory_guard(setup_gb, krylov_gb, available_gb, o.max_gb, o.force);
    } else {
        if (o.estimate_only)
            return 0;
        if (z_gb > o.max_gb)
            throw std::runtime_error("dense Z of " + std::to_string(z_gb) + " GB exceeds --max-gb");
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
        return run(o);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "box_validity: %s\n", e.what());
        return 1;
    }
}
