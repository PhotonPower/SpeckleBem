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
// Usage: specklebem_box_validity --material ag|si [--L 2e-6] [--L-gen 2.4e-6] [--seed 1]
//            [--waist W | --waist-factor 3] [--depth-factor 1 | --depth D]
//            [--box-mesh-size 400e-9|auto|uniform] [--fine-band] [--beam spectrum|paraxial]
//            [--tol 1e-6] [--max-iter 6000] [--max-gb 60] [--ff-degree 10] [--tag NAME]
//            [--out DIR] [--summary FILE] [--reference DIR]... [--mesh-only]
//        specklebem_box_validity --compare DIR_A DIR_B [DIR_C ...]
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

using namespace specklebem;

namespace {

constexpr Real kLambda = 500e-9;
constexpr Real kSigma = 50e-9;
constexpr Real kCorrelationLength = 500e-9;
constexpr Real kMeshSize = 50e-9;
constexpr Real kDeg = constants::pi / 180.0;

// Far-field grids (see the file comment).
constexpr int kHemiTheta = 90;
constexpr int kHemiPhi = 120;
constexpr Real kHemiDTheta = 1.0;  // deg
constexpr Real kHemiDPhi = 3.0;    // deg
constexpr int kCut = 359;
constexpr Real kCutStep = 0.5;  // deg
constexpr Real kOffSpecularTheta = 30.0;

using FieldMatrix = Eigen::Matrix<Complex, Eigen::Dynamic, 3>;

struct Options {
    std::string material = "ag";
    Real L = 2e-6;
    std::optional<Real> L_gen;
    std::uint64_t seed = 1;
    std::optional<Real> waist;
    Real waist_factor = 3.0;
    Real depth_factor = 1.0;
    std::optional<Real> depth;
    std::string box_mesh_size = "400e-9";
    bool fine_band = false;
    std::string beam = "spectrum";
    Real tol = 1e-6;
    int max_iter = 6000;
    Real max_gb = 60.0;
    int ff_degree = 10;
    std::string tag;
    std::string out_dir;
    std::string summary;
    std::vector<std::string> references;
    std::vector<std::string> compare;
    bool mesh_only = false;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::invalid_argument("missing value after " + a);
            return argv[++i];
        };
        if (a == "--material") {
            o.material = value();
        } else if (a == "--L") {
            o.L = std::stod(value());
        } else if (a == "--L-gen") {
            o.L_gen = std::stod(value());
        } else if (a == "--seed") {
            o.seed = std::stoull(value());
        } else if (a == "--waist") {
            o.waist = std::stod(value());
        } else if (a == "--waist-factor") {
            o.waist_factor = std::stod(value());
        } else if (a == "--depth-factor") {
            o.depth_factor = std::stod(value());
        } else if (a == "--depth") {
            o.depth = std::stod(value());
        } else if (a == "--box-mesh-size") {
            o.box_mesh_size = value();
        } else if (a == "--fine-band") {
            o.fine_band = true;
        } else if (a == "--beam") {
            o.beam = value();
        } else if (a == "--tol") {
            o.tol = std::stod(value());
        } else if (a == "--max-iter") {
            o.max_iter = std::stoi(value());
        } else if (a == "--max-gb") {
            o.max_gb = std::stod(value());
        } else if (a == "--ff-degree") {
            o.ff_degree = std::stoi(value());
        } else if (a == "--tag") {
            o.tag = value();
        } else if (a == "--out") {
            o.out_dir = value();
        } else if (a == "--summary") {
            o.summary = value();
        } else if (a == "--reference") {
            o.references.push_back(value());
        } else if (a == "--compare") {
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0)
                o.compare.emplace_back(argv[++i]);
        } else if (a == "--mesh-only") {
            o.mesh_only = true;
        } else {
            throw std::invalid_argument("unknown argument " + a);
        }
    }
    if (!o.compare.empty()) {
        if (o.compare.size() < 2)
            throw std::invalid_argument("--compare needs at least two directories");
        return o;
    }
    if (!(o.L > 0) || (o.L_gen && !(*o.L_gen >= o.L)) || o.seed == 0 ||
        (o.waist && !(*o.waist > 0)) || !(o.waist_factor > 0) || !(o.depth_factor > 0) ||
        (o.depth && !(*o.depth > 0)) || !(o.tol > 0) || o.max_iter < 1 || !(o.max_gb > 0)) {
        throw std::invalid_argument(
            "need L > 0, L-gen >= L, seed > 0 (fixed), waist > 0, waist-factor > 0, depth(-factor) "
            "> 0, tol > 0, max-iter >= 1, max-gb > 0");
    }
    if (o.beam != "spectrum" && o.beam != "paraxial")
        throw std::invalid_argument("--beam must be spectrum or paraxial");
    return o;
}

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
        ++generation_;
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

Vertices hemisphere_directions(Real z_sign) {
    Vertices d(kHemiTheta * kHemiPhi, 3);
    for (int i = 0; i < kHemiTheta; ++i) {
        const Real t = (i + 0.5) * kHemiDTheta * kDeg;
        for (int j = 0; j < kHemiPhi; ++j) {
            const Real p = j * kHemiDPhi * kDeg;
            const Index row = static_cast<Index>(i) * kHemiPhi + j;
            d(row, 0) = std::sin(t) * std::cos(p);
            d(row, 1) = std::sin(t) * std::sin(p);
            d(row, 2) = z_sign * std::cos(t);
        }
    }
    return d;
}

Vertices cut_directions(bool yz) {
    Vertices d(kCut, 3);
    for (int m = 0; m < kCut; ++m) {
        const Real t = (-89.5 + m * kCutStep) * kDeg;
        d(m, 0) = yz ? 0.0 : std::sin(t);
        d(m, 1) = yz ? std::sin(t) : 0.0;
        d(m, 2) = -std::cos(t);
    }
    return d;
}

MatrixXr intensity(const FieldMatrix& F, Index rows, Index cols) {
    MatrixXr I(rows, cols);
    for (Index r = 0; r < rows; ++r)
        for (Index c = 0; c < cols; ++c) I(r, c) = F.row(r * cols + c).squaredNorm();
    return I;
}

struct FarField {
    MatrixXr refl;    // kHemiTheta x kHemiPhi
    MatrixXr fwd;     // kHemiTheta x kHemiPhi
    MatrixXr cut_xz;  // kCut x 1
    MatrixXr cut_yz;  // kCut x 1
};

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

FarField load_far_field(const std::string& dir) {
    const std::filesystem::path d = core::path_from_utf8(dir);
    FarField ff;
    ff.refl = read_npy_real(d / "refl_intensity.npy");
    ff.fwd = read_npy_real(d / "fwd_intensity.npy");
    ff.cut_xz = read_npy_real(d / "cut_xz_intensity.npy");
    ff.cut_yz = read_npy_real(d / "cut_yz_intensity.npy");
    return ff;
}

struct Metrics {
    Real l2_refl = 0, l2_off30 = 0, l2_fwd = 0, dP_refl = 0, eps_xz = 0, eps_yz = 0;
};

/// Relative L2 difference over a hemisphere grid with solid-angle weights sin(theta), only rows
/// with theta >= theta_min_deg.
Real hemi_l2(const MatrixXr& t, const MatrixXr& r, Real theta_min_deg) {
    Real num = 0, den = 0;
    for (Index i = 0; i < r.rows(); ++i) {
        const Real th = (static_cast<Real>(i) + 0.5) * kHemiDTheta;
        if (th < theta_min_deg)
            continue;
        const Real w = std::sin(th * kDeg);
        for (Index j = 0; j < r.cols(); ++j) {
            num += w * (t(i, j) - r(i, j)) * (t(i, j) - r(i, j));
            den += w * r(i, j) * r(i, j);
        }
    }
    return std::sqrt(num / den);
}

Real hemi_power(const MatrixXr& I) {
    Real p = 0;
    for (Index i = 0; i < I.rows(); ++i) {
        const Real w = std::sin((static_cast<Real>(i) + 0.5) * kHemiDTheta * kDeg);
        p += w * I.row(i).sum();
    }
    return p * (kHemiDTheta * kDeg) * (kHemiDPhi * kDeg);
}

Real eps_rr(const MatrixXr& t, const MatrixXr& r) {
    const Real rms = std::sqrt((t - r).squaredNorm() / static_cast<Real>(r.size()));
    return rms / r.maxCoeff();
}

Metrics compare(const FarField& t, const FarField& r) {
    if (t.refl.rows() != r.refl.rows() || t.refl.cols() != r.refl.cols() ||
        t.cut_xz.size() != r.cut_xz.size())
        throw std::runtime_error("far-field grids differ");
    Metrics m;
    m.l2_refl = hemi_l2(t.refl, r.refl, 0.0);
    m.l2_off30 = hemi_l2(t.refl, r.refl, kOffSpecularTheta);
    m.l2_fwd = hemi_l2(t.fwd, r.fwd, 0.0);
    m.dP_refl = hemi_power(t.refl) / hemi_power(r.refl) - 1.0;
    m.eps_xz = eps_rr(t.cut_xz, r.cut_xz);
    m.eps_yz = eps_rr(t.cut_yz, r.cut_yz);
    return m;
}

void print_compare(const std::string& test, const std::string& ref, const Metrics& m) {
    std::printf(
        "COMPARE | %s vs %s | l2_refl %.3e | l2_off30 %.3e | l2_fwd %.3e | dP_refl %+.3e | "
        "eps_xz %.3e | eps_yz %.3e\n",
        test.c_str(), ref.c_str(), m.l2_refl, m.l2_off30, m.l2_fwd, m.dP_refl, m.eps_xz, m.eps_yz);
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
    const material::Material object = material_by_name(o.material);
    const Real delta = material::field_decay_length(object, kLambda);
    const Real depth = o.depth ? *o.depth : o.depth_factor * default_box_depth(object, kLambda);
    const Real L_gen = o.L_gen ? *o.L_gen : o.L;
    const Real w0 = o.waist ? *o.waist : o.L / o.waist_factor;

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
        // Probe: every 7th mesh vertex (corners and bottom included via the stride walk).
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
        "SETUP | %s | %s | delta %.4g m | L %.4g m (map %.4g m, seed %llu) | depth %.4g m | "
        "box_mesh_size %s | M %lld | fine band %s (%lld rows) | triangles %lld (top %lld, box %lld "
        "= %.1f %%) | 2N %lld | Z %.2f GB | w0 %.4g m (L/%.3g) | beam %s | |E_inc| bottom centre "
        "%.3e, rim x %.3e, rim y %.3e, corner %.3e, wall foot x %.3e\n",
        tag.c_str(), o.material.c_str(), delta, o.L, L_gen, static_cast<unsigned long long>(o.seed),
        depth, o.box_mesh_size.c_str(), static_cast<long long>(g.levels), fine_buf,
        static_cast<long long>(g.fine_rows), static_cast<long long>(tris),
        static_cast<long long>(top), static_cast<long long>(tris - top),
        100.0 * static_cast<double>(tris - top) / static_cast<double>(top),
        static_cast<long long>(2 * n_edges), z_gb, w0, o.L / w0, beam_info.c_str(), e_bottom, e_rim,
        e_rim_y, e_corner, e_wall_bottom);
    std::fflush(stdout);
    if (o.mesh_only)
        return 0;
    if (z_gb > o.max_gb)
        throw std::runtime_error("dense Z of " + std::to_string(z_gb) + " GB exceeds --max-gb");

    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.exterior = material::vacuum();
    cfg.object = object;
    cfg.formulation = formulation::Kind::ICTF;
    cfg.diagonal_preconditioner = false;
    cfg.gmres.tolerance = o.tol;
    cfg.gmres.max_iter = o.max_iter;
    cfg.gmres.restart = 0;
    cfg.gmres.verbose = true;
    Simulation sim(mesh, beam, cfg);
    const auto t0 = std::chrono::steady_clock::now();
    sim.assemble();
    const double t_asm = seconds_since(t0);
    const solver::GmresResult res = sim.solve();

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

    char line[640];
    std::snprintf(
        line, sizeof(line),
        "RESULT | %s | %s | L %.4g | w0 %.4g | depth %.4g | box_mesh_size %s | fine %s | beam %s | "
        "2N %lld | iterations %d | converged %s | true residual %.3e | assembly %.1f s | solve "
        "%.1f s | far field %.1f s | ff degree check %.2e | P_refl %.6e | P_fwd %.6e",
        tag.c_str(), o.material.c_str(), o.L, w0, depth, o.box_mesh_size.c_str(), fine_buf,
        o.beam.c_str(), static_cast<long long>(2 * n_edges), res.iterations,
        res.converged ? "yes" : "no", res.true_relative_residual, t_asm, res.wall_seconds, t_ff,
        ff_check, p_refl, p_fwd);
    std::printf("%s\n", line);
    std::fflush(stdout);
    append_summary(o.summary, line);
    const std::string parts = part_shares(sim.solution(), mesh, fo, p_refl);
    std::printf("PARTS | %s | P_part / P_refl%s\n", tag.c_str(), parts.c_str());
    std::fflush(stdout);
    append_summary(o.summary, "PARTS | " + tag + " | P_part / P_refl" + parts);

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
        w->write_attribute("iterations", static_cast<Real>(res.iterations));
        w->write_attribute("true_residual", res.true_relative_residual);
        w->write_attribute("assembly_seconds", t_asm);
        w->write_attribute("solve_seconds", res.wall_seconds);
        w->write_attribute("ff_degree", static_cast<Real>(o.ff_degree));
        w->write_attribute("hemisphere_dtheta_deg", kHemiDTheta);
        w->write_attribute("hemisphere_dphi_deg", kHemiDPhi);
        w->write_attribute("cut_step_deg", kCutStep);
    }
    for (const std::string& ref : o.references)
        print_compare(tag, ref, compare(ff, load_far_field(ref)));
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        if (!o.compare.empty()) {
            const FarField ref = load_far_field(o.compare[0]);
            for (std::size_t i = 1; i < o.compare.size(); ++i)
                print_compare(o.compare[i], o.compare[0],
                              compare(load_far_field(o.compare[i]), ref));
            return 0;
        }
        return run(o);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "box_validity: %s\n", e.what());
        return 1;
    }
}
