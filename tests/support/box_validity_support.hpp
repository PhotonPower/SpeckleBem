#pragma once
/// @file box_validity_support.hpp
/// Pure parts of the closing-box validity study (WP-V1 dense, WP-V2 MLFMM;
/// benchmarks/box_validity.cpp, records benchmarks/results/box_validity.md and
/// box_validity_mlfmm.md): the far-field grids, the comparison metrics of the docs/05
/// rough-surface sensitivity checks, the MLFMM memory guard decision and the command-line
/// options. Shared by the study executable and the unit test tests/unit/test_box_validity.cpp, so
/// it must not depend on Catch2.
///
/// Grids (directions k_hat, angles from -z on the reflection side):
///   * hemispheres: theta_i = (i + 1/2) deg, i = 0..89, phi_j = 3 j deg, j = 0..119, k_hat =
///     (sin t cos p, sin t sin p, z_sign cos t); z_sign = -1 is the reflection hemisphere (k_z <
///     0), +1 the forward (shadow) hemisphere; row index i * 120 + j;
///   * cuts in the xz- and yz-planes: theta = -89.5 .. 89.5 deg in 0.5 deg steps, k_hat =
///     (sin t, 0, -cos t) and (0, sin t, -cos t).
///
/// Metrics of a comparison (test T against reference R, intensities I = |F|^2):
///   l2_refl   = sqrt(sum w (I_T - I_R)^2 / sum w I_R^2) over the reflection hemisphere,
///               w = sin(theta) (the docs/05 / ADR 0006 verdict metric; the midpoint grid in
///               theta and phi has equal cells, so sin(theta) is the solid-angle weight),
///   l2_off30  = the same restricted to theta >= 30 deg (off-specular part),
///   l2_fwd    = the same over the forward hemisphere (diagnostic only),
///   dP_refl   = relative change of the power scattered into the reflection hemisphere,
///   eps_xz/yz = docs/05 eps_rr = sqrt(mean((I_T - I_R)^2)) / max I_R along the cuts.
#include "specklebem/core/types.hpp"

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace box_validity {

using namespace specklebem;

inline constexpr Real kDeg = constants::pi / 180.0;

// Far-field grids (see the file comment).
inline constexpr int kHemiTheta = 90;
inline constexpr int kHemiPhi = 120;
inline constexpr Real kHemiDTheta = 1.0;  // deg
inline constexpr Real kHemiDPhi = 3.0;    // deg
inline constexpr int kCut = 359;
inline constexpr Real kCutStep = 0.5;  // deg
inline constexpr Real kOffSpecularTheta = 30.0;

using FieldMatrix = Eigen::Matrix<Complex, Eigen::Dynamic, 3>;

/// Directions of a hemisphere grid (kHemiTheta * kHemiPhi rows); z_sign -1: reflection
/// hemisphere, +1: forward hemisphere.
inline Vertices hemisphere_directions(Real z_sign) {
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

/// Directions of the xz-cut (yz = false) or the yz-cut on the reflection side (kCut rows).
inline Vertices cut_directions(bool yz) {
    Vertices d(kCut, 3);
    for (int m = 0; m < kCut; ++m) {
        const Real t = (-89.5 + m * kCutStep) * kDeg;
        d(m, 0) = yz ? 0.0 : std::sin(t);
        d(m, 1) = yz ? std::sin(t) : 0.0;
        d(m, 2) = -std::cos(t);
    }
    return d;
}

/// |F|^2 of rows r * cols + c arranged as a rows x cols matrix.
/// @throws std::invalid_argument if F does not have rows * cols rows.
inline MatrixXr intensity(const FieldMatrix& F, Index rows, Index cols) {
    if (F.rows() != rows * cols)
        throw std::invalid_argument("box_validity::intensity: F has the wrong number of rows");
    MatrixXr I(rows, cols);
    for (Index r = 0; r < rows; ++r)
        for (Index c = 0; c < cols; ++c) I(r, c) = F.row(r * cols + c).squaredNorm();
    return I;
}

/// Far-field intensities of one run.
struct FarField {
    MatrixXr refl;    // kHemiTheta x kHemiPhi
    MatrixXr fwd;     // kHemiTheta x kHemiPhi
    MatrixXr cut_xz;  // kCut x 1
    MatrixXr cut_yz;  // kCut x 1
};

struct Metrics {
    Real l2_refl = 0, l2_off30 = 0, l2_fwd = 0, dP_refl = 0, eps_xz = 0, eps_yz = 0;
};

/// Relative L2 difference sqrt(sum w (t - r)^2 / sum w r^2) over a hemisphere grid (row i at
/// theta = (i + 1/2) kHemiDTheta) with w = sin(theta), only rows with theta >= theta_min_deg.
/// @throws std::invalid_argument for different shapes or a reference that is zero on the rows.
inline Real hemi_l2(const MatrixXr& t, const MatrixXr& r, Real theta_min_deg) {
    if (t.rows() != r.rows() || t.cols() != r.cols())
        throw std::invalid_argument("box_validity::hemi_l2: shapes differ");
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
    if (!(den > 0))
        throw std::invalid_argument("box_validity::hemi_l2: reference is zero");
    return std::sqrt(num / den);
}

/// Power integral sum sin(theta) I dtheta dphi over a hemisphere grid (midpoint rule).
inline Real hemi_power(const MatrixXr& I) {
    Real p = 0;
    for (Index i = 0; i < I.rows(); ++i) {
        const Real w = std::sin((static_cast<Real>(i) + 0.5) * kHemiDTheta * kDeg);
        p += w * I.row(i).sum();
    }
    return p * (kHemiDTheta * kDeg) * (kHemiDPhi * kDeg);
}

/// docs/05 eps_rr = sqrt(mean (t - r)^2) / max r.
/// @throws std::invalid_argument for different shapes or max r <= 0.
inline Real eps_rr(const MatrixXr& t, const MatrixXr& r) {
    if (t.rows() != r.rows() || t.cols() != r.cols() || r.size() == 0)
        throw std::invalid_argument("box_validity::eps_rr: shapes differ");
    if (!(r.maxCoeff() > 0))
        throw std::invalid_argument("box_validity::eps_rr: reference is zero");
    const Real rms = std::sqrt((t - r).squaredNorm() / static_cast<Real>(r.size()));
    return rms / r.maxCoeff();
}

/// All metrics of test t against reference r.
/// @throws std::runtime_error if the grids differ.
inline Metrics compare(const FarField& t, const FarField& r) {
    if (t.refl.rows() != r.refl.rows() || t.refl.cols() != r.refl.cols() ||
        t.fwd.rows() != r.fwd.rows() || t.fwd.cols() != r.fwd.cols() ||
        t.cut_xz.size() != r.cut_xz.size() || t.cut_yz.size() != r.cut_yz.size())
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

// ---------------------------------------------------------------------------------------------
// MLFMM memory guard (WP-V2).

/// Headroom [GB] the guard keeps between the setup peak estimate and the available memory.
inline constexpr Real kGuardHeadroomGb = 10.0;

/// Memory [GB, 1e9 bytes] of the full-GMRES Krylov basis at `max_iter`: (max_iter + 1) vectors of
/// `two_n` complex entries.
/// @throws std::invalid_argument for max_iter < 1 or two_n < 1.
inline Real krylov_basis_gb(int max_iter, Index two_n) {
    if (max_iter < 1 || two_n < 1)
        throw std::invalid_argument("krylov_basis_gb: need max_iter >= 1 and 2N >= 1");
    return static_cast<Real>(max_iter + 1) * static_cast<Real>(two_n) *
           static_cast<Real>(sizeof(Complex)) / 1e9;
}

/// Decision of the MLFMM memory guard of the study executable. Unless `force`, the run is refused
/// when the peak estimate setup_gb + krylov_gb exceeds `max_gb`, or when the setup peak estimate
/// plus kGuardHeadroomGb exceeds `available_gb` (the second test is skipped when the available
/// memory is unknown, available_gb <= 0). Returns normally when the run is allowed.
/// @throws std::invalid_argument for negative or non-finite sizes or max_gb <= 0 (also with force).
/// @throws std::runtime_error when the run is refused.
inline void check_memory_guard(Real setup_gb, Real krylov_gb, Real available_gb, Real max_gb,
                               bool force) {
    if (!std::isfinite(setup_gb) || !std::isfinite(krylov_gb) || !std::isfinite(available_gb) ||
        !std::isfinite(max_gb) || setup_gb < 0 || krylov_gb < 0 || !(max_gb > 0))
        throw std::invalid_argument(
            "check_memory_guard: need finite sizes, setup and Krylov >= 0, max_gb > 0");
    if (force)
        return;
    const Real peak_gb = setup_gb + krylov_gb;
    if (peak_gb > max_gb)
        throw std::runtime_error("MLFMM peak estimate " + std::to_string(peak_gb) +
                                 " GB exceeds --max-gb " + std::to_string(max_gb) +
                                 " (pass --force to run anyway)");
    if (available_gb > 0 && setup_gb + kGuardHeadroomGb > available_gb)
        throw std::runtime_error("MLFMM setup peak estimate " + std::to_string(setup_gb) +
                                 " GB + 10 GB exceeds the available memory " +
                                 std::to_string(available_gb) + " GB (wait or --force)");
}

// ---------------------------------------------------------------------------------------------
// Command line.

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
    std::optional<Real> eps_imag;  // diagnostic: override Im(eps_r) of the object
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
    bool allow_unconverged = false;
    // WP-V2 (MLFMM sizes).
    std::string compression = "dense";  ///< "dense" (WP-V1) or "mlfmm"
    Real digits = 3.0;                  ///< MLFMM accuracy digits d0
    /// MlfmmParams::leaf_radius_quantile (1 = global leaf rule; < 1 the local rule, d0 <= 3 only).
    Real leaf_quantile = 1.0;
    /// Jacobi preconditioner; empty: none for dense (the WP-V1 runs), the recommendation of
    /// formulation::recommend for mlfmm (Ag: left Jacobi, Si: none).
    std::optional<bool> jacobi;
    bool estimate_only = false;  ///< mlfmm: print the memory estimate and stop (dense: mesh only)
    bool force = false;          ///< mlfmm: skip the memory guard
    bool parts = true;           ///< PARTS diagnostic (six extra hemisphere far fields)
    std::string history;         ///< text file for the GMRES residual history (one per line)
};

/// Parses the arguments (without the program name). See benchmarks/box_validity.cpp.
/// @throws std::invalid_argument for unknown options, missing or invalid values.
inline Options parse_options(const std::vector<std::string>& args) {
    Options o;
    const std::size_t n = args.size();
    for (std::size_t i = 0; i < n; ++i) {
        const std::string& a = args[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= n)
                throw std::invalid_argument("missing value after " + a);
            return args[++i];
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
        } else if (a == "--eps-imag") {
            o.eps_imag = std::stod(value());
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
            while (i + 1 < n && args[i + 1].rfind("--", 0) != 0) o.compare.push_back(args[++i]);
        } else if (a == "--mesh-only") {
            o.mesh_only = true;
        } else if (a == "--allow-unconverged") {
            o.allow_unconverged = true;
        } else if (a == "--compression") {
            o.compression = value();
        } else if (a == "--d0") {
            o.digits = std::stod(value());
        } else if (a == "--leaf-quantile") {
            o.leaf_quantile = std::stod(value());
        } else if (a == "--jacobi") {
            const std::string v = value();
            if (v == "auto")
                o.jacobi.reset();
            else if (v == "0" || v == "1")
                o.jacobi = v == "1";
            else
                throw std::invalid_argument("--jacobi must be auto, 0 or 1");
        } else if (a == "--estimate-only") {
            o.estimate_only = true;
        } else if (a == "--force") {
            o.force = true;
        } else if (a == "--parts") {
            const std::string v = value();
            if (v != "0" && v != "1")
                throw std::invalid_argument("--parts must be 0 or 1");
            o.parts = v == "1";
        } else if (a == "--history") {
            o.history = value();
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
    if (o.eps_imag && !(*o.eps_imag < 0))
        throw std::invalid_argument("--eps-imag must be negative (passive, exp(+jwt))");
    if (o.beam != "spectrum" && o.beam != "paraxial")
        throw std::invalid_argument("--beam must be spectrum or paraxial");
    if (o.compression != "dense" && o.compression != "mlfmm")
        throw std::invalid_argument("--compression must be dense or mlfmm");
    if (!(o.digits > 0) || o.digits > 8)
        throw std::invalid_argument("--d0 must be in (0, 8]");
    if (!(o.leaf_quantile > 0) || o.leaf_quantile > 1)
        throw std::invalid_argument("--leaf-quantile must be in (0, 1]");
    return o;
}

}  // namespace box_validity
