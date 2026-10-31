// Recycling study (WP-KR1, ADR 0013 decision 1): one operator per process, the excitation
// sequence 0 deg s -> theta2 s -> 0 deg p (default theta2 = 30 deg: the b1 angle pair and the
// polarisation pair of the speckle studies) on the rough PML route of the preconditioner study
// (pml_rough_ag::u_case(): rough Ag, sigma 50 nm, Lc 500 nm, seed 1, h = lambda / 4, W_pml = 2
// lambda, w0 = lambda; --patch L --map-patch L for 16 um, --material si for the Si control),
// compression "mlfmm", d0 = 3, the default preconditioner of the route (the ILUT of the near
// blocks, WP-SC4), GMRES tolerance 1e-7. Per case:
//   1. SETUP: construction and assemble() of the Simulation (operator + right-hand side + ILUT
//      setup) = the fresh path of one excitation;
//   2. the first solve (x0 = 0) with keep_hessenberg / keep_basis; SPECTRUM: the harmonic Ritz
//      values theta of the ILUT-preconditioned operator Op = M^-1 Z from its Hessenberg (as WP-A2
//      E3, tests/support/krylov_spectrum.hpp): the count with |theta| < 0.1 and < 0.3, the three
//      smallest |theta|, sigma_min(H_bar) and sigma_max / sigma_min; with --parent the same for
//      the block-diagonal preconditioner one level above the leaves (same right-hand side);
//   3. DEFLATION: the harmonic Ritz vectors of the k smallest |theta| (U = V_k Y), C~ = Op U by k
//      explicit applications of Z and M^-1, C = Q of the thin QR of C~, U <- U R^-1 (Op U = C);
//   4. for excitations 2 and 3: Simulation::set_excitation (right-hand side only; timed), then
//      the solves with x0 = previous (the currents of the last solve, i.e. of the previous
//      excitation; Simulation::solve with InitialGuess::Previous), x0 = 0, and the deflation
//      probe for each k (a test-grade GCRO step: f = M^-1 b, r = (I - C C^H) f, GMRES on the
//      projected operator (I - C C^H) Op with the tolerance tol |f| / |r| so that it stops at the
//      same preconditioned residual |M^-1 (b - Z x)| / |M^-1 b| <= tol as the plain solve, then x
//      = y + U C^H (f - Op y)); R = P_refl / P_inc of the windowed far field and dR, L2 against
//      the x0 = 0 solve of the same excitation;
//   5. with --fresh: a fresh Simulation per excitation 2 and 3 (construct + assemble + solve),
//      timed, and its R / L2 against the reused x0 = 0 solve.
// Output: SETUP, SPECTRUM, DEFLATION and ROW lines (stdout, also appended to --summary FILE), the
// Simulation report. Record: benchmarks/results/recycling.md (+ recycling_rows.txt).
//
// Usage: specklebem_recycling_study [--tag NAME] [--material ag|si] [--patch L] [--map-patch L]
//            [--d0 3] [--tol 1e-7] [--max-iter 5000] [--theta2-deg 30] [--ks 10,20,40]
//            [--dark-band-tolerance 1e-4] [--parent] [--fresh] [--estimate-only] [--n-theta 90]
//            [--n-phi 180] [--summary FILE]
// --dark-band-tolerance: PmlTermination::dark_band_tolerance of the case (all excitations);
// --estimate-only prints the DARK line of each excitation (no assembly). At U the 30 deg beam
// with w0 = lambda has 1.4e-4 (w0 cos(theta) < lambda_1: grazing waves of the cut spectrum,
// WP-PML3b), above the 1e-4 target; the record runs U with 2e-4.
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/octree.hpp"
#include "specklebem/compression/mlfmm/pml_ring.hpp"
#include "specklebem/core/path.hpp"
#include "specklebem/excitation/angular_spectrum_beam.hpp"
#include "specklebem/simulation.hpp"
#include "specklebem/solver/gmres.hpp"
#include "specklebem/solver/incomplete_lu_preconditioner.hpp"
#include "specklebem/solver/preconditioner.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "krylov_spectrum.hpp"
#include "pml_rough_ag_support.hpp"
#include "system_memory.hpp"

namespace {

using namespace specklebem;

struct Options {
    pml_rough_ag::RoughCase c = pml_rough_ag::u_case();
    Real digits = 3.0;
    Real tolerance = 1e-7;
    int max_iter = 5000;
    Real theta2_deg = 30.0;
    std::vector<int> ks = {10, 20, 40};
    bool parent = false;
    bool fresh = false;
    bool estimate_only = false;
    std::string summary;
};

std::vector<int> split_ints(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty())
            out.push_back(std::stoi(item));
    return out;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::invalid_argument("missing value after " + a);
            return argv[++i];
        };
        const auto real = [&]() { return std::stod(value()); };
        if (a == "--tag") {
            o.c.tag = value();
        } else if (a == "--material") {
            o.c.material = value();
        } else if (a == "--patch") {
            o.c.patch = real();
        } else if (a == "--map-patch") {
            o.c.map_patch = real();
        } else if (a == "--d0") {
            o.digits = real();
        } else if (a == "--tol") {
            o.tolerance = real();
        } else if (a == "--max-iter") {
            o.max_iter = std::stoi(value());
        } else if (a == "--dark-band-tolerance") {
            o.c.dark_band_tolerance = real();
        } else if (a == "--theta2-deg") {
            o.theta2_deg = real();
        } else if (a == "--ks") {
            o.ks = split_ints(value());
        } else if (a == "--parent") {
            o.parent = true;
        } else if (a == "--fresh") {
            o.fresh = true;
        } else if (a == "--estimate-only") {
            o.estimate_only = true;
        } else if (a == "--n-theta") {
            o.c.n_theta = std::stoi(value());
        } else if (a == "--n-phi") {
            o.c.n_phi = std::stoi(value());
        } else if (a == "--summary") {
            o.summary = value();
        } else {
            throw std::invalid_argument("unknown option " + a);
        }
    }
    if (o.ks.empty())
        throw std::invalid_argument("--ks: no deflation dimension");
    for (const int k : o.ks)
        if (k < 1)
            throw std::invalid_argument("--ks: dimensions must be >= 1");
    std::sort(o.ks.begin(), o.ks.end());
    if (!(o.tolerance > 0.0) || o.max_iter < 1)
        throw std::invalid_argument("--tol must be > 0 and --max-iter >= 1");
    return o;
}

void emit(const std::string& path, const std::string& line) {
    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
    if (path.empty())
        return;
    std::ofstream f(core::path_from_utf8(path), std::ios::app | std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    f << line << '\n';
}

Real seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
}

/// printf into a std::string (lines up to 2047 characters).
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 1, 2)))
#endif
std::string strf(const char* format, ...) {
    char buf[2048];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buf, sizeof(buf), format, args);
    va_end(args);
    return buf;
}

// ---------------------------------------------------------------------------------------------
// Excitations
// ---------------------------------------------------------------------------------------------

struct ExcitationSpec {
    std::string name;  ///< e.g. "0s", "30s", "0p"
    Real theta_deg = 0.0;
    excitation::Polarization pol = excitation::Polarization::S;
};

std::vector<ExcitationSpec> sequence(Real theta2_deg) {
    char t2[32];
    std::snprintf(t2, sizeof(t2), "%gs", theta2_deg);
    return {{"0s", 0.0, excitation::Polarization::S},
            {t2, theta2_deg, excitation::Polarization::S},
            {"0p", 0.0, excitation::Polarization::P}};
}

/// The beam of pml_rough_ag::make_beam (rigorous angular-spectrum beam, focus at the origin,
/// waist c.waist, controlled ball 1.02 sqrt(2) pml_start) at the incidence angle and polarisation
/// of `e` (rotation R_y(theta) of +z, docs/06; at 0 deg p = E along x, s = E along y).
std::shared_ptr<excitation::AngularSpectrumBeam> make_beam(const pml_rough_ag::RoughCase& c,
                                                           Real pml_start,
                                                           const ExcitationSpec& e) {
    excitation::AngularSpectrumBeam::Params bp;
    bp.wavelength = pml_rough_ag::kLambda;
    bp.waist_radius = c.waist;
    bp.focus = Vec3::Zero();
    bp.incidence_angle = e.theta_deg * pml_rough_ag::kDeg;
    bp.polarization = e.pol;
    bp.region_radius = 1.02 * std::sqrt(2.0) * pml_start;
    return std::make_shared<excitation::AngularSpectrumBeam>(bp, material::vacuum());
}

// ---------------------------------------------------------------------------------------------
// Spectrum and deflation
// ---------------------------------------------------------------------------------------------

/// "SPECTRUM | ..." of the Hessenberg of one solve.
std::string spectrum_line(const std::string& head, const std::string& precond,
                          const solver::GmresResult& g) {
    namespace ks = krylov_spectrum;
    if (g.hessenberg.cols() < 1)
        return head + "precond " + precond + " | no Hessenberg";
    const VectorXc theta = ks::harmonic_ritz_values(g.hessenberg);
    const std::vector<Complex> sorted = ks::sorted_by_modulus(theta);
    int below01 = 0;
    int below03 = 0;
    for (const Complex& t : sorted) {
        below01 += std::abs(t) < 0.1 ? 1 : 0;
        below03 += std::abs(t) < 0.3 ? 1 : 0;
    }
    const VectorXr sv = ks::singular_values(g.hessenberg);
    std::string smallest;
    for (std::size_t i = 0; i < std::min<std::size_t>(3, sorted.size()); ++i)
        smallest += strf("%s%.3e (%+.3f%+.3fj)", i == 0 ? "" : ", ", std::abs(sorted[i]),
                         sorted[i].real(), sorted[i].imag());
    const Real largest = std::abs(sorted.back());
    return head + strf(
                      "precond %s | its %d | k %lld | |theta| < 0.1: %d | < 0.3: %d | smallest "
                      "|theta| %s | largest |theta| %.3f | sigma_min(Hbar) %.3e | sigma_max %.3f | "
                      "sigma_max/sigma_min %.3e",
                      precond.c_str(), g.iterations, static_cast<long long>(theta.size()), below01,
                      below03, smallest.c_str(), largest, sv(sv.size() - 1), sv(0),
                      sv(0) / sv(sv.size() - 1));
}

/// Recycle space of the deflation probe: U (n x k) and C = Op U orthonormal (n x k), k the
/// largest requested dimension that the first solve's Krylov space provides.
struct Deflation {
    MatrixXc U;
    MatrixXc C;
    VectorXc theta;  ///< the harmonic Ritz values of the columns, increasing |theta|
    int k = 0;
    int krylov = 0;    ///< Krylov dimension of the first solve
    Real cond_r = 0;   ///< condition number of R (C~ = C R)
    Real build_s = 0;  ///< wall time: eigenvectors, k applications of Op, QR
};

/// Op v = M^-1 (Z v).
void apply_op(const op::LinearOperator& Z, const solver::Preconditioner& M, const VectorXc& v,
              VectorXc& out) {
    VectorXc t;
    Z.apply(v, t);
    M.apply(t, out);
}

Deflation make_deflation(const solver::GmresResult& g, const op::LinearOperator& Z,
                         const solver::Preconditioner& M, int kmax) {
    const auto t0 = std::chrono::steady_clock::now();
    const MatrixXc& H = g.hessenberg;
    const Index m = H.cols();
    if (m < 1 || H.rows() != m + 1 || g.basis.cols() < m)
        throw std::logic_error("recycling_study: the first solve kept no Hessenberg / basis");
    // Harmonic Ritz pairs: G y = theta y, G = H_m + |h_{m+1,m}|^2 H_m^-H e_m e_m^T.
    const MatrixXc Hm = H.topRows(m);
    const Eigen::PartialPivLU<MatrixXc> lu(Hm.adjoint());
    const VectorXc f = lu.solve(VectorXc::Unit(m, m - 1));
    MatrixXc G = Hm;
    G.col(m - 1) += std::norm(H(m, m - 1)) * f;
    const Eigen::ComplexEigenSolver<MatrixXc> es(G, true);
    if (es.info() != Eigen::Success)
        throw std::runtime_error("recycling_study: harmonic Ritz eigenproblem failed");
    std::vector<Index> order(static_cast<std::size_t>(m));
    std::iota(order.begin(), order.end(), Index{0});
    const VectorXc ev = es.eigenvalues();
    std::stable_sort(order.begin(), order.end(),
                     [&](Index a, Index b) { return std::abs(ev(a)) < std::abs(ev(b)); });
    Deflation d;
    d.krylov = static_cast<int>(m);
    d.k = static_cast<int>(std::min<Index>(kmax, m));
    const auto k = static_cast<Index>(d.k);
    MatrixXc Y(m, k);
    d.theta.resize(k);
    for (Index j = 0; j < k; ++j) {
        Y.col(j) = es.eigenvectors().col(order[static_cast<std::size_t>(j)]);
        d.theta(j) = ev(order[static_cast<std::size_t>(j)]);
    }
    d.U = g.basis.leftCols(m) * Y;
    const Index n = d.U.rows();
    MatrixXc Ct(n, k);
    VectorXc w;
    for (Index j = 0; j < k; ++j) {
        apply_op(Z, M, d.U.col(j), w);
        Ct.col(j) = w;
    }
    const Eigen::HouseholderQR<MatrixXc> qr(Ct);
    d.C = qr.householderQ() * MatrixXc::Identity(n, k);
    const MatrixXc R = qr.matrixQR().topLeftCorner(k, k).triangularView<Eigen::Upper>();
    const Eigen::JacobiSVD<MatrixXc> svd(R);
    d.cond_r = svd.singularValues().maxCoeff() / svd.singularValues().minCoeff();
    // U <- U R^-1, so that Op U = C.
    const MatrixXc Rinv = R.triangularView<Eigen::Upper>().solve(MatrixXc::Identity(k, k).eval());
    d.U = d.U * Rinv;
    d.build_s = seconds_since(t0);
    return d;
}

/// (I - C C^H) M^-1 Z with the leading k columns of C: the operator of the deflated GMRES.
class ProjectedOperator final : public op::LinearOperator {
public:
    ProjectedOperator(const op::LinearOperator& Z, const solver::Preconditioner& M,
                      const MatrixXc& C, Index k)
        : Z_(Z), M_(M), C_(C), k_(k) {}
    [[nodiscard]] Index rows() const override { return Z_.rows(); }
    [[nodiscard]] Index cols() const override { return Z_.cols(); }
    void apply(const VectorXc& x, VectorXc& y) const override {
        apply_op(Z_, M_, x, y);
        const VectorXc h = C_.leftCols(k_).adjoint() * y;
        y.noalias() -= C_.leftCols(k_) * h;
    }
    [[nodiscard]] std::string describe() const override {
        return "(I - C C^H) M^-1 Z, k = " + std::to_string(k_);
    }
    [[nodiscard]] std::size_t memory_bytes() const override {
        return static_cast<std::size_t>(C_.rows() * k_) * sizeof(Complex);
    }

private:
    const op::LinearOperator& Z_;
    const solver::Preconditioner& M_;
    const MatrixXc& C_;
    Index k_;
};

struct ProbeResult {
    VectorXc x;
    int iterations = 0;
    bool converged = false;
    Real true_res = 0;  ///< |b - Z x| / |b|
    Real prec_res = 0;  ///< |M^-1 (b - Z x)| / |M^-1 b|
    Real r0 = 0;        ///< |(I - C C^H) f| / |f|: the residual after the projection
    Real gmres_s = 0;   ///< wall time of the probe (projection, GMRES, correction)
};

/// The deflation probe with the leading k columns of the recycle space (see the file header).
ProbeResult probe(const Deflation& d, Index k, const op::LinearOperator& Z,
                  const solver::Preconditioner& M, const VectorXc& b,
                  const solver::GmresParams& params) {
    const auto t0 = std::chrono::steady_clock::now();
    ProbeResult out;
    VectorXc f;
    M.apply(b, f);
    const Real f_norm = f.norm();
    const auto C = d.C.leftCols(k);
    const auto U = d.U.leftCols(k);
    const VectorXc r = f - C * (C.adjoint() * f);
    const Real r_norm = r.norm();
    out.r0 = r_norm / f_norm;
    const ProjectedOperator P(Z, M, d.C, k);
    solver::GmresParams p = params;
    p.keep_basis = false;
    p.keep_hessenberg = false;
    // Stop at |f - Op x| / |f| <= tol: the projected residual equals f - Op x (see the header).
    p.tolerance = params.tolerance * f_norm / r_norm;
    const solver::GmresResult g = solver::gmres(P, r, solver::IdentityPreconditioner(), p);
    out.iterations = g.iterations;
    out.converged = g.converged;
    VectorXc opy;
    apply_op(Z, M, g.x, opy);
    out.x = g.x + U * (C.adjoint() * (f - opy));
    out.gmres_s = seconds_since(t0);
    VectorXc zx;
    Z.apply(out.x, zx);
    const VectorXc res = b - zx;
    out.true_res = res.norm() / b.norm();
    VectorXc mres;
    M.apply(res, mres);
    out.prec_res = mres.norm() / f_norm;
    return out;
}

// ---------------------------------------------------------------------------------------------
// The study
// ---------------------------------------------------------------------------------------------

struct Solved {
    pml_rough_ag::FarField far;
    Real R = 0;
};

Solved evaluate(const Simulation& sim, const VectorXc& x, const pml_rough_ag::RoughCase& c) {
    const post::SurfaceSolution s{&sim.problem(), x};
    Solved out;
    out.far =
        pml_rough_ag::evaluate_far_field(sim, s, c, *sim.far_field_window(), sim.incident_power());
    out.R = out.far.reflectance();
    return out;
}

/// " | R ... | dR vs ref ... | l2 ..." against a reference far field (none: R only).
std::string compare_tail(const Solved& t, const Solved* ref, const char* ref_name) {
    if (ref == nullptr)
        return strf(" | R %.10f", t.R);
    const pml_rough_ag::Metrics m = pml_rough_ag::compare(t.far, ref->far);
    return strf(" | R %.10f | dR vs %s %+.3e (abs %+.3e) | l2 %.3e", t.R, ref_name, m.dR,
                t.R - ref->R, m.l2);
}

std::string gmres_segment(const solver::GmresResult& g) {
    return strf("its %d | converged %s | r0 %.3e | true res %.2e | GMRES %.2f s (orth %.2f s)",
                g.iterations, g.converged ? "yes" : "NO", g.residual_history.front(),
                g.true_relative_residual, g.wall_seconds, g.orthogonalization_seconds);
}

void run(const Options& o) {
    const pml_rough_ag::RoughCase& c = o.c;
    const pml_rough_ag::Estimate est = pml_rough_ag::estimate(c);
    const Real pml_start = est.size.half[2];
    const std::vector<ExcitationSpec> seq = sequence(o.theta2_deg);
    const material::Material object = pml_rough_ag::object_material(c.material);
    const Real fraction = beam_waist_fraction(object, pml_rough_ag::kLambda, c.mesh_size);
    std::vector<std::shared_ptr<excitation::AngularSpectrumBeam>> beams;
    for (const ExcitationSpec& e : seq) {
        // The beam-waist rule of rough patches at oblique incidence (ADR 0006 amendments).
        check_beam_waist(c.patch, c.waist,
                         {.incidence_angle = e.theta_deg * pml_rough_ag::kDeg,
                          .allow_wide = false,
                          .max_fraction = fraction});
        beams.push_back(make_beam(c, pml_start, e));
    }
    SimulationConfig cfg = pml_rough_ag::make_config(c);
    cfg.compression = "mlfmm";
    cfg.solver = SolverKind::Gmres;
    cfg.mlfmm.accuracy_digits = o.digits;
    cfg.gmres.tolerance = o.tolerance;
    cfg.gmres.max_iter = o.max_iter;
    cfg.gmres.restart = 0;
    // No preconditioner setting: the route's default (the ILUT of the near blocks, WP-SC4).
    const geometry::OpenSurfaceMesh open = pml_rough_ag::make_open_mesh(c);
    const std::string head =
        c.tag + " | " + c.material + " | 2N " + std::to_string(est.size.unknowns) + " | ";

    if (o.estimate_only) {
        // Dark band of each excitation (constructed with a relaxed tolerance, then
        // set_excitation; no assembly).
        pml_rough_ag::RoughCase relaxed = c;
        relaxed.dark_band_tolerance = 1e6;
        Simulation sim(open, pml_rough_ag::make_termination(relaxed), beams[0], cfg);
        for (std::size_t i = 0; i < seq.size(); ++i) {
            sim.set_excitation(beams[i]);
            emit(
                o.summary,
                "DARK | recycling | " + head +
                    strf("exc %s | dark band ratio %.3e (tolerance %.1e) | waist %.3g m | w0 "
                         "limit %.3g m (fraction %.4f)",
                         seq[i].name.c_str(), sim.dark_band_ratio(), c.dark_band_tolerance, c.waist,
                         fraction * c.patch * std::cos(seq[i].theta_deg * pml_rough_ag::kDeg),
                         fraction));
        }
        return;
    }

    const auto t_construct = std::chrono::steady_clock::now();
    Simulation sim(open, pml_rough_ag::make_termination(c), beams[0], cfg);
    const Real construct_s = seconds_since(t_construct);
    const auto t_assemble = std::chrono::steady_clock::now();
    sim.assemble();
    const Real assemble_s = seconds_since(t_assemble);
    const auto* Z = dynamic_cast<const mlfmm::MlfmmOperator*>(sim.system_operator().get());
    const solver::IncompleteLuPreconditioner* M = sim.incomplete_lu_preconditioner();
    if (Z == nullptr || Z->ring_operator() == nullptr || M == nullptr)
        throw std::logic_error(
            "recycling_study: expected an MLFMM operator with a PML ring and the ILUT default");
    const Index n2 = Z->rows();
    std::printf("%s\n", M->describe().c_str());
    emit(o.summary,
         "SETUP | recycling | " + head +
             strf("rms %.3g | L %.4g | Lgen %.4g | h %.4g | w0 %.3g | d0 %.1f | tol %.1e | "
                  "dark tol %.1e | levels %d | leaf %.3f lambda | op %.1f MB | ILUT fill %.3g drop "
                  "%.1e P %.1f MB "
                  "(setup A %.2f s, D %.2f s) | fresh path: construct %.2f s + assemble %.1f s "
                  "(operator + rhs + ILUT) = %.1f s | sequence %s -> %s -> %s",
                  c.rms, c.patch, c.map_patch, c.mesh_size, c.waist, o.digits, o.tolerance,
                  c.dark_band_tolerance, Z->octree().levels(),
                  Z->octree().box_size(Z->octree().leaf_level()) / pml_rough_ag::kLambda,
                  static_cast<Real>(Z->memory_bytes()) / 1e6, M->params().fill_factor,
                  M->params().drop_tol, static_cast<Real>(M->memory_bytes()) / 1e6,
                  M->setup_seconds_a(), M->setup_seconds_d(), construct_s, assemble_s,
                  construct_s + assemble_s, seq[0].name.c_str(), seq[1].name.c_str(),
                  seq[2].name.c_str()));
    const std::string row = "ROW | recycling | " + head;

    // 1. First solve, x0 = 0, with the Hessenberg and the basis.
    solver::GmresParams p1 = cfg.gmres;
    p1.keep_hessenberg = true;
    p1.keep_basis = true;
    solver::GmresResult g1 = sim.solve(p1);
    const Solved s1 = evaluate(sim, sim.solution().currents, c);
    emit(o.summary, row +
                        strf("exc %s | method zero | dark %.2e | ", seq[0].name.c_str(),
                             sim.dark_band_ratio()) +
                        gmres_segment(g1) + compare_tail(s1, nullptr, ""));
    emit(o.summary, "SPECTRUM | recycling | " + head + strf("exc %s | ", seq[0].name.c_str()) +
                        spectrum_line("", "ilut-default", g1));
    if (o.parent) {
        // The block-diagonal preconditioner one level up on the same operator and right-hand
        // side (the WP-PC1 default), for a comparison of the spectra.
        const auto P = make_block_preconditioner(*Z, sim.problem(), cfg.mlfmm, 1);
        solver::GmresParams pp = cfg.gmres;
        pp.keep_hessenberg = true;
        const solver::GmresResult gp = solver::gmres(*Z, sim.rhs(), *P, pp);
        emit(o.summary, "SPECTRUM | recycling | " + head + strf("exc %s | ", seq[0].name.c_str()) +
                            spectrum_line("", "parent", gp));
    }
    const Deflation d = make_deflation(g1, *Z, *M, o.ks.back());
    g1.basis = MatrixXc();  // the recycle space is built; release the Krylov basis
    std::string thetas;
    for (Index j = 0; j < d.theta.size(); ++j)
        thetas += strf("%s%.3e", j == 0 ? "" : " ", std::abs(d.theta(j)));
    emit(o.summary, "DEFLATION | recycling | " + head +
                        strf("k %d of Krylov %d | build %.2f s (eigenvectors, k Op applications, "
                             "QR) | cond(R) %.3e | U, C %.3f GB | |theta| of the columns: %s",
                             d.k, d.krylov, d.build_s, d.cond_r,
                             32.0 * static_cast<Real>(n2) * d.k / 1e9, thetas.c_str()));

    // 2. Excitations 2 and 3 on the same operator.
    std::vector<Solved> zero_solves(seq.size());
    zero_solves[0] = s1;
    for (std::size_t i = 1; i < seq.size(); ++i) {
        const char* name = seq[i].name.c_str();
        const auto t_swap = std::chrono::steady_clock::now();
        sim.set_excitation(beams[i]);
        const Real swap_s = seconds_since(t_swap);
        emit(o.summary, row + strf("exc %s | method swap | set_excitation %.3f s (rhs only) | "
                                   "fresh path %.1f s | ratio %.0f | dark %.2e",
                                   name, swap_s, construct_s + assemble_s,
                                   (construct_s + assemble_s) / swap_s, sim.dark_band_ratio()));
        // x0 = the currents of the previous excitation.
        const solver::GmresResult gp =
            sim.solve(cfg.gmres, SolveOptions{.initial_guess = InitialGuess::Previous});
        const Solved sp = evaluate(sim, sim.solution().currents, c);
        // x0 = 0.
        const solver::GmresResult gz = sim.solve(cfg.gmres);
        const VectorXc xz = sim.solution().currents;
        const Solved sz = evaluate(sim, xz, c);
        zero_solves[i] = sz;
        emit(o.summary, row + strf("exc %s | method zero | ", name) + gmres_segment(gz) +
                            compare_tail(sz, nullptr, ""));
        emit(o.summary, row + strf("exc %s | method previous | ", name) + gmres_segment(gp) +
                            compare_tail(sp, &sz, "zero") +
                            strf(" | its ratio zero/previous %.2f",
                                 static_cast<Real>(gz.iterations) / std::max(1, gp.iterations)));
        for (const int kreq : o.ks) {
            const Index k = std::min<Index>(kreq, d.k);
            const ProbeResult pr = probe(d, k, *Z, *M, sim.rhs(), cfg.gmres);
            const Solved sd = evaluate(sim, pr.x, c);
            emit(
                o.summary,
                row +
                    strf("exc %s | method probe k %lld (requested %d) | its %d | converged %s "
                         "| r0 %.3e | prec res %.2e | true res %.2e | probe %.2f s",
                         name, static_cast<long long>(k), kreq, pr.iterations,
                         pr.converged ? "yes" : "NO", pr.r0, pr.prec_res, pr.true_res, pr.gmres_s) +
                    compare_tail(sd, &sz, "zero") +
                    strf(" | its ratio zero/probe %.2f",
                         static_cast<Real>(gz.iterations) / std::max(1, pr.iterations)));
        }
    }
    std::printf("%s\n", sim.report().c_str());

    // 3. Fresh Simulations of excitations 2 and 3 (timed), against the reused x0 = 0 solves.
    if (o.fresh) {
        for (std::size_t i = 1; i < seq.size(); ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            Simulation fresh(open, pml_rough_ag::make_termination(c), beams[i], cfg);
            fresh.assemble();
            const Real setup_s = seconds_since(t0);
            const solver::GmresResult gf = fresh.solve();
            const Solved sf = evaluate(fresh, fresh.solution().currents, c);
            emit(o.summary, row +
                                strf("exc %s | method fresh | construct + assemble %.1f s | ",
                                     seq[i].name.c_str(), setup_s) +
                                gmres_segment(gf) +
                                compare_tail(sf, &zero_solves[i], "reused zero") +
                                strf(" | peak %.2f GB", system_memory::peak_rss_bytes() / 1e9));
        }
    }
    emit(o.summary,
         "END | recycling | " + head + strf("peak %.2f GB", system_memory::peak_rss_bytes() / 1e9));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        run(parse(argc, argv));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "recycling_study: %s\n", e.what());
        return 1;
    }
}
