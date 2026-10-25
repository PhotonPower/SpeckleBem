// Preconditioner study (WP-PC1, ADR 0011 decision 3, the gate of the block-diagonal default):
// one case per process, assembled once through Simulation with compression "mlfmm" (no
// preconditioner data), then solved with each requested preconditioner on the same operator and
// right-hand side (solver::gmres with the config's GMRES parameters):
//   jacobi  the diagonal of Z (MlfmmOperator::diagonal(), as Simulation's Jacobi path),
//   none    no preconditioner,
//   leaf    make_block_preconditioner(levels_above_leaf = 0): LU of the near-field block of
//           every octree leaf over its J and M unknowns,
//   parent  make_block_preconditioner(levels_above_leaf = 1): the same one level up.
// Cases:
//   --case pml  the rough PML route of WP-PML5b (pml_rough_ag::u_case(): rough Ag, sigma 50 nm,
//               Lc 500 nm, seed 1, h = lambda/4, W_pml = 2 lambda, w0 = lambda; --patch L
//               --map-patch L for the L = 16 / 24 um sizes, --material si for the Si control),
//               d0 = 3, GMRES tolerance 1e-7 (as WP-PML5b); per variant R = P_refl / P_inc of the
//               windowed far field and its difference (dR, L2 of pml_rough_ag::compare) to the
//               first variant;
//   --case mie  the Ag sphere of ag_sphere_mlfmm_support.hpp (vacuum, 500 nm, k = +z, E along x;
//               --diameter, --subdiv, icosphere), d0 = 3, tolerance 1e-6; per variant eps_rr
//               against Mie (xz / yz planes, 1 degree) and the RCS difference to the first variant.
// Output: SETUP and one ROW line per variant (stdout, also appended to --summary FILE), the
// preconditioner's describe() and the Simulation report. Record:
// benchmarks/results/preconditioner.md (+ preconditioner_rows.txt).
//
// Usage: specklebem_preconditioner_study [--case pml|mie] [--tag NAME] [--material ag|si]
//            [--patch L] [--map-patch L] [--d0 3] [--tol T] [--max-iter 5000]
//            [--precond jacobi,leaf,parent] [--diameter D] [--subdiv N] [--n-theta 90]
//            [--n-phi 180] [--summary FILE]
// --case must come first (it resets the case).
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/pml_ring.hpp"
#include "specklebem/core/path.hpp"
#include "specklebem/reference/mie.hpp"
#include "specklebem/simulation.hpp"
#include "specklebem/solver/gmres.hpp"
#include "specklebem/solver/preconditioner.hpp"

#include <chrono>
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ag_sphere_mlfmm_support.hpp"
#include "pml_rough_ag_support.hpp"
#include "system_memory.hpp"

namespace {

using namespace specklebem;

struct Options {
    std::string kind = "pml";
    pml_rough_ag::RoughCase c = pml_rough_ag::u_case();
    Real digits = 3.0;
    std::optional<Real> tolerance;  ///< default 1e-7 (pml) / 1e-6 (mie)
    int max_iter = 5000;
    std::vector<std::string> variants = {"jacobi", "leaf", "parent"};
    Real diameter = 1e-6;
    int subdivisions = 4;
    std::string summary;
};

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty())
            out.push_back(item);
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
        if (a == "--case" && i != 1)
            throw std::invalid_argument("--case must be the first option (it resets the case)");
        if (a == "--case") {
            o.kind = value();
            if (o.kind != "pml" && o.kind != "mie")
                throw std::invalid_argument("unknown --case " + o.kind);
            o.c = pml_rough_ag::u_case();
        } else if (a == "--tag") {
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
        } else if (a == "--precond") {
            o.variants = split(value());
        } else if (a == "--diameter") {
            o.diameter = real();
        } else if (a == "--subdiv") {
            o.subdivisions = std::stoi(value());
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
    for (const std::string& v : o.variants) {
        if (v != "jacobi" && v != "none" && v != "leaf" && v != "parent")
            throw std::invalid_argument("unknown --precond variant " + v);
    }
    if (o.variants.empty())
        throw std::invalid_argument("--precond: no variant");
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

/// A preconditioner of one variant on the assembled Simulation, with its setup cost.
struct Built {
    std::unique_ptr<solver::Preconditioner> M;
    const solver::BlockDiagonalPreconditioner* block = nullptr;
    Real setup_s = 0;
    std::size_t bytes = 0;
};

Built build(const std::string& variant, const Simulation& sim, const mlfmm::MlfmmOperator& Z) {
    Built b;
    const auto t0 = std::chrono::steady_clock::now();
    if (variant == "jacobi") {
        b.M = std::make_unique<solver::DiagonalPreconditioner>(Z.diagonal());
        b.bytes = static_cast<std::size_t>(Z.rows()) * sizeof(Complex);
    } else if (variant == "none") {
        b.M = std::make_unique<solver::IdentityPreconditioner>();
    } else {
        auto P = make_block_preconditioner(Z, sim.problem(), sim.config().mlfmm,
                                           variant == "leaf" ? 0 : 1);
        b.block = P.get();
        b.bytes = P->memory_bytes();
        b.M = std::move(P);
    }
    b.setup_s = seconds_since(t0);
    return b;
}

/// Common part of a ROW line: the variant, its preconditioner and the GMRES result.
std::string solve_segment(const std::string& variant, const Built& b, const solver::GmresResult& g,
                          Index n2, Real matvec_s) {
    char buf[700];
    const solver::BlockDiagonalPreconditioner* P = b.block;
    std::snprintf(
        buf, sizeof(buf),
        "variant %s | blocks %lld (n %lld / %.1f / %lld) | P %.1f MB | setup %.2f s (extract "
        "%.2f, LU %.2f) | rcond %.2e | its %d | converged %s | true res %.2e | GMRES %.1f s "
        "(orth %.1f s) | matvec %.3f s | basis %.2f GB",
        variant.c_str(), static_cast<long long>(P ? P->num_blocks() : 0),
        static_cast<long long>(P ? P->min_block_size() : 0),
        P ? static_cast<Real>(n2) / static_cast<Real>(P->num_blocks()) : 0.0,
        static_cast<long long>(P ? P->max_block_size() : 0), static_cast<Real>(b.bytes) / 1e6,
        b.setup_s, P ? P->extraction_seconds() : 0.0, P ? P->factorization_seconds() : 0.0,
        P ? P->min_reciprocal_condition() : 0.0, g.iterations, g.converged ? "yes" : "NO",
        g.true_relative_residual, g.wall_seconds, g.orthogonalization_seconds, matvec_s,
        16.0 * static_cast<Real>(g.iterations + 1) * static_cast<Real>(n2) / 1e9);
    return buf;
}

/// Mean time of two operator applications (a load indicator next to the record's matvec).
Real matvec_seconds(const op::LinearOperator& Z) {
    VectorXc x = VectorXc::Ones(Z.cols()), y;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 2; ++i) Z.apply(x, y);
    return seconds_since(t0) / 2.0;
}

void run_pml(const Options& o) {
    const pml_rough_ag::RoughCase& c = o.c;
    const pml_rough_ag::Estimate est = pml_rough_ag::estimate(c);
    const auto beam = pml_rough_ag::make_beam(c, est.size.half[2]);
    SimulationConfig cfg = pml_rough_ag::make_config(c);
    cfg.compression = "mlfmm";
    cfg.solver = SolverKind::Gmres;
    cfg.mlfmm.accuracy_digits = o.digits;
    cfg.preconditioner = PreconditionerKind::None;  // built per variant below
    cfg.gmres.tolerance = o.tolerance.value_or(1e-7);
    cfg.gmres.max_iter = o.max_iter;
    cfg.gmres.restart = 0;
    Simulation sim(pml_rough_ag::make_open_mesh(c), pml_rough_ag::make_termination(c), beam, cfg);
    const auto t0 = std::chrono::steady_clock::now();
    sim.assemble();
    const Real assemble_s = seconds_since(t0);
    const auto* Z = dynamic_cast<const mlfmm::MlfmmOperator*>(sim.system_operator().get());
    if (Z == nullptr || Z->ring_operator() == nullptr)
        throw std::logic_error("preconditioner_study: not an MLFMM operator with a PML ring");
    const Index n2 = Z->rows();
    const int levels = Z->octree().levels();
    const Real leaf = Z->octree().box_size(Z->octree().leaf_level()) / pml_rough_ag::kLambda;
    char setup[600];
    std::snprintf(setup, sizeof(setup),
                  "SETUP | preconditioner | %s | %s | rms %.3g | L %.4g | Lgen %.4g | h %.4g | d0 "
                  "%.1f | tol %.1e | 2N %lld | stretched %lld | levels %d | leaf %.3f lambda | op "
                  "%.1f MB (rr near %.1f, ring near %.1f, ring far %.1f) | assemble %.1f s",
                  c.tag.c_str(), c.material.c_str(), c.rms, c.patch, c.map_patch, c.mesh_size,
                  o.digits, cfg.gmres.tolerance, static_cast<long long>(n2),
                  static_cast<long long>(Z->ring_operator()->stretched_count()), levels, leaf,
                  static_cast<Real>(Z->memory_bytes()) / 1e6,
                  static_cast<Real>(Z->near_operator().memory_bytes()) / 1e6,
                  static_cast<Real>(Z->ring_operator()->near_bytes()) / 1e6,
                  static_cast<Real>(Z->ring_operator()->far_bytes()) / 1e6, assemble_s);
    emit(o.summary, setup);
    std::optional<pml_rough_ag::FarField> first;
    for (const std::string& v : o.variants) {
        const Built b = build(v, sim, *Z);
        if (b.block != nullptr)
            std::printf("%s\n", b.block->describe().c_str());
        const solver::GmresResult g = solver::gmres(*Z, sim.rhs(), *b.M, cfg.gmres);
        const Real mv = matvec_seconds(*Z);
        const post::SurfaceSolution s{&sim.problem(), g.x};
        const pml_rough_ag::FarField far =
            pml_rough_ag::evaluate_far_field(sim, s, c, *sim.far_field_window(), beam->power());
        pml_rough_ag::Metrics m;
        if (first)
            m = pml_rough_ag::compare(far, *first);
        else
            first = far;
        char tail[300];
        std::snprintf(tail, sizeof(tail), " | R %.8f | dR vs %s %+.3e | l2 %.3e | peak %.2f GB",
                      far.reflectance(), o.variants.front().c_str(), m.dR, m.l2,
                      system_memory::peak_rss_bytes() / 1e9);
        emit(o.summary, "ROW | preconditioner | " + c.tag + " | " + c.material + " | 2N " +
                            std::to_string(n2) + " | " + solve_segment(v, b, g, n2, mv) + tail);
    }
    std::printf("%s\n", sim.report().c_str());
}

void run_mie(const Options& o) {
    ag_sphere_mlfmm::Case c;
    c.mesh = "ico";
    c.subdivisions = o.subdivisions;
    c.diameter = o.diameter;
    c.digits = o.digits;
    c.tolerance = o.tolerance.value_or(1e-6);
    c.max_iter = o.max_iter;
    SimulationConfig cfg = ag_sphere_mlfmm::make_config(c);
    cfg.preconditioner = PreconditionerKind::None;
    const geometry::TriangleMesh mesh =
        ag_sphere_mlfmm::make_mesh(c.mesh, c.diameter, c.subdivisions);
    Simulation sim(mesh, ag_sphere_mlfmm::make_wave(), cfg);
    const auto t0 = std::chrono::steady_clock::now();
    sim.assemble();
    const Real assemble_s = seconds_since(t0);
    const auto* Z = dynamic_cast<const mlfmm::MlfmmOperator*>(sim.system_operator().get());
    if (Z == nullptr)
        throw std::logic_error("preconditioner_study: not an MLFMM operator");
    const Index n2 = Z->rows();
    const std::string tag = o.c.tag == "u" ? "mie_ag" : o.c.tag;
    char setup[400];
    std::snprintf(setup, sizeof(setup),
                  "SETUP | preconditioner | %s | Ag sphere d %.3g ico %d | d0 %.1f | tol %.1e | 2N "
                  "%lld | levels %d | leaf %.3f lambda | op %.1f MB | assemble %.1f s",
                  tag.c_str(), c.diameter, c.subdivisions, c.digits, c.tolerance,
                  static_cast<long long>(n2), Z->octree().levels(),
                  Z->octree().box_size(Z->octree().leaf_level()) / ag_sphere_mlfmm::kLambda,
                  static_cast<Real>(Z->memory_bytes()) / 1e6, assemble_s);
    emit(o.summary, setup);
    const reference::MieSolution mie(
        reference::MieParams{0.5 * c.diameter, ag_sphere_mlfmm::kLambda, material::silver_500nm(),
                             material::vacuum(), 0});
    const VectorXr theta = VectorXr::LinSpaced(ag_sphere_mlfmm::kRcsAngles, 0.0, constants::pi);
    std::optional<VectorXr> first;
    for (const std::string& v : o.variants) {
        const Built b = build(v, sim, *Z);
        if (b.block != nullptr)
            std::printf("%s\n", b.block->describe().c_str());
        const solver::GmresResult g = solver::gmres(*Z, sim.rhs(), *b.M, cfg.gmres);
        const Real mv = matvec_seconds(*Z);
        const post::SurfaceSolution s{&sim.problem(), g.x};
        const Real exz =
            ag_sphere_mlfmm::eps_rr(s, mie, ag_sphere_mlfmm::kRcsAngles, Vec3::UnitY(), 0.0);
        const Real eyz = ag_sphere_mlfmm::eps_rr(s, mie, ag_sphere_mlfmm::kRcsAngles,
                                                 -Vec3::UnitX(), 0.5 * constants::pi);
        const VectorXr rcs = post::bistatic_rcs(s, Vec3::UnitY(), theta);
        Real drcs = 0;
        if (first)
            drcs = (rcs - *first).cwiseAbs().maxCoeff() / first->maxCoeff();
        else
            first = rcs;
        char tail[300];
        std::snprintf(tail, sizeof(tail),
                      " | eps_rr xz %.4e yz %.4e | rcs diff vs %s %.3e | peak %.2f GB", exz, eyz,
                      o.variants.front().c_str(), drcs, system_memory::peak_rss_bytes() / 1e9);
        emit(o.summary, "ROW | preconditioner | " + tag + " | ag | 2N " + std::to_string(n2) +
                            " | " + solve_segment(v, b, g, n2, mv) + tail);
    }
    std::printf("%s\n", sim.report().c_str());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        if (o.kind == "pml")
            run_pml(o);
        else
            run_mie(o);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "preconditioner_study: %s\n", e.what());
        return 1;
    }
}
