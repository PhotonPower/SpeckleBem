// Preconditioner study (WP-PC1, ADR 0011 decision 3, the gate of the block-diagonal default):
// one case per process, assembled once through Simulation with compression "mlfmm" (no
// preconditioner data), then solved with each requested preconditioner on the same operator and
// right-hand side (solver::gmres with the config's GMRES parameters):
//   jacobi  the diagonal of Z (MlfmmOperator::diagonal(), as Simulation's Jacobi path),
//   none    no preconditioner,
//   leaf    make_block_preconditioner(levels_above_leaf = 0): LU of the near-field block of
//           every octree leaf over its J and M unknowns,
//   parent  make_block_preconditioner(levels_above_leaf = 1): the same one level up,
//   grandparent  make_block_preconditioner(levels_above_leaf = 2): two levels up, the blocks of
//           the near-field matrix Z_n (same-box far pairs zero; WP-SC1, ADR 0012),
//   nearlu  make_near_field_lu_preconditioner: the exact sparse LU of Z_n = Z_rr's near part +
//           the near ring (WP-SC1, ADR 0012: the ceiling of near-field preconditioners), with the
//           budget --near-lu-budget GB (default 40) and the expected fill --near-lu-fill F
//           (default 1; pass the fill factor measured on a smaller case); a refusal (estimate or
//           factor memory above the budget, singular Z_n) is reported as a ROW line and the
//           remaining variants still run.
//   schur   make_schur_preconditioner(levels_above_leaf = --block-level, default 1): the
//           near-field Schur complement of WP-SC2 (ADR 0012 decision 2): block LU of A = Z_n[J, J]
//           over the J unknowns of each box, S = D - C A^-1 B on the pattern of D, block LU of S
//           over the M unknowns of each box, B and C exact (read from the operator's near parts).
//   nearlu_bc, nearlu_ad  diagnostics of WP-SC2 (where the near-field-inverse gain sits): the
//           exact sparse LU (as nearlu, same budget and fill options) of Z_n with the inter-box
//           entries (box pairs on the --block-level level) of A = Z_n[J, J] and D = Z_n[M, M]
//           removed (nearlu_bc: A and D box-diagonal, B and C full: the ceiling of the Schur
//           variant, whose Ã and S̃ blocks are box-diagonal too), or with those of B and C
//           removed (nearlu_ad: A and D full, B and C box-diagonal).
//   --near-norms  emits a NORMS line: Frobenius norms of A, B, C, D of Z_n split into the
//           same-box and the inter-box entries of the --block-level boxes.
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
//            [--precond jacobi,leaf,parent,grandparent,nearlu,schur] [--near-lu-budget 40]
//            [--near-lu-fill 1] [--block-level 1] [--near-norms] [--diameter D] [--subdiv N]
//            [--n-theta 90] [--n-phi 180] [--summary FILE]
// --case must come first (it resets the case).
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/octree.hpp"
#include "specklebem/compression/mlfmm/pml_ring.hpp"
#include "specklebem/core/path.hpp"
#include "specklebem/reference/mie.hpp"
#include "specklebem/simulation.hpp"
#include "specklebem/solver/gmres.hpp"
#include "specklebem/solver/preconditioner.hpp"
#include "specklebem/solver/schur_preconditioner.hpp"
#include "specklebem/solver/sparse_lu_preconditioner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
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
    Real near_lu_budget_gb = 40.0;
    Real near_lu_fill = 1.0;
    int schur_level = 1;  ///< --block-level: levels above the leaves of the schur variant
    bool near_norms = false;
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
        } else if (a == "--near-lu-budget") {
            o.near_lu_budget_gb = real();
        } else if (a == "--near-lu-fill") {
            o.near_lu_fill = real();
        } else if (a == "--block-level") {
            o.schur_level = std::stoi(value());
        } else if (a == "--near-norms") {
            o.near_norms = true;
        } else if (a == "--summary") {
            o.summary = value();
        } else {
            throw std::invalid_argument("unknown option " + a);
        }
    }
    for (const std::string& v : o.variants) {
        if (v != "jacobi" && v != "none" && v != "leaf" && v != "parent" && v != "grandparent" &&
            v != "nearlu" && v != "schur" && v != "nearlu_bc" && v != "nearlu_ad")
            throw std::invalid_argument("unknown --precond variant " + v);
    }
    if (o.variants.empty())
        throw std::invalid_argument("--precond: no variant");
    if (!(o.near_lu_budget_gb > 0.0) || !(o.near_lu_fill > 0.0))
        throw std::invalid_argument("--near-lu-budget and --near-lu-fill must be > 0");
    if (o.schur_level < 0 || o.schur_level > 2)
        throw std::invalid_argument("--block-level must be 0, 1 or 2");
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
    const solver::SparseLuPreconditioner* lu = nullptr;
    const solver::SchurComplementPreconditioner* schur = nullptr;
    Real setup_s = 0;
    std::size_t bytes = 0;
};

/// Box (block number) of every unknown on the level `levels_above_leaf` above the leaves.
std::vector<Index> unknown_boxes(const mlfmm::MlfmmOperator& Z, int levels_above_leaf) {
    const mlfmm::Octree& tree = Z.octree();
    const int level = std::max(0, tree.leaf_level() - levels_above_leaf);
    const std::vector<std::vector<Index>> blocks = mlfmm::level_unknown_blocks(tree, level);
    std::vector<Index> box(static_cast<std::size_t>(Z.rows()), -1);
    for (std::size_t k = 0; k < blocks.size(); ++k)
        for (const Index i : blocks[k]) box[static_cast<std::size_t>(i)] = static_cast<Index>(k);
    return box;
}

/// Z_n without the inter-box entries of A and D (keep_bc) or of B and C (!keep_bc).
op::SparseOperator::Matrix filtered_near_field(const mlfmm::MlfmmOperator& Z, int levels_above_leaf,
                                               bool keep_bc) {
    const std::vector<Index> box = unknown_boxes(Z, levels_above_leaf);
    const Index n = Z.rows() / 2;
    op::SparseOperator::Matrix Zn = near_field_matrix(Z);
    Zn.prune([&](Index r, Index c, const Complex&) {
        const bool same_box = box[static_cast<std::size_t>(r)] == box[static_cast<std::size_t>(c)];
        const bool off_half = (r < n) != (c < n);  // B or C
        return same_box || (off_half == keep_bc);
    });
    Zn.makeCompressed();
    return Zn;
}

/// "NORMS | ... | A same a1 other a2 | B ... | C ... | D ..." (Frobenius norms of Z_n's parts).
std::string near_norms_line(const mlfmm::MlfmmOperator& Z, int levels_above_leaf) {
    const std::vector<Index> box = unknown_boxes(Z, levels_above_leaf);
    const Index n = Z.rows() / 2;
    const op::SparseOperator::Matrix Zn = near_field_matrix(Z);
    Real sq[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    Index cnt[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    for (Index r = 0; r < Zn.outerSize(); ++r) {
        for (op::SparseOperator::Matrix::InnerIterator it(Zn, r); it; ++it) {
            const Index c = it.col();
            const int part = (r < n ? 0 : 2) + (c < n ? 0 : 1);  // A, B, C, D
            const int other =
                box[static_cast<std::size_t>(r)] == box[static_cast<std::size_t>(c)] ? 0 : 1;
            sq[part][other] += std::norm(it.value());
            ++cnt[part][other];
        }
    }
    std::string line = "NORMS | near-field parts (Frobenius norm, entries), level " +
                       std::to_string(levels_above_leaf) + " above the leaves";
    const char* names[4] = {"A", "B", "C", "D"};
    for (int p = 0; p < 4; ++p) {
        char buf[200];
        std::snprintf(buf, sizeof(buf), " | %s same-box %.4e (%lld) inter-box %.4e (%lld)",
                      names[p], std::sqrt(sq[p][0]), static_cast<long long>(cnt[p][0]),
                      std::sqrt(sq[p][1]), static_cast<long long>(cnt[p][1]));
        line += buf;
    }
    return line;
}

Built build(const std::string& variant, const Simulation& sim, const mlfmm::MlfmmOperator& Z,
            const Options& o) {
    Built b;
    const auto t0 = std::chrono::steady_clock::now();
    if (variant == "nearlu") {
        auto P = make_near_field_lu_preconditioner(
            Z, static_cast<std::size_t>(o.near_lu_budget_gb * 1e9), o.near_lu_fill);
        b.lu = P.get();
        b.bytes = P->memory_bytes();
        b.M = std::move(P);
    } else if (variant == "nearlu_bc" || variant == "nearlu_ad") {
        const op::SparseOperator::Matrix Zf =
            filtered_near_field(Z, o.schur_level, variant == "nearlu_bc");
        auto P = std::make_unique<solver::SparseLuPreconditioner>(
            Zf, static_cast<std::size_t>(o.near_lu_budget_gb * 1e9), o.near_lu_fill);
        b.lu = P.get();
        b.bytes = P->memory_bytes();
        b.M = std::move(P);
    } else if (variant == "schur") {
        auto P = make_schur_preconditioner(Z, sim.problem(), o.schur_level);
        b.schur = P.get();
        b.bytes = P->memory_bytes();
        b.M = std::move(P);
    } else if (variant == "jacobi") {
        b.M = std::make_unique<solver::DiagonalPreconditioner>(Z.diagonal());
        b.bytes = static_cast<std::size_t>(Z.rows()) * sizeof(Complex);
    } else if (variant == "none") {
        b.M = std::make_unique<solver::IdentityPreconditioner>();
    } else {
        const int up = variant == "leaf" ? 0 : variant == "parent" ? 1 : 2;
        auto P = make_block_preconditioner(Z, sim.problem(), sim.config().mlfmm, up);
        b.block = P.get();
        b.bytes = P->memory_bytes();
        b.M = std::move(P);
    }
    b.setup_s = seconds_since(t0);
    return b;
}

/// Common part of a ROW line: the variant, its preconditioner and the GMRES result. Block
/// variants: blocks (unknowns min / mean / max; with entries outside the near field, two levels
/// up, their share "far-zero"), extraction / LU times and the smallest block rcond; nearlu:
/// nnz(Z_n), nnz(L + U), the fill factor, analysis / factorisation times; schur: the level, the J
/// and M blocks (unknowns min / mean / max), nnz(B), nnz(C), the M-block entries of S (masked
/// share outside the pattern of D, if any), the coupling pairs and the formation work, the setup
/// split (A LU, S formation, S LU) and the smallest rcond of A's and S's blocks.
std::string solve_segment(const std::string& variant, const Built& b, const solver::GmresResult& g,
                          Index n2, Real matvec_s, int schur_level) {
    char head[700];
    const solver::BlockDiagonalPreconditioner* P = b.block;
    if (b.schur != nullptr) {
        const solver::SchurComplementPreconditioner& S = *b.schur;
        char masked[64] = "";
        if (S.masked_entries_s() > 0) {
            std::snprintf(masked, sizeof(masked), " masked %.1f %%",
                          100.0 * static_cast<Real>(S.masked_entries_s()) /
                              static_cast<Real>(S.m_block_entries()));
        }
        const Real n = static_cast<Real>(S.half_size());
        std::snprintf(
            head, sizeof(head),
            "variant schur | level %d | J blocks %lld (n %lld / %.1f / %lld) | M blocks %lld (n "
            "%lld / %.1f / %lld) | nnz B %lld C %lld | S entries %lld%s | pairs %lld (%.1f GFlop) "
            "| P %.1f MB | setup %.2f s (A LU %.2f, S form %.2f, S LU %.2f) | rcond A %.2e S %.2e",
            schur_level, static_cast<long long>(S.num_j_blocks()),
            static_cast<long long>(S.min_j_block_size()), n / static_cast<Real>(S.num_j_blocks()),
            static_cast<long long>(S.max_j_block_size()), static_cast<long long>(S.num_m_blocks()),
            static_cast<long long>(S.min_m_block_size()), n / static_cast<Real>(S.num_m_blocks()),
            static_cast<long long>(S.max_m_block_size()), static_cast<long long>(S.nonzeros_b()),
            static_cast<long long>(S.nonzeros_c()), static_cast<long long>(S.m_block_entries()),
            masked, static_cast<long long>(S.coupling_pairs()), 8.0 * S.formation_macs() / 1e9,
            static_cast<Real>(b.bytes) / 1e6, b.setup_s, S.a_factorization_seconds(),
            S.schur_formation_seconds(), S.schur_factorization_seconds(),
            S.min_reciprocal_condition_a(), S.min_reciprocal_condition_s());
    } else if (b.lu != nullptr) {
        const solver::SparseLuPreconditioner& L = *b.lu;
        std::snprintf(
            head, sizeof(head),
            "variant %s | nnz Zn %lld | nnz LU %lld (L %lld, U %lld) | fill %.2f | P "
            "%.1f MB (estimate %.1f MB) | setup %.2f s (analysis %.2f, LU %.2f) | rcond -",
            variant.c_str(), static_cast<long long>(L.input_nonzeros()),
            static_cast<long long>(L.factor_nonzeros()), static_cast<long long>(L.nonzeros_l()),
            static_cast<long long>(L.nonzeros_u()), L.fill_factor(),
            static_cast<Real>(b.bytes) / 1e6, static_cast<Real>(L.estimated_bytes()) / 1e6,
            b.setup_s, L.analysis_seconds(), L.factorization_seconds());
    } else {
        char zero[64] = "";
        if (P != nullptr && P->missing_entries() > 0) {
            std::snprintf(zero, sizeof(zero), " far-zero %.1f %%",
                          100.0 * static_cast<Real>(P->missing_entries()) /
                              static_cast<Real>(P->block_entries()));
        }
        std::snprintf(
            head, sizeof(head),
            "variant %s | blocks %lld (n %lld / %.1f / %lld)%s | P %.1f MB | setup %.2f s (extract "
            "%.2f, LU %.2f) | rcond %.2e",
            variant.c_str(), static_cast<long long>(P ? P->num_blocks() : 0),
            static_cast<long long>(P ? P->min_block_size() : 0),
            P ? static_cast<Real>(n2) / static_cast<Real>(P->num_blocks()) : 0.0,
            static_cast<long long>(P ? P->max_block_size() : 0), zero,
            static_cast<Real>(b.bytes) / 1e6, b.setup_s, P ? P->extraction_seconds() : 0.0,
            P ? P->factorization_seconds() : 0.0, P ? P->min_reciprocal_condition() : 0.0);
    }
    char buf[1000];
    std::snprintf(buf, sizeof(buf),
                  "%s | its %d | converged %s | true res %.2e | GMRES %.1f s (orth %.1f s) | "
                  "matvec %.3f s | basis %.2f GB",
                  head, g.iterations, g.converged ? "yes" : "NO", g.true_relative_residual,
                  g.wall_seconds, g.orthogonalization_seconds, matvec_s,
                  16.0 * static_cast<Real>(g.iterations + 1) * static_cast<Real>(n2) / 1e9);
    return buf;
}

/// build() of one variant; a refused nearlu (budget, singular) is reported, not fatal.
std::optional<Built> try_build(const std::string& variant, const Simulation& sim,
                               const mlfmm::MlfmmOperator& Z, const Options& o,
                               const std::string& row_head) {
    try {
        return build(variant, sim, Z, o);
    } catch (const std::runtime_error& e) {
        if (variant.rfind("nearlu", 0) != 0)
            throw;
        emit(o.summary, row_head + "variant " + variant + " | refused: " + e.what());
        return std::nullopt;
    }
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
    if (o.near_norms)
        emit(o.summary, near_norms_line(*Z, o.schur_level));
    std::optional<pml_rough_ag::FarField> first;
    const std::string row_head = "ROW | preconditioner | " + c.tag + " | " + c.material + " | 2N " +
                                 std::to_string(n2) + " | ";
    for (const std::string& v : o.variants) {
        const std::optional<Built> built = try_build(v, sim, *Z, o, row_head);
        if (!built)
            continue;
        const Built& b = *built;
        if (b.block != nullptr)
            std::printf("%s\n", b.block->describe().c_str());
        if (b.lu != nullptr)
            std::printf("%s\n", b.lu->describe().c_str());
        if (b.schur != nullptr)
            std::printf("%s\n", b.schur->describe().c_str());
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
        emit(o.summary, row_head + solve_segment(v, b, g, n2, mv, o.schur_level) + tail);
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
    const std::string row_head =
        "ROW | preconditioner | " + tag + " | ag | 2N " + std::to_string(n2) + " | ";
    for (const std::string& v : o.variants) {
        const std::optional<Built> built = try_build(v, sim, *Z, o, row_head);
        if (!built)
            continue;
        const Built& b = *built;
        if (b.block != nullptr)
            std::printf("%s\n", b.block->describe().c_str());
        if (b.lu != nullptr)
            std::printf("%s\n", b.lu->describe().c_str());
        if (b.schur != nullptr)
            std::printf("%s\n", b.schur->describe().c_str());
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
        emit(o.summary, row_head + solve_segment(v, b, g, n2, mv, o.schur_level) + tail);
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
