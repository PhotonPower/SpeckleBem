// Formulation convergence study (WP15, Phase 3 definition of done): GMRES iteration counts of
// PMCHWT, ICTF and MCTF, without and with the left Jacobi preconditioner, on Gaussian rough
// surfaces of Si and Ag at 500 nm, dense operator (Fu et al. 2023, Sec. 4 and Fig. 2a,b).
//
// Setup (defaults): sigma = 50 nm, Lc = 500 nm, mesh 50 nm (lambda/10), L = 3.2 um (reduced from
// the paper's 10 um so that the dense operator fits, 2N ~ 3e4), closing box of depth
// default_box_depth(object) (ADR 0006), graded, no fine band; paraxial Gaussian beam at normal
// incidence (+z, from z < 0), waist L/3, focused on z = 0, p-polarised (E along x); full GMRES
// (no restart), tolerance 1e-3, max_iter 2000.
//
// Per material and formulation one Simulation assembles the operator once; it is solved without
// preconditioner through Simulation::solve and with the left Jacobi preconditioner
// (op::assemble_diagonal) by solver::gmres on the same operator. The systems are processed one
// after the other (Z is ~14 GB at 2N = 3e4).
//
// Output: a summary line per run on stdout, the residual histories (and the mesh) in a .npy
// directory (io::open_npy_directory) and, optionally, a compact CSV of the histories (every
// `--csv-stride`-th iteration plus the last), appended run by run.
//
// Usage: specklebem_formulation_convergence [--materials si,ag] [--L 3.2e-6] [--seed 1]
//            [--out DIR] [--csv FILE] [--csv-stride 10] [--max-iter 2000] [--tol 1e-3]
//            [--formulations pmchwt,ictf,mctf] [--waist-factor 3] [--fine-band] [--mesh-only]
#include "specklebem/core/logging.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/io/result_writer.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/simulation.hpp"
#include "specklebem/solver/gmres.hpp"
#include "specklebem/solver/preconditioner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
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

struct Options {
    std::vector<std::string> materials{"si", "ag"};
    std::vector<std::string> formulations{"pmchwt", "ictf", "mctf"};
    Real L = 3.2e-6;
    std::uint64_t seed = 1;
    std::string out_dir;
    std::string csv;
    int csv_stride = 10;
    int max_iter = 2000;
    Real tol = 1e-3;
    Real waist_factor = 3.0;  // w0 = L / waist_factor
    bool fine_band = false;
    bool mesh_only = false;
};

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty())
            out.push_back(item);
    }
    return out;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::invalid_argument("missing value after " + a);
            return argv[++i];
        };
        if (a == "--materials") {
            o.materials = split(value());
        } else if (a == "--formulations") {
            o.formulations = split(value());
        } else if (a == "--L") {
            o.L = std::stod(value());
        } else if (a == "--seed") {
            o.seed = std::stoull(value());
        } else if (a == "--out") {
            o.out_dir = value();
        } else if (a == "--csv") {
            o.csv = value();
        } else if (a == "--csv-stride") {
            o.csv_stride = std::stoi(value());
        } else if (a == "--max-iter") {
            o.max_iter = std::stoi(value());
        } else if (a == "--tol") {
            o.tol = std::stod(value());
        } else if (a == "--waist-factor") {
            o.waist_factor = std::stod(value());
        } else if (a == "--fine-band") {
            o.fine_band = true;
        } else if (a == "--mesh-only") {
            o.mesh_only = true;
        } else {
            throw std::invalid_argument("unknown argument " + a);
        }
    }
    if (!(o.L > 0) || o.seed == 0 || o.csv_stride < 1 || o.max_iter < 1 || !(o.tol > 0) ||
        !(o.waist_factor > 0)) {
        throw std::invalid_argument(
            "need L > 0, seed > 0 (fixed), csv-stride >= 1, max-iter >= 1, tol > 0, "
            "waist-factor > 0");
    }
    return o;
}

material::Material material_by_name(const std::string& name) {
    if (name == "si")
        return material::silicon_500nm();
    if (name == "ag")
        return material::silver_500nm();
    throw std::invalid_argument("unknown material " + name + " (si or ag)");
}

formulation::Kind formulation_by_name(const std::string& name) {
    if (name == "pmchwt")
        return formulation::Kind::PMCHWT;
    if (name == "ictf")
        return formulation::Kind::ICTF;
    if (name == "mctf")
        return formulation::Kind::MCTF;
    throw std::invalid_argument("unknown formulation " + name + " (pmchwt, ictf or mctf)");
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

struct Run {
    std::string material;
    std::string formulation;
    std::string preconditioner;  // "none" or "jacobi"
    solver::GmresResult result;
    double assembly_seconds = 0;
};

/// Appends the history of one run (every stride-th iteration plus the last) to the CSV.
void append_csv(const std::string& path, int stride, const Run& r) {
    if (path.empty())
        return;
    const bool fresh = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
    std::ofstream f(path, std::ios::app | std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    if (fresh)
        f << "material,formulation,preconditioner,iteration,residual\n";
    const auto& h = r.result.residual_history;
    const int last = static_cast<int>(h.size()) - 1;
    char buf[160];
    for (int k = 0; k <= last; ++k) {
        if (k % stride != 0 && k != last)
            continue;
        std::snprintf(buf, sizeof(buf), "%s,%s,%s,%d,%.6e\n", r.material.c_str(),
                      r.formulation.c_str(), r.preconditioner.c_str(), k,
                      h[static_cast<std::size_t>(k)]);
        f << buf;
    }
}

void print_run(const Run& r, Index unknowns) {
    const auto& g = r.result;
    std::printf(
        "RESULT | %s | %s | %s | 2N %lld | iterations %d | converged %s | monitored %.3e | true "
        "%.3e | assembly %.1f s | solve %.1f s | %.3f s/it\n",
        r.material.c_str(), r.formulation.c_str(), r.preconditioner.c_str(),
        static_cast<long long>(unknowns), g.iterations, g.converged ? "yes" : "no",
        g.residual_history.back(), g.true_relative_residual, r.assembly_seconds, g.wall_seconds,
        g.iterations > 0 ? g.wall_seconds / g.iterations : 0.0);
    std::fflush(stdout);
}

void write_run(io::ResultWriter* w, const Run& r) {
    if (w == nullptr)
        return;
    const std::string p = r.material + "_" + r.formulation + "_" + r.preconditioner;
    const auto& h = r.result.residual_history;
    MatrixXr m(static_cast<Index>(h.size()), 1);
    for (std::size_t k = 0; k < h.size(); ++k) m(static_cast<Index>(k), 0) = h[k];
    w->write_matrix(p + "_residual_history", m);
    w->write_attribute(p + "_iterations", static_cast<Real>(r.result.iterations));
    w->write_attribute(p + "_converged", r.result.converged ? "true" : "false");
    w->write_attribute(p + "_monitored_residual", h.back());
    w->write_attribute(p + "_true_residual", r.result.true_relative_residual);
    w->write_attribute(p + "_assembly_seconds", r.assembly_seconds);
    w->write_attribute(p + "_solve_seconds", r.result.wall_seconds);
}

/// Largest relative difference |a_k - b_k| / b_k over the common iterations.
Real max_rel_history_diff(const std::vector<Real>& a, const std::vector<Real>& b) {
    Real d = 0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t k = 0; k < n; ++k) d = std::max(d, std::abs(a[k] - b[k]) / b[k]);
    return d;
}

void run_material(const Options& o, const std::string& mat_name, io::ResultWriter* writer) {
    const material::Material object = material_by_name(mat_name);
    const Real delta = material::field_decay_length(object, kLambda);
    const Real depth = default_box_depth(object, kLambda);

    geometry::RoughSurfaceParams rp;
    rp.edge_length_L = o.L;
    rp.rms_roughness = kSigma;
    rp.correlation_length = kCorrelationLength;
    rp.mesh_size = kMeshSize;
    rp.seed = o.seed;
    rp.box_depth = depth;
    if (o.fine_band)
        rp.box_fine_depth = default_box_fine_depth(object, kLambda, kSigma);

    const geometry::HeightMap hm = geometry::generate_gaussian_height_map(rp);
    geometry::TriangleMesh mesh =
        geometry::make_mesh_from_height_map(hm, rp.box_depth, rp.box_mesh_size, rp.box_fine_depth);
    const Index top = 2 * (hm.z.rows() - 1) * (hm.z.cols() - 1);
    const Index tris = mesh.num_triangles();
    const Index n_edges = mesh.num_edges();
    const Real w0 = o.L / o.waist_factor;
    const Real paraxial = std::pow(kLambda / (constants::pi * w0), 2);

    char fine_buf[32] = "none";
    if (rp.box_fine_depth)
        std::snprintf(fine_buf, sizeof(fine_buf), "%.4g m", *rp.box_fine_depth);
    const std::string fine_band = fine_buf;
    std::printf(
        "SETUP | %s | eps_r %.4g%+.4gj | delta %.4g m | box depth %.4g m | fine band %s | L %.4g m "
        "| grid %lld x %lld (dx %.4g m) | rms %.4g m | Lc est %.4g m | triangles %lld (top %lld, "
        "box %lld = %.1f %%) | N %lld | 2N %lld | Z %.2f GB | w0 %.4g m | (lambda/(pi w0))^2 "
        "%.3g | seed %llu\n",
        mat_name.c_str(), object.eps_r.real(), object.eps_r.imag(), delta, depth, fine_band.c_str(),
        o.L, static_cast<long long>(hm.z.rows()), static_cast<long long>(hm.z.cols()), hm.dx,
        hm.rms(), hm.estimated_correlation_length(), static_cast<long long>(tris),
        static_cast<long long>(top), static_cast<long long>(tris - top),
        100.0 * static_cast<double>(tris - top) / static_cast<double>(top),
        static_cast<long long>(n_edges), static_cast<long long>(2 * n_edges),
        16.0 * std::pow(2.0 * static_cast<double>(n_edges), 2) / 1e9, w0, paraxial,
        static_cast<unsigned long long>(o.seed));
    std::fflush(stdout);
    if (writer != nullptr) {
        writer->write_mesh(mat_name + "_mesh", mesh);
        writer->write_attribute(mat_name + "_box_depth", depth);
        writer->write_attribute(mat_name + "_unknowns", static_cast<Real>(2 * n_edges));
        writer->write_attribute(mat_name + "_triangles", static_cast<Real>(tris));
        writer->write_attribute(mat_name + "_top_triangles", static_cast<Real>(top));
    }
    if (o.mesh_only)
        return;

    excitation::GaussianBeam::Params bp;
    bp.wavelength = kLambda;
    bp.waist_radius = w0;
    bp.focus = Vec3::Zero();
    bp.incidence_angle = 0.0;
    bp.polarization = excitation::Polarization::P;
    auto beam = std::make_shared<excitation::GaussianBeam>(bp);

    solver::GmresParams gp;
    gp.tolerance = o.tol;
    gp.max_iter = o.max_iter;
    gp.restart = 0;
    gp.verbose = true;
    gp.side = solver::PreconditionerSide::Left;

    std::vector<Run> jacobi_runs;
    for (const std::string& f_name : o.formulations) {
        SimulationConfig cfg;
        cfg.wavelength = kLambda;
        cfg.exterior = material::vacuum();
        cfg.object = object;
        cfg.formulation = formulation_by_name(f_name);
        cfg.diagonal_preconditioner = false;
        cfg.gmres = gp;
        Simulation sim(mesh, beam, cfg);

        const auto t0 = std::chrono::steady_clock::now();
        sim.assemble();
        const double t_asm = seconds_since(t0);
        SBEM_INFO("{} {}: assembled 2N = {} in {:.1f} s ({:.2f} GB)", mat_name, f_name,
                  sim.num_unknowns(), t_asm,
                  static_cast<double>(sim.system_operator()->memory_bytes()) / 1e9);

        Run plain{mat_name, f_name, "none", sim.solve(gp), t_asm};
        print_run(plain, sim.num_unknowns());
        write_run(writer, plain);
        append_csv(o.csv, o.csv_stride, plain);

        const auto t1 = std::chrono::steady_clock::now();
        const solver::DiagonalPreconditioner jacobi(op::assemble_diagonal(sim.problem()));
        const double t_diag = seconds_since(t1);
        Run pre{mat_name, f_name, "jacobi",
                solver::gmres(*sim.system_operator(), sim.rhs(), jacobi, gp), t_asm + t_diag};
        print_run(pre, sim.num_unknowns());
        write_run(writer, pre);
        append_csv(o.csv, o.csv_stride, pre);
        jacobi_runs.push_back(std::move(pre));
    }

    // Issue #15: left-Jacobi PMCHWT, ICTF and MCTF are block-row scalings of each other and
    // must iterate identically up to rounding.
    for (std::size_t i = 1; i < jacobi_runs.size(); ++i) {
        const Run& a = jacobi_runs[i];
        const Run& b = jacobi_runs[0];
        std::printf(
            "ISSUE15 | %s | %s+jacobi vs %s+jacobi | iterations %d vs %d | max rel history diff "
            "%.3e | true residual %.3e vs %.3e\n",
            mat_name.c_str(), a.formulation.c_str(), b.formulation.c_str(), a.result.iterations,
            b.result.iterations,
            max_rel_history_diff(a.result.residual_history, b.result.residual_history),
            a.result.true_relative_residual, b.result.true_relative_residual);
    }
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        std::unique_ptr<io::ResultWriter> writer;
        if (!o.out_dir.empty()) {
            writer = io::open_npy_directory(o.out_dir);
            writer->write_attribute("wavelength", kLambda);
            writer->write_attribute("rms_roughness", kSigma);
            writer->write_attribute("correlation_length", kCorrelationLength);
            writer->write_attribute("mesh_size", kMeshSize);
            writer->write_attribute("edge_length_L", o.L);
            writer->write_attribute("seed", static_cast<Real>(o.seed));
            writer->write_attribute("tolerance", o.tol);
            writer->write_attribute("max_iter", static_cast<Real>(o.max_iter));
            writer->write_attribute("waist_radius", o.L / o.waist_factor);
        }
        for (const std::string& m : o.materials) run_material(o, m, writer.get());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "formulation_convergence: %s\n", e.what());
        return 1;
    }
}
