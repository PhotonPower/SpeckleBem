// Validation 7e of the open-surface PML route (WP-PML4c, ADR 0009 decision 7e; docs/05 "PML
// route" row 7e): a Gaussian rough Si surface (seed, sigma, Lc) solved on the PML route (open
// surface with taper, gap and complex-coordinate-stretching PML, dense operator, LU or GMRES) and
// on the validated uniform closing box of ADR 0006 (MLFMM d0 = 3 + GMRES), with the same height
// map, the same patch mesh and the same rigorous beam (normal incidence, E along x); the far
// fields are compared with the ADR 0006 box-validity metric (L2 of |F|^2 over the reflection
// hemisphere with w = sin theta, eps_rr on the xz/yz cuts, reflectance R = P_refl / P_inc).
// Shared code: tests/support/pml_si_crosscheck_support.hpp. Record:
// benchmarks/results/pml_si_crosscheck.md, rows appended to pml_validation_rows.txt.
//
// One route per process (costs and peak memory per route): `--route pml|box` solves, prints the
// SETUP, IDENTITY and ESTIMATE lines and one ROW line (also appended to --summary FILE), and
// stores the far field in --out DIR; `--reference DIR` (repeatable) compares the run with stored
// runs, `--compare REF TEST...` compares stored runs only (COMPARE lines with the 7e tolerances).
// The height map is generated once per process on --map x --map at --gen-h and subsampled to
// --h, so that both routes (and meshes at lambda/4, lambda/6, lambda/10) see the same surface;
// the IDENTITY line checks that the open mesh's patch heights are bitwise the box top face.
// `--estimate-only` prints the setup, the identity check, the memory estimates of both routes and
// the PML route's dark-band ratio and masked-function count without assembling. A run whose peak
// estimate exceeds --max-gb (default 50) or the available memory minus 4 GB is refused without
// --force.
//
// Usage: specklebem_pml_si_crosscheck --route pml|box [--L 4.5e-6] [--w0 0.75e-6] [--h 125e-9]
//            [--gen-h 8.3333333333333e-9] [--map 6.5e-6] [--seed 1] [--sigma 50e-9]
//            [--lc 500e-9] [--beam-radius 7e-6] [--taper 0.5e-6] [--gap 0.5e-6] [--pml 1e-6]
//            [--sigma-max 4] [--order 2] [--dark-band-tol 1e-4] [--acc 1e-5] [--stp 10]
//            [--pml-solver lu|gmres] [--eigen-lu] [--pml-tol 1e-8] [--pml-max-iter 2000]
//            [--depth-factor 1] [--d0 3] [--tol 1e-5] [--max-iter 400] [--ff-degree 10]
//            [--max-gb 50] [--force] [--estimate-only] [--tag NAME] [--out DIR]
//            [--reference DIR]... [--summary FILE]
//        specklebem_pml_si_crosscheck --compare REF TEST [TEST...] [--summary FILE]
#include <cstdio>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pml_si_crosscheck_support.hpp"

namespace {

using namespace pml_si_crosscheck;

struct Options {
    Case c;
    std::string route;
    Real max_gb = 50.0;
    bool force = false;
    bool estimate_only = false;
    std::string tag;
    std::string out;
    std::string summary;
    std::vector<std::string> references;
    std::vector<std::string> compare;
};

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
        if (a == "--route") {
            o.route = value();
        } else if (a == "--L") {
            o.c.L = real();
        } else if (a == "--w0") {
            o.c.waist = real();
        } else if (a == "--h") {
            o.c.mesh_size = real();
        } else if (a == "--gen-h") {
            o.c.gen_spacing = real();
        } else if (a == "--map") {
            o.c.map_size = real();
        } else if (a == "--seed") {
            o.c.seed = std::stoull(value());
        } else if (a == "--sigma") {
            o.c.sigma = real();
        } else if (a == "--lc") {
            o.c.correlation_length = real();
        } else if (a == "--beam-radius") {
            o.c.beam_radius = real();
        } else if (a == "--taper") {
            o.c.taper = real();
        } else if (a == "--gap") {
            o.c.gap = real();
        } else if (a == "--pml") {
            o.c.pml_width = real();
        } else if (a == "--sigma-max") {
            o.c.sigma_max = real();
        } else if (a == "--order") {
            o.c.order = real();
        } else if (a == "--dark-band-tol") {
            o.c.dark_band_tolerance = real();
        } else if (a == "--acc") {
            o.c.target_accuracy = real();
        } else if (a == "--stp") {
            o.c.stretched_touching_points = std::stoi(value());
        } else if (a == "--pml-solver") {
            const std::string v = value();
            if (v == "lu")
                o.c.pml_solver = PmlSolver::Lu;
            else if (v == "gmres")
                o.c.pml_solver = PmlSolver::Gmres;
            else
                throw std::invalid_argument("--pml-solver must be lu or gmres");
        } else if (a == "--eigen-lu") {
            o.c.use_lapack = false;
        } else if (a == "--pml-tol") {
            o.c.pml_tol = real();
        } else if (a == "--pml-max-iter") {
            o.c.pml_max_iter = std::stoi(value());
        } else if (a == "--depth-factor") {
            o.c.depth_factor = real();
        } else if (a == "--d0") {
            o.c.digits = real();
        } else if (a == "--tol") {
            o.c.box_tol = real();
        } else if (a == "--max-iter") {
            o.c.box_max_iter = std::stoi(value());
        } else if (a == "--ff-degree") {
            o.c.ff_degree = std::stoi(value());
        } else if (a == "--max-gb") {
            o.max_gb = real();
        } else if (a == "--force") {
            o.force = true;
        } else if (a == "--estimate-only") {
            o.estimate_only = true;
        } else if (a == "--tag") {
            o.tag = value();
        } else if (a == "--out") {
            o.out = value();
        } else if (a == "--reference") {
            o.references.push_back(value());
        } else if (a == "--summary") {
            o.summary = value();
        } else if (a == "--compare") {
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
                o.compare.emplace_back(argv[++i]);
        } else {
            throw std::invalid_argument("unknown option " + a);
        }
    }
    if (!o.compare.empty()) {
        if (o.compare.size() < 2)
            throw std::invalid_argument("--compare needs a reference and at least one test");
        return o;
    }
    if (o.route != "pml" && o.route != "box")
        throw std::invalid_argument("--route must be pml or box (or use --compare)");
    if (!o.references.empty() && o.out.empty())
        throw std::invalid_argument("--reference needs --out (the stored attributes are compared)");
    if (!(o.max_gb > 0))
        throw std::invalid_argument("--max-gb must be > 0");
    validate(o.c);
    return o;
}

void emit(const std::string& summary, const std::string& line) {
    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
    if (summary.empty())
        return;
    std::ofstream f(summary, std::ios::app | std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + summary);
    f << line << '\n';
}

int run_compare(const Options& o) {
    const StoredRun ref = load(o.compare.front());
    for (std::size_t i = 1; i < o.compare.size(); ++i) {
        const StoredRun t = load(o.compare[i]);
        emit(o.summary, compare_line(t.tag, ref.tag, compare_stored(t, ref), kTolerances));
    }
    return 0;
}

int run(const Options& o) {
    const Case& c = o.c;
    const std::string tag = o.tag.empty() ? o.route : o.tag;
    const geometry::HeightMap map = mesh_map(c);
    // Both meshes are built in every run (cheap) for the identity check of the shared map.
    const geometry::OpenSurfaceMesh om = make_open_mesh(map, c);
    const BoxGeometry box = make_box(map, c);
    const Index open_patch = count_patch_triangles(om);
    const HeightIdentity id = height_identity(box.top, om.mesh, box.mesh);
    const TriangleIdentity tri = patch_triangle_identity(box.top, om, box.mesh);
    const Real avail = system_memory::available_memory_bytes() / 1e9;
    const PmlEstimate pe = estimate_pml(c, om);
    std::printf(
        "SETUP | %s | Si, lambda %.4g m | seed %llu, sigma %.4g m, Lc %.4g m | map %.4g m at gen "
        "%.6g m, subsampled to h %.6g m (%lld x %lld nodes) | L %.4g m | w0 %.4g m (L/%.3g) | "
        "beam radius %.4g m | available %.1f GB\n",
        tag.c_str(), kLambda, static_cast<unsigned long long>(c.seed), c.sigma,
        c.correlation_length, c.map_size, c.gen_spacing, map.dx,
        static_cast<long long>(map.z.rows()), static_cast<long long>(map.z.cols()), c.L, c.waist,
        c.L / c.waist, c.beam_radius, avail);
    emit(o.summary, identity_line(id, open_patch, box.top_triangles));
    if (!id.bitwise())
        throw std::runtime_error("the open mesh's patch heights differ from the box top face");
    std::printf(
        "TRIANGLES | %s | open patch %lld, box top %lld, vertex triples matched %lld | %s\n",
        tag.c_str(), static_cast<long long>(tri.open_patch), static_cast<long long>(tri.box_top),
        static_cast<long long>(tri.matched), tri.agree() ? "agree" : "DIFFER");
    if (!tri.agree())
        throw std::runtime_error("the open mesh's patch triangles differ from the box top face");
    std::printf(
        "ESTIMATE | %s | pml | half widths %.4g / %.4g / %.4g / %.4g m | triangles %lld | 2N %lld "
        "| dense %.2f GB | peak (%s) %.2f GB\n",
        tag.c_str(), pe.half[0], pe.half[1], pe.half[2], pe.half[3],
        static_cast<long long>(pe.triangles), static_cast<long long>(pe.unknowns), pe.dense_gb,
        c.pml_solver == PmlSolver::Lu ? "matrix + LU copy" : "matrix + Krylov", pe.peak_gb);
    std::fflush(stdout);
    Real box_peak = 0.0;
    if (o.route == "box" || o.estimate_only) {
        const BoxEstimate be = estimate_box(c, box);
        box_peak = be.peak_gb();
        std::printf(
            "ESTIMATE | %s | box | depth %.4g m | fine band %.4g m | triangles %lld (top %lld) | "
            "%s | Krylov at max-iter %d: %.2f GB | peak estimate %.2f GB\n",
            tag.c_str(), box.depth, box.params.box_fine_depth.value_or(0.0),
            static_cast<long long>(box.mesh.num_triangles()),
            static_cast<long long>(box.top_triangles), mlfmm_scaling::summary(be.mlfmm).c_str(),
            c.box_max_iter, be.krylov_gb, box_peak);
        std::fflush(stdout);
    }
    if (o.route == "pml" || o.estimate_only) {
        const PmlChecks k = pml_checks(c, map);
        std::printf("DARK | %s | ratio %.3e (tolerance %.1e) | masked %lld of %lld functions\n",
                    tag.c_str(), k.dark_band, c.dark_band_tolerance,
                    static_cast<long long>(k.masked), static_cast<long long>(k.functions));
        std::fflush(stdout);
    }
    if (o.estimate_only)
        return 0;
    check_memory(o.route == "pml" ? pe.peak_gb : box_peak, avail, o.max_gb, o.force);
    RouteResult r = o.route == "pml" ? run_pml(c, map) : run_box(c, map);
    r.peak_estimate_gb = o.route == "pml" ? pe.peak_gb : box_peak;
    emit(o.summary, row_line(c, r, tag));
    std::printf("%s\n", r.report.c_str());
    std::fflush(stdout);
    if (!r.converged)
        throw std::runtime_error("the solve did not converge; nothing stored or compared");
    if (!o.out.empty())
        store(o.out, c, r, tag);
    for (const std::string& ref_dir : o.references) {
        const StoredRun ref = load(ref_dir);
        const StoredRun self = load(o.out);  // the stored attributes carry the shared keys
        emit(o.summary, compare_line(tag, ref.tag, compare_stored(self, ref), kTolerances));
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        return o.compare.empty() ? run(o) : run_compare(o);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "pml_si_crosscheck: %s\n", ex.what());
        return 1;
    }
}
