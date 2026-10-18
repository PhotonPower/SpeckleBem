// Fast path of the PML route vs dense (WP-PML5a, validation 7h; ADR 0008 amendment 2026-10-18,
// docs/05 "PML route" row 7h): one case of tests/support/pml_ring_support.hpp per process. The
// fast run (Simulation with compression "mlfmm" on the PML route: MLFMM on the real functions plus
// the exact PML ring, GMRES) comes first, then the dense run (DenseStrategy, LU in place) with the
// matvec comparison of the stored fast products, R and the windowed far field (L2 metric of
// pml_rough_ag::compare). --fast-only skips the dense run (sizes beyond the dense budget).
// --estimate-only builds only the mesh and the octree and prints the PML ring's pre-assembly sizes
// (mlfmm::estimate_pml_ring) and the dense size: the memory model of WP-PML5a and its projection.
//
// Output (stdout): SETUP and ESTIMATE lines, the ROW line (pml_ring::row_line; also appended to
// --summary FILE), the fast Simulation report. Record: benchmarks/results/pml_ring.md.
//
// Usage: specklebem_pml_ring [--case u|flat_u|flat_ci] [--tag NAME] [--material ag|si]
//            [--patch L] [--map-patch L] [--taper W] [--gap W] [--pml W] [--h H] [--sigma S]
//            [--rms SIGMA (0: flat)] [--lc LC] [--w0 W0] [--dark-band-tol T] [--d0 3]
//            [--tol 1e-7] [--n-theta 90] [--n-phi 180] [--summary FILE] [--fast-only]
//            [--estimate-only] [--force]
// --case must come first (it resets the case).
#include "specklebem/compression/mlfmm/pml_ring.hpp"

#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"

#include <cstdio>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>

#include "pml_ring_support.hpp"

namespace {

using namespace pml_ring;

struct Options {
    RoughCase c = pml_rough_ag::u_case();
    FastSettings f;
    std::string summary;
    bool fast_only = false;
    bool estimate_only = false;
    bool force = false;
};

Options parse(int argc, char** argv) {
    Options o;
    o.c.tag = "u";
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
            const std::string name = value();
            if (name == "u") {
                o.c = pml_rough_ag::u_case();
            } else if (name == "flat_u") {  // flat Ag with the zones of U
                o.c = pml_rough_ag::u_case();
                o.c.tag = "flat_u";
                o.c.rms = 0.0;
                o.c.correlation_length = 0.0;
                o.c.map_patch = o.c.patch;
            } else if (name == "flat_ci") {
                o.c = flat_ag_ci_case();
            } else {
                throw std::invalid_argument("unknown --case " + name);
            }
        } else if (a == "--tag") {
            o.c.tag = value();
        } else if (a == "--material") {
            o.c.material = value();
        } else if (a == "--patch") {
            o.c.patch = real();
        } else if (a == "--map-patch") {
            o.c.map_patch = real();
        } else if (a == "--taper") {
            o.c.taper = real();
        } else if (a == "--gap") {
            o.c.gap = real();
        } else if (a == "--pml") {
            o.c.pml_width = real();
        } else if (a == "--h") {
            o.c.mesh_size = real();
        } else if (a == "--sigma") {
            o.c.sigma_max = real();
        } else if (a == "--rms") {
            o.c.rms = real();
        } else if (a == "--lc") {
            o.c.correlation_length = real();
        } else if (a == "--w0") {
            o.c.waist = real();
        } else if (a == "--dark-band-tol") {
            o.c.dark_band_tolerance = real();
        } else if (a == "--d0") {
            o.f.digits = real();
        } else if (a == "--tol") {
            o.f.gmres_tolerance = real();
        } else if (a == "--n-theta") {
            o.c.n_theta = std::stoi(value());
        } else if (a == "--n-phi") {
            o.c.n_phi = std::stoi(value());
        } else if (a == "--summary") {
            o.summary = value();
        } else if (a == "--fast-only") {
            o.fast_only = true;
        } else if (a == "--estimate-only") {
            o.estimate_only = true;
        } else if (a == "--force") {
            o.force = true;
        } else {
            throw std::invalid_argument("unknown option " + a);
        }
    }
    return o;
}

void append_summary(const std::string& path, const std::string& line) {
    if (path.empty())
        return;
    std::ofstream f(core::path_from_utf8(path), std::ios::app | std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    f << line << '\n';
}

/// Octree and ring estimate of the fast path (the MlfmmOperator's tree: leaf rule of d0).
std::string estimate_line(const RoughCase& c, const FastSettings& f) {
    const geometry::OpenSurfaceMesh om = pml_rough_ag::make_open_mesh(c);
    const basis::RwgSpace space(om.mesh);
    const geometry::CoordinateStretch st = geometry::make_pml_stretch(om, c.sigma_max, c.order);
    mlfmm::MlfmmParams p;
    p.accuracy_digits = f.digits;
    const mlfmm::Octree tree = mlfmm::make_octree(space, pml_rough_ag::kLambda, p);
    const std::vector<std::uint8_t> s = mlfmm::classify_stretched(space, st);
    const mlfmm::PmlRingEstimate e = mlfmm::estimate_pml_ring(tree, space, s);
    char buf[800];
    std::snprintf(
        buf, sizeof(buf),
        "ESTIMATE | %s | L %.4g | pml %.4g | h %.4g | d0 %.1f | 2N %lld | stretched %lld (%.1f %%) "
        "| levels %d | leaf %.3f lambda | row leaves %lld | ring near %lld pairs %.1f MB | ring "
        "far "
        "%lld pairs (%lld stored) %.1f MB | rr near %lld pairs %.1f MB | dense %.1f MB | ring / "
        "dense %.3f",
        c.tag.c_str(), c.patch, c.pml_width, c.mesh_size, f.digits, static_cast<long long>(2 * e.n),
        static_cast<long long>(e.stretched),
        100.0 * static_cast<Real>(e.stretched) / static_cast<Real>(e.n), tree.levels(),
        tree.box_size(tree.leaf_level()) / pml_rough_ag::kLambda,
        static_cast<long long>(e.row_leaves), static_cast<long long>(e.near_pairs),
        static_cast<Real>(e.near_bytes) / 1e6, static_cast<long long>(e.far_pairs),
        static_cast<long long>(e.far_stored), static_cast<Real>(e.far_bytes) / 1e6,
        static_cast<long long>(e.rr_near_pairs), static_cast<Real>(4 * e.rr_near_pairs * 24) / 1e6,
        static_cast<Real>(e.dense_bytes) / 1e6,
        static_cast<Real>(e.near_bytes + e.far_bytes) / static_cast<Real>(e.dense_bytes));
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        const RoughCase& c = o.c;
        const pml_rough_ag::Size s = pml_rough_ag::size_of(c);
        const Real avail = system_memory::available_memory_bytes();
        std::printf(
            "SETUP | %s | %s | rms %.3g | L %.4g | half widths %.4g / %.4g / %.4g / %.4g m | "
            "triangles %lld | 2N %lld | dense %.2f GB | available %.1f GB\n",
            c.tag.c_str(), c.material.c_str(), c.rms, c.patch, s.half[0], s.half[1], s.half[2],
            s.half[3], static_cast<long long>(s.triangles), static_cast<long long>(s.unknowns),
            s.dense_bytes / 1e9, avail / 1e9);
        const std::string est = estimate_line(c, o.f);
        std::printf("%s\n", est.c_str());
        std::fflush(stdout);
        append_summary(o.summary, est);
        if (o.estimate_only)
            return 0;
        if (!o.fast_only && !o.force && !(avail > s.dense_bytes + 4e9)) {
            std::fprintf(stderr,
                         "pml_ring: dense %.1f GB + 4 GB margin exceeds the available %.1f GB "
                         "(--fast-only or --force)\n",
                         s.dense_bytes / 1e9, avail / 1e9);
            return 2;
        }
        const FastRun fr = run_fast(c, o.f);
        std::printf("%s\n", fr.report.c_str());
        std::fflush(stdout);
        if (o.fast_only) {
            char buf[600];
            std::snprintf(
                buf, sizeof(buf),
                "FAST | %s | d0 %.1f | 2N %lld | stretched %lld | levels %d | R %.8f | "
                "its %d | true res %.2e | op %.1f MB (ring near %.1f, far %.1f) | "
                "assemble (op + rhs + diag) %.1f s | GMRES %.1f s | matvec %.3f s | peak %.2f GB",
                c.tag.c_str(), o.f.digits, static_cast<long long>(fr.unknowns),
                static_cast<long long>(fr.stretched), fr.levels, fr.far.reflectance(),
                fr.iterations, fr.true_residual, static_cast<Real>(fr.operator_bytes) / 1e6,
                static_cast<Real>(fr.ring_near_bytes) / 1e6,
                static_cast<Real>(fr.ring_far_bytes) / 1e6, fr.assemble_s, fr.solve_s, fr.matvec_s,
                fr.peak_rss / 1e9);
            std::printf("%s\n", buf);
            append_summary(o.summary, buf);
            return 0;
        }
        const DenseRun dr = run_dense(c, fr);
        const std::string line = row_line(c, o.f, fr, dr);
        std::printf("%s\n", line.c_str());
        append_summary(o.summary, line);
        const pml_rough_ag::Metrics m = pml_rough_ag::compare(fr.far, dr.far);
        const std::string cmp = pml_rough_ag::compare_line(fr.far, dr.far, m);
        std::printf("%s\n", cmp.c_str());
        append_summary(o.summary, cmp);
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "pml_ring: %s\n", ex.what());
        return 1;
    }
}
