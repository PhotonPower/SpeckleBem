// Ag sphere with MLFMM vs Mie (WP22a, Phase 4 definition of done, docs/01 and docs/05 row "Ag
// sphere d = 4 um, lambda/27, MLFMM": eps_rr <= 0.5 % in both scattering planes; Fu et al. 2023:
// 0.26 % / 0.37 %, 393 216 unknowns, preconditioned, 424 iterations).
//
// One case per process (a background command is limited in time, and the peak memory of a case
// should be its own): sphere of diameter --d in vacuum, lambda = 500 nm, Ag eps_r = -9.794 -
// 0.313j, plane wave +z, E along x; mesh "octa" (octahedron-based, 8 4^n triangles, the paper's
// mesh for n = 7) or "ico" (icosphere, 20 4^n); Simulation with compression "mlfmm" (d0 = --digits,
// automatic leaf rule and exact-part budget unless --exact-budget-gb), formulation per
// formulation::recommend (ICTF + left Jacobi for Ag) unless --formulation / --jacobi, full GMRES
// with tolerance --tol. Output (stdout): the setup, eps_rr in the xz- and yz-planes (181 and 1801
// angles), iterations, timings, the memory per component and the peak working set, then the
// operator description and the Simulation report. Record:
// benchmarks/results/ag_sphere_4um_mlfmm.md.
//
// --estimate-only prints the mesh, the octree after the leaf rule and the near-field estimate
// (no assembly; seconds).
//
// Usage: specklebem_ag_sphere_mlfmm [--mesh octa|ico] [--n 7] [--d 4e-6] [--digits 3]
//            [--tol 1e-3] [--max-iter 3000] [--exact-budget-gb G] [--formulation pmchwt|ictf|mctf]
//            [--jacobi 0|1] [--estimate-only]
#include "specklebem/formulation/formulation.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

#include "ag_sphere_mlfmm_support.hpp"

namespace {

using namespace ag_sphere_mlfmm;

struct Options {
    Case c;
    bool estimate_only = false;
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
        if (a == "--mesh") {
            o.c.mesh = value();
        } else if (a == "--n") {
            o.c.subdivisions = std::stoi(value());
        } else if (a == "--d") {
            o.c.diameter = std::stod(value());
        } else if (a == "--digits") {
            o.c.digits = std::stod(value());
        } else if (a == "--tol") {
            o.c.tolerance = std::stod(value());
        } else if (a == "--max-iter") {
            o.c.max_iter = std::stoi(value());
        } else if (a == "--exact-budget-gb") {
            o.c.max_exact_far_bytes = static_cast<std::size_t>(std::stod(value()) * 1e9);
        } else if (a == "--formulation") {
            o.c.formulation = formulation::parse_kind(value());
        } else if (a == "--jacobi") {
            o.c.jacobi = std::stoi(value()) != 0;
        } else if (a == "--estimate-only") {
            o.estimate_only = true;
        } else {
            throw std::invalid_argument("unknown option " + a);
        }
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        std::printf("%s\n", label(o.c).c_str());
        std::printf("available memory at start: %.1f GB of %.1f GB\n",
                    system_memory::available_memory_bytes() / 1e9,
                    system_memory::physical_memory_bytes() / 1e9);
        std::fflush(stdout);
        if (o.estimate_only) {
            const geometry::TriangleMesh mesh = make_mesh(o.c.mesh, o.c.diameter, o.c.subdivisions);
            std::printf("%s\n", summary(estimate(mesh, o.c)).c_str());
            return 0;
        }
        const Result r = run(o.c);
        std::printf("RESULT %s\n%s\n\n%s\n\n%s\n", label(o.c).c_str(), summary(r).c_str(),
                    r.describe.c_str(), r.report.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ag_sphere_mlfmm: %s\n", e.what());
        return 1;
    }
}
