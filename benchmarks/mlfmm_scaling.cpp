// MLFMM scaling study (WP22b1, Phase 4 definition of done in docs/01: scaling exponent gamma of
// the time per solve vs N <= 1.5 for Si rough surfaces, <= 2.0 for Ag, over N = 5e4 ... 1e6).
//
// One case per process (a background command is limited in time, and the peak memory of a case
// should be its own): Gaussian rough surface (sigma = 50 nm, Lc = 500 nm, mesh 50 nm, seed 1) on
// an L x L patch with the ADR 0006 closing box of rough_surface_box_params (coarse spacing
// --box-mesh-size, default 100 nm = lambda1 / 5; "auto": the automatic rule capped at lambda1 /
// 5; "uncapped": without the cap, the record's "auto" series; Si with the fine band 3 delta + 3
// sigma), Si or Ag at 500 nm, Gaussian beam w0 = L / 4 at normal incidence (paraxial by default,
// --beam angular-spectrum for the rigorous beam; check_beam_waist refuses w0 > L / 4 unless
// --allow-wide-beam); Simulation with compression "mlfmm" (d0 = 3, automatic leaf rule and
// exact-part budget), formulation per formulation::recommend, full GMRES with tolerance 1e-3.
// The case code is shared with the smoke test
// (tests/support/mlfmm_scaling_support.hpp). Output (stdout): the setup, a summary (2N, octree,
// setup time split, time per matvec, iterations, solve time, memory per component, peak working
// set), a machine-readable ROW line (benchmarks/mlfmm_scaling_fit.py fits the exponents), the
// operator description and the Simulation report. Record: benchmarks/results/mlfmm_scaling.md.
//
// --estimate-only prints the mesh, the octree after the leaf rule, the near-field estimate, the
// estimated Ag exact part and the far patterns and tables (no assembly; seconds to a minute).
// Without --force a case whose setup peak estimate plus 10 GB exceeds the available memory is
// refused. The estimate is computed once per process.
//
// Usage: specklebem_mlfmm_scaling --material si|ag --L 4e-6
//            [--box-mesh-size 100e-9|auto|uncapped] [--fine-band 0|1] [--mesh-size 50e-9]
//            [--seed 1] [--waist-factor 4] [--allow-wide-beam] [--beam paraxial|angular-spectrum]
//            [--digits 3] [--leaf-radius-quantile 0.99] [--tol 1e-3] [--max-iter 3000] [--restart 0] [--matvecs 5]
//            [--exact-budget-gb G] [--formulation pmchwt|ictf|mctf] [--jacobi 0|1] [--no-solve]
//            [--estimate-only] [--force]
//
// --no-solve stops after the setup and the timed matvecs (no GMRES): the matvec and setup scaling
// at sizes whose full-GMRES solve does not fit the time budget.
#include "specklebem/formulation/formulation.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

#include "mlfmm_scaling_support.hpp"

namespace {

using namespace mlfmm_scaling;

struct Options {
    Case c;
    bool estimate_only = false;
    bool force = false;
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
        if (a == "--material") {
            o.c.material = value();
        } else if (a == "--L") {
            o.c.L = std::stod(value());
        } else if (a == "--box-mesh-size") {
            const std::string v = value();
            o.c.uncapped_box = false;
            if (v == "auto") {
                o.c.box_mesh_size.reset();
            } else if (v == "uncapped") {
                o.c.box_mesh_size.reset();
                o.c.uncapped_box = true;
            } else {
                o.c.box_mesh_size = std::stod(v);
            }
        } else if (a == "--fine-band") {
            o.c.fine_band = std::stoi(value()) != 0;
        } else if (a == "--mesh-size") {
            o.c.mesh_size = std::stod(value());
        } else if (a == "--seed") {
            o.c.seed = std::stoull(value());
        } else if (a == "--waist-factor") {
            o.c.waist_factor = std::stod(value());
        } else if (a == "--allow-wide-beam") {
            o.c.allow_wide_beam = true;
        } else if (a == "--beam") {
            const std::string v = value();
            if (v == "paraxial")
                o.c.beam = Beam::paraxial;
            else if (v == "angular-spectrum")
                o.c.beam = Beam::angular_spectrum;
            else
                throw std::invalid_argument("--beam must be paraxial or angular-spectrum");
        } else if (a == "--digits") {
            o.c.digits = std::stod(value());
        } else if (a == "--leaf-radius-quantile") {
            o.c.leaf_radius_quantile = std::stod(value());
        } else if (a == "--tol") {
            o.c.tolerance = std::stod(value());
        } else if (a == "--max-iter") {
            o.c.max_iter = std::stoi(value());
        } else if (a == "--restart") {
            o.c.restart = std::stoi(value());
        } else if (a == "--matvecs") {
            o.c.matvecs = std::stoi(value());
        } else if (a == "--exact-budget-gb") {
            const double g = std::stod(value());
            if (!(g > 0.0))
                throw std::invalid_argument("--exact-budget-gb must be positive");
            o.c.max_exact_far_bytes = static_cast<std::size_t>(g * 1e9);
        } else if (a == "--formulation") {
            o.c.formulation = formulation::parse_kind(value());
        } else if (a == "--jacobi") {
            o.c.jacobi = std::stoi(value()) != 0;
        } else if (a == "--estimate-only") {
            o.estimate_only = true;
        } else if (a == "--no-solve") {
            o.c.solve = false;
        } else if (a == "--force") {
            o.force = true;
        } else {
            throw std::invalid_argument("unknown option " + a);
        }
    }
    if (o.c.matvecs < 1)
        throw std::invalid_argument("--matvecs must be >= 1");
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        std::printf("%s\n", label(o.c).c_str());
        const Real available = system_memory::available_memory_bytes();
        std::printf("available memory at start: %.1f GB of %.1f GB\n", available / 1e9,
                    system_memory::physical_memory_bytes() / 1e9);
        std::fflush(stdout);
        const Geometry geo = make_geometry(o.c);
        const Estimate e = estimate(geo, o.c);
        if (o.estimate_only) {
            std::printf("ESTIMATE %s\n%s\n", label(o.c).c_str(), summary(e).c_str());
            std::printf(
                "EST material=%s L_um=%g box=%s N2=%lld levels=%d leaf_nm=%g near_gb=%g "
                "exact_est_gb=%g pattern_gb=%g elevated_pattern_gb=%g far_table_gb=%g "
                "peak_setup_est_gb=%g lq=%g elevated=%lld\n",
                o.c.material.c_str(), o.c.L * 1e6, box_option(o.c).c_str(),
                static_cast<long long>(e.unknowns), e.levels, e.leaf_edge * 1e9, gb(e.near_bytes),
                gb(e.exact_bytes), gb(e.pattern_bytes), gb(e.elevated_pattern_bytes),
                gb(e.far_table_bytes), gb(e.peak_setup_bytes), o.c.leaf_radius_quantile,
                static_cast<long long>(e.elevated));
            return 0;
        }
        if (!o.force && available > 0.0 && e.peak_setup_bytes + 10e9 > available) {
            std::fprintf(stderr,
                         "mlfmm_scaling: setup peak estimate %.1f GB + 10 GB margin exceeds the "
                         "available %.1f GB; wait or pass --force\n",
                         gb(e.peak_setup_bytes), available / 1e9);
            return 2;
        }
        const Result r = run(geo, o.c, e);
        std::printf("RESULT %s\n%s\n%s\n\n%s\n\n%s\n", label(o.c).c_str(), summary(r).c_str(),
                    row(o.c, r).c_str(), r.describe.c_str(), r.report.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mlfmm_scaling: %s\n", e.what());
        return 1;
    }
}
