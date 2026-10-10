// Flat-interface limit (WP22c, former WP12; docs/01 "Flat interface limit", docs/05 row "Flat box,
// tapered beam, 0 deg and 45 deg | Fresnel r_p, r_s | rel on |r|^2 | < 1 %"): reflected power
// fraction of a flat patch (sigma = 0) closed by the ADR 0006 box under the rigorous
// angular-spectrum beam, against the beam reflectance R_beam of an infinite interface and the
// plane-wave Fresnel |r|^2 at the central angle (tests/support/fresnel_flat_support.hpp).
//
// One case per process (a background command is limited in time and the peak memory of a case
// should be its own): Ag (eps_r = -9.794 - 0.313j) or Si (18.478 - 0.606j) at lambda = 500 nm in
// vacuum, patch edge --L (multiple of the 50 nm spacing), waist --w0 (<= L / 4), incidence --theta
// [deg] about y from z < 0, polarisation p or s; Simulation with compression "mlfmm" (d0 =
// --digits), formulation per formulation::recommend, full GMRES (--tol). Output (stdout): SETUP
// (mesh, octree and memory estimate, beam reference), the solve and a RESULT line; the RESULT line
// is also appended to --summary FILE. Record: benchmarks/results/fresnel_flat.md.
//
// --estimate-only prints the mesh, octree and memory estimate and the beam reference (R_beam,
// Fresnel, power() vs flux, edge loss), without assembly. Without --force a case whose setup peak
// estimate plus the Krylov basis at --max-iter plus a 10 GB margin exceeds the available memory is
// refused.
//
// Usage: specklebem_fresnel_flat --material ag|si [--L 4e-6] [--w0 1e-6] [--theta 0] [--pol p|s]
//            [--mesh-size 50e-9] [--digits 3] [--tol 1e-4] [--max-iter 6000] [--restart 0]
//            [--ff-dtheta 1]
//            [--formulation pmchwt|ictf|mctf] [--jacobi 0|1] [--summary FILE] [--estimate-only]
//            [--force]
#include "specklebem/formulation/formulation.hpp"

#include <chrono>
#include <cstdio>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>

#include "fresnel_flat_support.hpp"

namespace {

using namespace fresnel_flat;

struct Options {
    Case c;
    std::string summary;
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
        } else if (a == "--w0") {
            o.c.waist = std::stod(value());
        } else if (a == "--theta") {
            o.c.theta_deg = std::stod(value());
        } else if (a == "--pol") {
            const std::string p = value();
            if (p != "p" && p != "s")
                throw std::invalid_argument("--pol must be p or s");
            o.c.pol = p == "p" ? excitation::Polarization::P : excitation::Polarization::S;
        } else if (a == "--digits") {
            o.c.digits = std::stod(value());
        } else if (a == "--tol") {
            o.c.tolerance = std::stod(value());
        } else if (a == "--max-iter") {
            o.c.max_iter = std::stoi(value());
        } else if (a == "--restart") {
            o.c.restart = std::stoi(value());
        } else if (a == "--mesh-size") {
            o.c.mesh_size = std::stod(value());
        } else if (a == "--ff-dtheta") {
            o.c.ff_dtheta_deg = std::stod(value());
        } else if (a == "--formulation") {
            o.c.formulation = formulation::parse_kind(value());
        } else if (a == "--jacobi") {
            o.c.jacobi = std::stoi(value()) != 0;
        } else if (a == "--summary") {
            o.summary = value();
        } else if (a == "--estimate-only") {
            o.estimate_only = true;
        } else if (a == "--force") {
            o.force = true;
        } else {
            throw std::invalid_argument("unknown option " + a);
        }
    }
    object_material(o.c.material);  // validates the name
    return o;
}

void append_summary(const std::string& path, const std::string& line) {
    if (path.empty())
        return;
    std::ofstream f(path, std::ios::app | std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    f << line << '\n';
}

/// One machine-readable line of a solved case.
std::string result_line(const Case& c, const Result& r) {
    char buf[1600];
    std::snprintf(
        buf, sizeof(buf),
        "RESULT | %s | L %.4g | w0 %.4g | theta %.4g | pol %s | 2N %lld | it %d | converged %s | "
        "true_res %.3e | R_sim %.7f | R_beam %.7f | F %.7f | err_beam %+.5f %% | err_F %+.5f %% "
        "| R_beam/F-1 %+.5f %% | edge_loss %.3e | hemi_grid %+.2e | hemi_deg8 %+.2e | cone %.6f | "
        "grazing %.2e | A_sim %.6f | 1-R-A %+.3e | "
        "P_fwd/P_inc %.6f | flux/power-1 %+.2e | assembly %.1f s | solve %.1f s | post %.1f s | "
        "near %.2f GB | far %.2f GB | krylov %.2f GB | peak %.2f GB",
        c.material.c_str(), c.L, c.waist, c.theta_deg,
        c.pol == excitation::Polarization::P ? "p" : "s", static_cast<long long>(r.est.unknowns),
        r.iterations, r.converged ? "yes" : "no", r.true_residual, r.r_sim(),
        r.ref.beam.reflectance, r.ref.beam.fresnel_central, 100.0 * r.error_beam(),
        100.0 * r.error_fresnel(),
        100.0 * (r.ref.beam.reflectance / r.ref.beam.fresnel_central - 1.0), r.ref.edge_loss,
        r.p_refl_grid / r.p_refl - 1.0, r.p_refl_deg / r.p_refl - 1.0, r.refl.cone / r.p_refl,
        r.refl.grazing / r.p_refl, r.a_sim(), 1.0 - r.r_sim() - r.a_sim(), r.p_fwd / r.ref.power,
        r.ref.flux.square / r.ref.power - 1.0, r.assembly_s, r.solve_s, r.post_s, gb(r.near_bytes),
        gb(r.far_bytes), gb(r.krylov_bytes), gb(r.peak_rss));
    return buf;
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
        const Geometry geo = make_geometry(o.c);
        const auto tb = std::chrono::steady_clock::now();
        const auto beam = make_beam(o.c, geo.r_max);
        const Real build_s =
            std::chrono::duration<Real>(std::chrono::steady_clock::now() - tb).count();
        const Estimate e = estimate(geo, o.c, beam);
        std::printf("SETUP | %s\n", summary(e).c_str());
        std::fflush(stdout);
        if (o.estimate_only) {
            std::printf("%s\n", summary(reference(o.c, *beam, build_s)).c_str());
            return 0;
        }
        const Real need = e.peak_setup_bytes + e.krylov_max_bytes;
        const Real avail = system_memory::available_memory_bytes();
        if (!o.force && !(avail > need + 10e9)) {
            std::fprintf(stderr,
                         "fresnel_flat: estimated peak %.1f GB (setup %.1f + Krylov at max_iter "
                         "%.1f) + 10 GB margin exceeds the available %.1f GB (--force to run)\n",
                         need / 1e9, e.peak_setup_bytes / 1e9, e.krylov_max_bytes / 1e9,
                         avail / 1e9);
            return 2;
        }
        const Result r = run(o.c, geo, beam, build_s);
        const std::string line = result_line(o.c, r);
        std::printf("%s\n%s\n\n%s\n", line.c_str(), summary(r).c_str(), r.report.c_str());
        std::fflush(stdout);
        append_summary(o.summary, line);
        return r.converged ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fresnel_flat: %s\n", ex.what());
        return 1;
    }
}
