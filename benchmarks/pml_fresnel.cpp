// Validation 7a of the open-surface PML route (WP-PML3b, ADR 0009; docs/05 "PML route"): flat Ag
// (eps_r = -9.794 - 0.313j) or Si (18.478 - 0.606j) open surface at lambda = 500 nm in vacuum,
// patch / taper / gap / PML zones (height map zero; the taper is still a zone of the open mesh),
// complex-coordinate-stretching PML (sigma_max, order), rigorous angular-spectrum beam at
// incidence --theta [deg] about y from z < 0, dense operator, recommended formulation, direct LU
// (LAPACK zgetrf when available; one factorisation for all requested polarisations). The
// windowed reflected power (far field of the real part, raised-cosine window across the gap,
// Simulation::far_field_options(), or --window-inner/--window-outer) over the beam power against
// the beam reflectance R_beam of the same beam on the infinite interface
// (tests/support/pml_fresnel_support.hpp, fresnel_flat_support.hpp).
//
// Output (stdout): SETUP (zones, 2N, dense memory), one ROW line per polarisation (also appended
// to --summary FILE) and the Simulation report. Record: benchmarks/results/pml_validation.md.
// One run per process; --estimate-only prints the setup and the dark-band ratios without
// assembling. Without --force a
// run whose matrix plus LU copy (32 (2N)^2 bytes) plus a 4 GB margin exceeds the available memory
// is refused.
//
// Usage: specklebem_pml_fresnel --material si|ag [--theta 0] [--pol p|s|both] [--w0 0.5e-6]
//            [--patch 1.5e-6] [--taper 0.5e-6] [--gap 0.5e-6] [--pml 1e-6] [--h 125e-9]
//            [--sigma 4] [--order 2] [--dark-band-tol 1e-4] [--acc 1e-5] [--stp 10]
//            [--window-inner X --window-outer Y] [--ff-dtheta 1] [--ff-degree 6] [--eigen-lu]
//            [--summary FILE] [--estimate-only] [--force]
#include <cstdio>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pml_fresnel_support.hpp"

namespace {

using namespace pml_fresnel;

struct Options {
    PmlCase c;
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
        } else if (a == "--theta") {
            o.c.theta_deg = std::stod(value());
        } else if (a == "--pol") {
            const std::string p = value();
            if (p == "p")
                o.c.pols = {excitation::Polarization::P};
            else if (p == "s")
                o.c.pols = {excitation::Polarization::S};
            else if (p == "both")
                o.c.pols = {excitation::Polarization::P, excitation::Polarization::S};
            else
                throw std::invalid_argument("--pol must be p, s or both");
        } else if (a == "--w0") {
            o.c.waist = std::stod(value());
        } else if (a == "--patch") {
            o.c.patch = std::stod(value());
        } else if (a == "--taper") {
            o.c.taper = std::stod(value());
        } else if (a == "--gap") {
            o.c.gap = std::stod(value());
        } else if (a == "--pml") {
            o.c.pml_width = std::stod(value());
        } else if (a == "--h") {
            o.c.mesh_size = std::stod(value());
        } else if (a == "--sigma") {
            o.c.sigma_max = std::stod(value());
        } else if (a == "--order") {
            o.c.order = std::stod(value());
        } else if (a == "--dark-band-tol") {
            o.c.dark_band_tolerance = std::stod(value());
        } else if (a == "--acc") {
            o.c.target_accuracy = std::stod(value());
        } else if (a == "--stp") {
            o.c.stretched_touching_points = std::stoi(value());
        } else if (a == "--window-inner") {
            o.c.window_inner = std::stod(value());
        } else if (a == "--window-outer") {
            o.c.window_outer = std::stod(value());
        } else if (a == "--ff-dtheta") {
            o.c.ff_dtheta_deg = std::stod(value());
        } else if (a == "--ff-degree") {
            o.c.ff_degree = std::stoi(value());
        } else if (a == "--eigen-lu") {
            o.c.use_lapack = false;
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
    fresnel_flat::object_material(o.c.material);  // validates the name
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

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        const Size s = size_of(o.c);
        const Real avail = system_memory::available_memory_bytes();
        std::printf(
            "SETUP | %s theta %g deg | half widths %.4g / %.4g / %.4g / %.4g m | triangles %lld | "
            "2N %lld | dense %.2f GB, peak (matrix + LU copy) %.2f GB | available %.1f GB\n",
            o.c.material.c_str(), o.c.theta_deg, s.half[0], s.half[1], s.half[2], s.half[3],
            static_cast<long long>(s.triangles), static_cast<long long>(s.unknowns),
            s.dense_bytes / 1e9, s.peak_bytes() / 1e9, avail / 1e9);
        std::fflush(stdout);
        if (o.estimate_only) {
            const std::vector<Real> dark = dark_band_ratios(o.c);
            for (std::size_t i = 0; i < dark.size(); ++i)
                std::printf("DARK | pol %s | ratio %.3e (tolerance %.1e)\n", pol_name(o.c.pols[i]),
                            dark[i], o.c.dark_band_tolerance);
            return 0;
        }
        if (!o.force && !(avail > s.peak_bytes() + 4e9)) {
            std::fprintf(stderr,
                         "pml_fresnel: peak %.1f GB + 4 GB margin exceeds the available %.1f GB "
                         "(--force to run)\n",
                         s.peak_bytes() / 1e9, avail / 1e9);
            return 2;
        }
        const RunResult r = run(o.c);
        for (const PolResult& p : r.pols) {
            const std::string line = row_line(o.c, r, p);
            std::printf("%s\n", line.c_str());
            append_summary(o.summary, line);
        }
        std::printf("formulation %s, setup %.1f s\n%s\n", r.formulation.c_str(), r.setup_s,
                    r.report.c_str());
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "pml_fresnel: %s\n", ex.what());
        return 1;
    }
}
