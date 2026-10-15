// Validation 7b of the open-surface PML route (WP-PML4a, ADR 0009 decision 7b; docs/05 "PML
// route", row 7b): reflection of a surface plasmon polariton at the PML start. Flat Ag
// (eps_r = -9.794 - 0.313j) open surface at lambda = 500 nm in vacuum (patch / taper / gap / PML
// zones), a small Gaussian groove at the origin under the rigorous angular-spectrum beam at normal
// incidence (E along x); the grooved minus the flat solution launches the SPP along +-x, and a
// localized source (the beam's right-hand side restricted to a disc around the origin, flat
// operator) launches it a second way. J_x, M_y and div_s J are sampled along the x-axis and an
// outgoing plus a reflected cylindrical SPP wave (plus outgoing background terms) are fitted
// between rho_min and the PML start; |r_spp| = |in / out| at the PML start
// (tests/support/pml_spp_support.hpp).
//
// Output (stdout): SETUP (zones, 2N, dense memory), one ROW line per source (also appended to
// --summary FILE), one FIT line per source, quantity and half-axis, one ZOUT line per source and
// half-axis (|out(M_y) / out(J_x)| against the SPP's |E_x / H_y|, pml_spp::out_impedance), and the
// Simulation report. --refit prints REFIT and ZOUT lines.
// Record: benchmarks/results/pml_validation.md ("7b SPP absorption"). One run per process;
// --estimate-only prints the setup without assembling. Without --force a run whose matrix
// (16 (2N)^2 bytes, LU in place) plus a 10 GB margin exceeds the available memory is refused.
//
// Usage: specklebem_pml_spp [--launch groove|source] [--patch 1e-6] [--taper 0.5e-6]
//            [--pml-start 2e-6] [--pml 1e-6] [--h 8.333e-8] [--sigma 4] [--order 2]
//            [--groove-depth 50e-9] [--groove-radius 100e-9] [--source-radius 150e-9]
//            [--w0 0.5e-6] [--dark-band-tol 1e-2] [--acc 1e-5] [--stp 10] [--rho-min 0.6e-6]
//            [--bg 1,1.5,2] [--fit-imag] [--eigen-lu] [--dump FILE.csv] [--summary FILE]
//            [--refit FILE.csv (fit a dump again with these options; plane = --pml-start)]
//            [--estimate-only] [--force]
#include <algorithm>
#include <cstdio>
#include <exception>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pml_spp_support.hpp"

namespace {

using namespace pml_spp;

struct Options {
    SppCase c;
    std::string summary;
    std::string dump;
    std::string refit;
    bool estimate_only = false;
    bool force = false;
};

std::vector<Real> parse_list(const std::string& s) {
    std::vector<Real> out;
    if (s == "none")
        return out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(std::stod(item));
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
        if (a == "--launch") {
            const std::string l = value();
            if (l == "groove")
                o.c.launch = Launch::Groove;
            else if (l == "source")
                o.c.launch = Launch::Source;
            else
                throw std::invalid_argument("--launch must be groove or source");
        } else if (a == "--patch") {
            o.c.patch = std::stod(value());
        } else if (a == "--taper") {
            o.c.taper = std::stod(value());
        } else if (a == "--pml-start") {
            o.c.pml_start = std::stod(value());
        } else if (a == "--pml") {
            o.c.pml_width = std::stod(value());
        } else if (a == "--h") {
            o.c.mesh_size = std::stod(value());
        } else if (a == "--sigma") {
            o.c.sigma_max = std::stod(value());
        } else if (a == "--order") {
            o.c.order = std::stod(value());
        } else if (a == "--groove-depth") {
            o.c.groove_depth = std::stod(value());
        } else if (a == "--groove-radius") {
            o.c.groove_radius = std::stod(value());
        } else if (a == "--source-radius") {
            o.c.source_radius = std::stod(value());
        } else if (a == "--w0") {
            o.c.waist = std::stod(value());
        } else if (a == "--dark-band-tol") {
            o.c.dark_band_tolerance = std::stod(value());
        } else if (a == "--acc") {
            o.c.target_accuracy = std::stod(value());
        } else if (a == "--stp") {
            o.c.stretched_touching_points = std::stoi(value());
        } else if (a == "--rho-min") {
            o.c.rho_min = std::stod(value());
        } else if (a == "--bg") {
            o.c.background_powers = parse_list(value());
        } else if (a == "--fit-imag") {
            o.c.fit_imag = true;
        } else if (a == "--eigen-lu") {
            o.c.use_lapack = false;
        } else if (a == "--dump") {
            o.dump = value();
        } else if (a == "--summary") {
            o.summary = value();
        } else if (a == "--estimate-only") {
            o.estimate_only = true;
        } else if (a == "--refit") {
            o.refit = value();
        } else if (a == "--force") {
            o.force = true;
        } else {
            throw std::invalid_argument("unknown option " + a);
        }
    }
    return o;
}

/// ZOUT lines: |out(M_y) / out(J_x)| per source (in order of appearance) and half-axis.
void print_out_impedance(const std::vector<QuantityFit>& fits, Real k0, Complex k_spp) {
    const Real z_spp = spp_impedance(k0, k_spp);
    std::vector<std::string> sources;
    for (const QuantityFit& f : fits)
        if (std::find(sources.begin(), sources.end(), f.source) == sources.end())
            sources.push_back(f.source);
    for (const std::string& src : sources) {
        for (const int side : {1, -1}) {
            const Real z = out_impedance(fits, src, side);
            std::printf(
                "ZOUT | %s | side %+d | |out My/Jx| %.2f Ohm | SPP |Ex/Hy| %.2f Ohm | "
                "ratio %.3f\n",
                src.c_str(), side, z, z_spp, z / z_spp);
        }
    }
}

void append_summary(const std::string& path, const std::string& line) {
    if (path.empty())
        return;
    std::ofstream f(core::path_from_utf8(path), std::ios::app | std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    f << line << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        if (!o.refit.empty()) {
            // Refit a dump with this run's fit options (plane = --pml-start); no solve.
            const Real k0 = 2.0 * constants::pi / kLambda;
            const Complex k_spp =
                spp_wavenumber(k0, Complex(1.0, 0.0), material::silver_500nm().eps_r);
            std::vector<QuantityFit> fits;
            for (const auto& [src, d] : read_dump(o.refit))
                fit_all(o.c, src, d, o.c.pml_start, k0, k_spp, fits);
            for (const QuantityFit& f : fits) {
                if (f.side != 1)
                    continue;
                std::printf(
                    "REFIT | %s | %s | plane %.4g | rho_min %.3g | n %lld | |r| %.4e | arg %+.4f | "
                    "k/kspp %.6f%+.6fj | |out| %.4e | res %.3e\n",
                    f.source.c_str(), f.quantity.c_str(), o.c.pml_start, o.c.rho_min,
                    static_cast<long long>(f.fit.points), std::abs(f.fit.r()), std::arg(f.fit.r()),
                    (f.fit.k / k_spp).real(), (f.fit.k / k_spp).imag(), std::abs(f.fit.out),
                    f.fit.residual);
            }
            print_out_impedance(fits, k0, k_spp);
            return 0;
        }
        const Size s = size_of(o.c);
        const Real avail = system_memory::available_memory_bytes();
        std::printf(
            "SETUP | spp %s | half widths %.4g / %.4g / %.4g / %.4g m | triangles %lld | 2N %lld "
            "| dense %.2f GB (LU in place) | available %.1f GB\n",
            o.c.launch == Launch::Groove ? "groove" : "source", s.half[0], s.half[1], s.half[2],
            s.half[3], static_cast<long long>(s.triangles), static_cast<long long>(s.unknowns),
            s.dense_bytes / 1e9, avail / 1e9);
        std::fflush(stdout);
        if (o.estimate_only)
            return 0;
        if (!o.force && !(avail > s.dense_bytes + 10e9)) {
            std::fprintf(stderr,
                         "pml_spp: matrix %.1f GB + 10 GB margin exceeds the available %.1f GB "
                         "(--force to run)\n",
                         s.dense_bytes / 1e9, avail / 1e9);
            return 2;
        }
        const RunResult r = run(o.c, o.dump);
        std::vector<std::string> sources;
        if (o.c.launch == Launch::Groove)
            sources.emplace_back("groove");
        sources.emplace_back("local");
        for (const std::string& src : sources) {
            const std::string line = row_line(o.c, r, src);
            std::printf("%s\n", line.c_str());
            append_summary(o.summary, line);
        }
        for (const QuantityFit& f : r.fits) {
            std::printf(
                "FIT | %s | %s | side %+d | n %lld | |r| %.4e | arg %+.4f | k/kspp %.6f%+.6fj | "
                "|out| %.4e | res %.3e\n",
                f.source.c_str(), f.quantity.c_str(), f.side, static_cast<long long>(f.fit.points),
                std::abs(f.fit.r()), std::arg(f.fit.r()), (f.fit.k / r.k_spp).real(),
                (f.fit.k / r.k_spp).imag(), std::abs(f.fit.out), f.fit.residual);
        }
        print_out_impedance(r.fits, r.k0, r.k_spp);
        std::printf("formulation %s | k_spp/k0 %.6f%+.6fj\n%s\n", r.formulation.c_str(),
                    (r.k_spp / r.k0).real(), (r.k_spp / r.k0).imag(), r.report.c_str());
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "pml_spp: %s\n", ex.what());
        return 1;
    }
}
