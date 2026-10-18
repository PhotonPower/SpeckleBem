// Validations 7c and 7d of the open-surface PML route on rough Ag (WP-PML4b, ADR 0009 decision 7;
// docs/05 "PML route", rows 7c / 7d): Gaussian rough Ag patch (sigma, Lc, seed; map generated on
// --map-patch + 2 W_t and cropped centrally to --patch + 2 W_t, so that runs with different L share
// the surface), taper / gap / PML zones of geometry::make_open_surface_mesh, complex-coordinate-
// stretching PML, rigorous angular-spectrum beam at normal incidence (E along x), dense operator,
// LU in place (tests/support/pml_rough_ag_support.hpp). The far field |F|^2 of the windowed
// currents on the reflection hemisphere and the xz / yz cuts is written to --out (one file per
// window: OUT.ff for Simulation's default window, OUT.w1.ff ... for --window), and --compare
// evaluates the docs/05 box-check metric between two such files without a solve.
//
// Output (stdout): SETUP (zones, 2N, dense memory, dark band, masked functions), the ROW line (also
// appended to --summary FILE), one DRC line per window (plane of incidence), the Simulation report.
// Record: benchmarks/results/pml_rough_ag.md. One run per process; --estimate-only prints the setup
// without assembling. Without --force a run whose dense matrix (16 (2N)^2 bytes, LU in place) plus
// a 4 GB margin exceeds the available memory is refused.
//
// Usage: specklebem_pml_rough_ag [--case u|w15|s15|l12|si_u|si_l12] [--tag NAME] [--material ag|si]
//            [--patch 3.75e-6]
//            [--map-patch 4.5e-6] [--taper 0.5e-6] [--gap 0.5e-6] [--pml 1e-6] [--h 125e-9]
//            [--sigma 4] [--order 2] [--dark-band-tol 1e-4] [--acc 1e-5] [--stp 10] [--w0 0.5e-6]
//            [--rms 50e-9 (0: flat control)] [--lc 500e-9] [--seed 1] [--n-theta 90] [--n-phi 180]
//            [--ff-degree 6] [--eps-imag X (diagnostic: Im eps_r of Ag, < 0)]
//            [--window INNER,OUTER]... [--eigen-lu] [--out BASE] [--summary FILE]
//            [--estimate-only] [--scan-waists W1,W2,...] [--force]
//        specklebem_pml_rough_ag --compare TEST.ff REF.ff [--summary FILE]
// --case must come first (it resets the case); --scan-waists prints one DARK line per waist (the
// dark-band ratio of the case with that waist, no assembly) and implies --estimate-only.
#include <algorithm>
#include <cstdio>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pml_rough_ag_support.hpp"

namespace {

using namespace pml_rough_ag;

struct Options {
    RoughCase c;
    std::string out;
    std::string summary;
    std::vector<std::string> compare;
    std::vector<Real> scan_waists;  ///< --scan-waists: dark-band ratio per waist (no assembly)
    bool estimate_only = false;
    bool force = false;
};

post::CurrentWindow parse_window(const std::string& v) {
    const std::size_t comma = v.find(',');
    if (comma == std::string::npos)
        throw std::invalid_argument("--window needs INNER,OUTER");
    return {std::stod(v.substr(0, comma)), std::stod(v.substr(comma + 1))};
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
        if (a == "--case" && i != 1)
            throw std::invalid_argument("--case must be the first option (it resets the case)");
        if (a == "--case") {
            const std::string name = value();
            if (name == "u")
                o.c = u_case();
            else if (name == "w15")
                o.c = pml_width_case();
            else if (name == "s15")
                o.c = sigma_case();
            else if (name == "l12")
                o.c = patch_case();
            else if (name == "si_u")
                o.c = si_u_case();
            else if (name == "si_l12")
                o.c = si_patch_case();
            else
                throw std::invalid_argument("unknown case " + name);
        } else if (a == "--tag") {
            o.c.tag = value();
        } else if (a == "--material") {
            o.c.material = value();
            (void)object_material(o.c.material);  // validate early
        } else if (a == "--patch") {
            o.c.patch = std::stod(value());
        } else if (a == "--map-patch") {
            o.c.map_patch = std::stod(value());
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
        } else if (a == "--w0") {
            o.c.waist = std::stod(value());
        } else if (a == "--rms") {
            o.c.rms = std::stod(value());
        } else if (a == "--lc") {
            o.c.correlation_length = std::stod(value());
        } else if (a == "--seed") {
            o.c.seed = std::stoull(value());
        } else if (a == "--n-theta") {
            o.c.n_theta = std::stoi(value());
        } else if (a == "--n-phi") {
            o.c.n_phi = std::stoi(value());
        } else if (a == "--ff-degree") {
            o.c.ff_degree = std::stoi(value());
        } else if (a == "--window") {
            o.c.extra_windows.push_back(parse_window(value()));
        } else if (a == "--eps-imag") {
            o.c.eps_imag = std::stod(value());
        } else if (a == "--eigen-lu") {
            o.c.use_lapack = false;
        } else if (a == "--out") {
            o.out = value();
        } else if (a == "--summary") {
            o.summary = value();
        } else if (a == "--compare") {
            o.compare.push_back(value());
            o.compare.push_back(value());
        } else if (a == "--scan-waists") {
            const std::string v = value();
            for (std::size_t p = 0; p < v.size();) {
                const std::size_t q = std::min(v.find(',', p), v.size());
                o.scan_waists.push_back(std::stod(v.substr(p, q - p)));
                p = q + 1;
            }
            o.estimate_only = true;
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

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        if (!o.compare.empty()) {
            const FarField t = read_far_field(o.compare[0]);
            const FarField r = read_far_field(o.compare[1]);
            const std::string line = compare_line(t, r, compare(t, r));
            std::printf("%s\n", line.c_str());
            append_summary(o.summary, line);
            return 0;
        }
        const RoughCase& c = o.c;
        const Estimate e = estimate(c);
        const Real avail = system_memory::available_memory_bytes();
        std::printf(
            "SETUP | %s | L %.4g (map %.4g) | half widths %.4g / %.4g / %.4g / %.4g m | triangles "
            "%lld | 2N %lld | dense %.2f GB (LU in place) | available %.1f GB | dark %.3e (tol "
            "%.1e) | masked %lld of %lld | map rms %.4g m\n",
            c.tag.c_str(), c.patch, c.map_patch, e.size.half[0], e.size.half[1], e.size.half[2],
            e.size.half[3], static_cast<long long>(e.size.triangles),
            static_cast<long long>(e.size.unknowns), e.size.dense_bytes / 1e9, avail / 1e9,
            e.dark_band, c.dark_band_tolerance, static_cast<long long>(e.masked),
            static_cast<long long>(e.size.unknowns / 2), e.patch_rms);
        std::fflush(stdout);
        for (const Real w0 : o.scan_waists) {
            RoughCase s = c;
            s.waist = w0;
            const Estimate es = estimate(s);
            char buf[300];
            std::snprintf(buf, sizeof(buf),
                          "DARK | %s | L %.4g | gap %.4g | pml_start %.4g | w0 %.4g | pml_start / "
                          "w0 %.3f | ratio %.3e | 2N %lld | dense %.2f GB",
                          c.tag.c_str(), c.patch, c.gap, es.size.half[2], w0, es.size.half[2] / w0,
                          es.dark_band, static_cast<long long>(es.size.unknowns),
                          es.size.dense_bytes / 1e9);
            std::printf("%s\n", buf);
            std::fflush(stdout);
            append_summary(o.summary, buf);
        }
        if (o.estimate_only)
            return 0;
        if (!o.force && !(avail > e.size.dense_bytes + 4e9)) {
            std::fprintf(stderr,
                         "pml_rough_ag: dense %.1f GB + 4 GB margin exceeds the available %.1f GB "
                         "(--force to run)\n",
                         e.size.dense_bytes / 1e9, avail / 1e9);
            return 2;
        }
        const RunResult r = run(c);
        const std::string line = row_line(c, r);
        std::printf("%s\n", line.c_str());
        append_summary(o.summary, line);
        for (std::size_t i = 0; i < r.far.size(); ++i) {
            std::printf("%s | window %.4g/%.4g | R %.7f\n", drc_line(r.far[i]).c_str(),
                        r.far[i].window.inner_half, r.far[i].window.outer_half,
                        r.far[i].reflectance());
            if (!o.out.empty()) {
                const std::string path =
                    i == 0 ? o.out + ".ff" : o.out + ".w" + std::to_string(i) + ".ff";
                write_far_field(path, r.far[i]);
                std::printf("FARFIELD | %s\n", path.c_str());
            }
        }
        std::printf("formulation %s\n%s\n", r.formulation.c_str(), r.report.c_str());
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "pml_rough_ag: %s\n", ex.what());
        return 1;
    }
}
