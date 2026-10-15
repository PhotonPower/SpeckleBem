// Near field of the Ag sphere in the xz-plane vs Mie (WP24, docs/05 row "Near field Ag sphere,
// xz-plane": relative error per point < 0.1 lit, < 1 shadow; docs/01 Phase 5 DoD).
//
// One case per process: the WP22a case of ag_sphere_mlfmm_support.hpp (Ag sphere of diameter --d
// in vacuum, lambda = 500 nm, plane wave +z, E along x; mesh --mesh/--n; Simulation with MLFMM
// d0 = --digits, ICTF + left Jacobi, full GMRES with tolerance --tol) is solved, or its currents
// are read back from an earlier run (--load DIR), and the total field is evaluated on
//  * radial lines in the xz-plane (theta = 180 ... 0 deg from +z; 25 distances lambda / 100 ...
//    lambda outside, 9 distances lambda / 100 ... lambda / 5 inside) once per Dunavant degree of
//    --line-degrees (near-surface profile: quadrature vs discretisation error);
//  * the xz grid x, z in [-1.5 d, 1.5 d] (--half-factor) at spacing <= --spacing, points within
//    lambda / 100 of the surface dropped, with Dunavant degree --degree and --near-degree within
//    --near-band-h mean edges of the surface;
// and compared with Mie (ag_near_field_support.hpp: zones, metric). With --check, first the
// acceptance check of the tests (ag_near_field::check: grid x, z in [-1.5 d, 1.5 d] at spacing
// <= --spacing plus the default shells, acceptance quadrature, union statistics and d_min scan).
// Output (stdout): the setup, the CHECK summary, PROFILE lines (e on the front half z <= 0 and
// the back half z > 0 of each shell), AXIS lines (decimated grid cuts z = 0 and x = 0), ROW lines
// (per-zone statistics of the grid for each exclusion distance of --d-min-h, in units of the mean
// edge h; zones as in the tests: shadow = geometric shadow |x| < a, z > 0, lit = every other
// exterior point), timings and peak memory. With --out DIR the currents, grid and line results are
// written as .npy files (io::open_npy_directory; keep DIR outside the repository). Record:
// benchmarks/results/ag_near_field.md.
//
// Usage: specklebem_ag_near_field [--mesh octa|ico] [--n 7] [--d 4e-6] [--digits 3] [--tol 1e-3]
//            [--max-iter 3000] [--spacing 5e-8] [--half-factor 1.5] [--degree 6]
//            [--near-degree 19] [--near-band-h 3] [--line-degrees 6,12,19] [--d-min-h 0.5,1,2]
//            [--no-shells] [--no-grid] [--check] [--out DIR] [--load DIR]
#include "specklebem/basis/rwg.hpp"
#include "specklebem/core/path.hpp"
#include "specklebem/io/result_writer.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ag_box_mechanism_support.hpp"  // read_npy
#include "ag_near_field_support.hpp"

namespace {

using namespace ag_near_field;
using ag_sphere_mlfmm::Case;

struct Options {
    Case c;
    Real spacing = kLambda / 10.0;
    Real half_factor = 1.5;  ///< grid half extent / diameter
    int degree = 6;
    int near_degree = 6;
    Real near_band_h = 0.0;
    std::vector<int> line_degrees{6};
    std::vector<Real> d_min_h{0.5, 1.0, 2.0};
    bool shells = true;
    bool grid = true;
    bool check = false;
    std::string out;
    std::string load;
};

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) v.push_back(item);
    return v;
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
        } else if (a == "--spacing") {
            o.spacing = std::stod(value());
        } else if (a == "--half-factor") {
            o.half_factor = std::stod(value());
        } else if (a == "--degree") {
            o.degree = std::stoi(value());
        } else if (a == "--near-degree") {
            o.near_degree = std::stoi(value());
        } else if (a == "--near-band-h") {
            o.near_band_h = std::stod(value());
        } else if (a == "--line-degrees") {
            o.line_degrees.clear();
            for (const std::string& s : split(value())) o.line_degrees.push_back(std::stoi(s));
        } else if (a == "--d-min-h") {
            o.d_min_h.clear();
            for (const std::string& s : split(value())) o.d_min_h.push_back(std::stod(s));
        } else if (a == "--no-shells") {
            o.shells = false;
        } else if (a == "--no-grid") {
            o.grid = false;
        } else if (a == "--check") {
            o.check = true;
        } else if (a == "--out") {
            o.out = value();
        } else if (a == "--load") {
            o.load = value();
        } else {
            throw std::invalid_argument("unknown option " + a);
        }
    }
    if (!(o.spacing > 0.0) || !(o.half_factor > 0.5) || (o.shells && o.line_degrees.empty()))
        throw std::invalid_argument("invalid --spacing, --half-factor or --line-degrees");
    return o;
}

/// Problem of a stored solution (--load): the mesh is rebuilt deterministically, so the RWG
/// numbering equals that of the solved Simulation.
struct StoredProblem {
    StoredProblem(const Case& c, VectorXc currents)
        : mesh(ag_sphere_mlfmm::make_mesh(c.mesh, c.diameter, c.subdivisions)),
          space(mesh),
          wave(kLambda, Vec3::UnitZ(), Vec3c(1.0, 0.0, 0.0)) {
        problem.space = &space;
        problem.exterior = material::vacuum();
        problem.object = material::silver_500nm();
        problem.excitation = &wave;
        problem.omega = wave.omega();
        if (currents.size() != 2 * space.size())
            throw std::invalid_argument("stored currents have length " +
                                        std::to_string(currents.size()) +
                                        ", expected 2N = " + std::to_string(2 * space.size()));
        solution.problem = &problem;
        solution.currents = std::move(currents);
    }
    StoredProblem(const StoredProblem&) = delete;
    StoredProblem& operator=(const StoredProblem&) = delete;

    geometry::TriangleMesh mesh;
    basis::RwgSpace space;
    excitation::PlaneWave wave;
    op::Problem problem;
    post::SurfaceSolution solution;
};

std::string vec2(const Vec3& r) {
    std::ostringstream os;
    os << "(" << r.x() * 1e9 << ", " << r.z() * 1e9 << ") nm";
    return os.str();
}

void print_stats(const std::string& tag, const char* zone, const Stats& s, Real d_min, Real h) {
    std::printf(
        "ROW %s zone=%s d_min=%.2f nm (%.2f h, lambda/%.1f) count=%lld max=%.4g p99=%.4g "
        "p95=%.4g median=%.4g argmax=%s\n",
        tag.c_str(), zone, d_min * 1e9, d_min / h, kLambda / d_min, static_cast<long long>(s.count),
        s.max, s.p99, s.p95, s.median, vec2(s.argmax).c_str());
}

/// Real column of a full-grid quantity, NaN at the grid points that were not evaluated.
MatrixXr grid_column(const Grid& g, const VectorXr& values) {
    MatrixXr m = MatrixXr::Constant(g.points.rows(), 1, std::nan(""));
    for (std::size_t i = 0; i < g.rows.size(); ++i) m(g.rows[i], 0) = values(static_cast<Index>(i));
    return m;
}

MatrixXc grid_field(const Grid& g, const FieldMatrix& f) {
    MatrixXc m = MatrixXc::Constant(g.points.rows(), 3, Complex(std::nan(""), std::nan("")));
    for (std::size_t i = 0; i < g.rows.size(); ++i) m.row(g.rows[i]) = f.row(static_cast<Index>(i));
    return m;
}

/// Grid cut through the centre: along x (z = 0, axis 0) or along z (x = 0, axis 1).
void print_axis(const Grid& g, const Evaluation& ev, int axis) {
    std::vector<Index> pos(static_cast<std::size_t>(g.points.rows()), -1);
    for (std::size_t i = 0; i < g.rows.size(); ++i)
        pos[static_cast<std::size_t>(g.rows[i])] = static_cast<Index>(i);
    const Index mid = (g.n - 1) / 2;
    for (Index k = 0; k < g.n; ++k) {
        const Index row = axis == 0 ? k * g.n + mid : mid * g.n + k;
        const Index e = pos[static_cast<std::size_t>(row)];
        if (e < 0)
            continue;
        const Vec3 r = g.points.row(row).transpose();
        std::printf("AXIS %s %+.4f um d=%+.1f nm |E_mie|=%.4g |E_sim|=%.4g e=%.4g e_mag=%.4g\n",
                    axis == 0 ? "x" : "z", (axis == 0 ? r.x() : r.z()) * 1e6,
                    surface_distance(r, g.radius) * 1e9, ev.e_mie.row(e).norm(),
                    ev.e_sim.row(e).norm(), ev.rel(e), ev.mag_rel(e));
    }
}

void post_process(const Options& o, const post::SurfaceSolution& s, io::ResultWriter* out) {
    using Clock = std::chrono::steady_clock;
    const Real radius = 0.5 * o.c.diameter;
    const geometry::TriangleMesh& mesh = s.problem->space->mesh();
    const Real h = octasphere::edge_stats(mesh).mean;
    std::printf("mean edge h = %.3f nm (lambda / %.2f)\n", h * 1e9, kLambda / h);
    const auto t_mie = Clock::now();
    const reference::MieSolution mie(
        reference::MieParams{radius, kLambda, material::silver_500nm(), material::vacuum(), 0});
    std::printf("Mie n_max = %lld (%.2f s)\n", static_cast<long long>(mie.a_n().size()),
                std::chrono::duration<Real>(Clock::now() - t_mie).count());
    if (o.check) {
        const Check chk = check(s, o.c.diameter, o.spacing);
        std::printf("CHECK %s verdict=%s\n%s\n", ag_sphere_mlfmm::label(o.c).c_str(),
                    passes(chk.zones) ? "pass" : "FAIL", summary(chk).c_str());
        std::fflush(stdout);
    }
    // Near-surface profile on the shells, per quadrature degree.
    const ShellSet sh = default_shells(radius);
    for (const int deg : o.shells ? o.line_degrees : std::vector<int>{}) {
        EvalOptions eo;
        eo.quad_degree = deg;
        const Evaluation ev = evaluate(s, mie, sh.points, radius, eo);
        std::printf("SHELLS degree %d: %lld points, field %.2f s, Mie %.2f s\n", deg,
                    static_cast<long long>(sh.points.rows()), ev.field_seconds, ev.mie_seconds);
        for (std::size_t k = 0; k < sh.d.size(); ++k) {
            const ShellStats st = shell_stats(sh, ev, k);
            // e (d > 0) or |E_sim - E_Mie| (d < 0) on the front (z <= 0) / back (z > 0) half of
            // the shell, and the values at the poles and on the +x axis.
            const auto at = [&](Index a) { return static_cast<Index>(k) * sh.n_angles + a; };
            const VectorXr& val = sh.d[k] > 0.0 ? ev.rel : ev.abs;
            std::printf(
                "PROFILE degree=%d d=%+.3f nm (%+.3f h) %s front max=%.4g p95=%.4g "
                "median=%.4g | back max=%.4g p95=%.4g median=%.4g | psi=180 %.4g psi=90 "
                "%.4g psi=0 %.4g\n",
                deg, sh.d[k] * 1e9, sh.d[k] / h, sh.d[k] > 0.0 ? "e  " : "abs", st.front.max,
                st.front.p95, st.front.median, st.back.max, st.back.p95, st.back.median,
                val(at(sh.n_angles / 2)), val(at(sh.n_angles / 4)), val(at(0)));
        }
        if (out != nullptr) {
            const std::string g = "shells/degree_" + std::to_string(deg) + "/";
            out->write_matrix(g + "rel", MatrixXr(ev.rel));
            out->write_matrix(g + "abs", MatrixXr(ev.abs));
            out->write_matrix(g + "E_sim", MatrixXc(ev.e_sim));
            out->write_matrix(g + "E_mie", MatrixXc(ev.e_mie));
        }
        std::fflush(stdout);
    }
    if (out != nullptr && o.shells) {
        out->write_matrix("shells/points", MatrixXr(sh.points));
        VectorXr dd(static_cast<Index>(sh.d.size()));
        for (std::size_t i = 0; i < sh.d.size(); ++i) dd(static_cast<Index>(i)) = sh.d[i];
        out->write_matrix("shells/d", MatrixXr(dd));
    }
    if (!o.grid)
        return;
    const Grid g = make_grid(radius, o.half_factor * o.c.diameter, o.spacing, kLambda / 100.0);
    EvalOptions eo;
    eo.quad_degree = o.degree;
    eo.near_quad_degree = o.near_degree;
    eo.near_band = o.near_band_h * h;
    std::printf(
        "GRID x, z in [%.3f, %.3f] um, %lld x %lld points, spacing %.2f nm (lambda / "
        "%.1f), %lld evaluated; degree %d, degree %d within %.2f nm (%.2f h)\n",
        -g.half * 1e6, g.half * 1e6, static_cast<long long>(g.n), static_cast<long long>(g.n),
        g.spacing * 1e9, kLambda / g.spacing, static_cast<long long>(g.eval.rows()), eo.quad_degree,
        eo.near_quad_degree, eo.near_band * 1e9, o.near_band_h);
    std::fflush(stdout);
    const Evaluation ev = evaluate(s, mie, g.eval, radius, eo);
    std::printf("GRID field %.1f s (%.3f ms per point), Mie %.1f s\n", ev.field_seconds,
                1e3 * ev.field_seconds / static_cast<Real>(g.eval.rows()), ev.mie_seconds);
    print_axis(g, ev, 0);
    print_axis(g, ev, 1);
    const std::string tag = ag_sphere_mlfmm::label(o.c);
    for (const Real f : o.d_min_h) {
        const Real d_min = std::max(f * h, kLambda / 100.0);
        const ZoneStats z = zone_stats(g.eval, ev, radius, d_min);
        print_stats(tag, "lit", z.lit, d_min, h);
        print_stats(tag, "lit_x_axis", z.lit_x_axis, d_min, h);
        print_stats(tag, "lit_forward", z.lit_forward, d_min, h);
        print_stats(tag, "lit_mag", z.lit_mag, d_min, h);
        print_stats(tag, "shadow", z.shadow, d_min, h);
        print_stats(tag, "shadow_mag", z.shadow_mag, d_min, h);
        print_stats(tag, "interior_abs", z.interior_abs, d_min, h);
        std::printf("ROW %s d_min=%.2f nm interior max|E_mie|=%.4g band=%lld verdict=%s\n",
                    tag.c_str(), d_min * 1e9, z.interior_max_mie, static_cast<long long>(z.band),
                    passes(z) ? "pass" : "FAIL");
    }
    if (out != nullptr) {
        out->write_matrix("grid/points", MatrixXr(g.points));
        out->write_matrix("grid/rel", grid_column(g, ev.rel));
        out->write_matrix("grid/abs", grid_column(g, ev.abs));
        out->write_matrix("grid/mag_rel", grid_column(g, ev.mag_rel));
        out->write_matrix("grid/E_sim", grid_field(g, ev.e_sim));
        out->write_matrix("grid/E_mie", grid_field(g, ev.e_mie));
        out->write_attribute("grid_n", static_cast<Real>(g.n));
        out->write_attribute("grid_half", g.half);
        out->write_attribute("grid_spacing", g.spacing);
        out->write_attribute("grid_degree", static_cast<Real>(eo.quad_degree));
        out->write_attribute("grid_near_degree", static_cast<Real>(eo.near_quad_degree));
        out->write_attribute("grid_near_band", eo.near_band);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        const std::string tag = ag_sphere_mlfmm::label(o.c);
        std::printf("%s\n", tag.c_str());
        std::printf("available memory at start: %.1f GB of %.1f GB\n",
                    system_memory::available_memory_bytes() / 1e9,
                    system_memory::physical_memory_bytes() / 1e9);
        std::fflush(stdout);
        std::unique_ptr<io::ResultWriter> out;
        if (!o.out.empty()) {
            out = io::open_npy_directory(o.out);
            out->write_attribute("case", tag);
            out->write_attribute("radius", 0.5 * o.c.diameter);
            out->write_attribute("wavelength", kLambda);
        }
        if (!o.load.empty()) {
            const VectorXc x =
                ag_box_mechanism::read_npy<Complex>(core::path_from_utf8(o.load) / "currents.npy");
            const StoredProblem sp(o.c, x);
            std::printf("loaded currents from %s (2N = %lld)\n", o.load.c_str(),
                        static_cast<long long>(x.size()));
            post_process(o, sp.solution, out.get());
        } else {
            const ag_sphere_mlfmm::Result r = ag_sphere_mlfmm::run(o.c, [&](const Simulation& sim) {
                std::printf("solved\n%s\n", sim.report().c_str());
                std::fflush(stdout);
                if (out != nullptr)
                    out->write_vector("currents", sim.solution().currents);
                post_process(o, sim.solution(), out.get());
            });
            std::printf("RESULT %s\n%s\n", tag.c_str(), ag_sphere_mlfmm::summary(r).c_str());
        }
        std::printf("peak RSS %.2f GB\n", system_memory::peak_rss_bytes() / 1e9);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ag_near_field: %s\n", e.what());
        return 1;
    }
}
