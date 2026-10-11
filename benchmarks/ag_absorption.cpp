// Absorbed power of the flat box (WP-A1, follow-up of WP22c): one flat-patch case of
// benchmarks/fresnel_flat.cpp (same options, same solve) with diagnostics of the absorbed power
// from the surface currents (tests/support/ag_absorption_support.hpp):
//  * P_abs = 1/2 Re oint (n x M) . J* dS per part (top face, walls, bottom) with the exact
//    Dunavant degree 2 (as the WP22c study) and with degree 6 (pointwise check);
//  * P_ext - P_sca: extinction from the currents and the incident beam minus the scattered power
//    of both hemispheres (far field), an absorbed power that does not use n x M . J*;
//  * the analytic currents of the beam on the infinite interface (every plane wave reflected
//    with its Fresnel coefficients) and their RWG interpolant, compared with the simulated
//    currents on the centre (ellipse of semi-axes w0 / (2 cos theta), w0 / 2), the inner square
//    |x|, |y| <= L / 4 and the whole top face: absorbed power (simulated, reference,
//    interpolant, cross terms) and the correlation of J and M with the reference (amplitude,
//    phase).
// Output (stdout): the fresnel_flat summary, then DIAG lines and an ABS line; the ABS line is also
// appended to --summary FILE. Record: benchmarks/results/ag_absorption.md.
//
// Usage: specklebem_ag_absorption --material ag|si [--L 2e-6] [--w0 0.4e-6] [--theta 0]
//            [--pol p|s] [--mesh-size 50e-9] [--compression mlfmm|dense] [--digits|--d0 3]
//            [--tol 1e-5] [--max-iter 6000] [--restart 0] [--ff-dtheta 1]
//            [--formulation pmchwt|ictf|mctf] [--jacobi 0|1] [--summary FILE] [--force]
//            [--save-currents FILE] [--load-currents FILE] [--refine 1e-6,1e-7]
//            [--row-check RADIUS] [--profile BINS] [--spectrum] [--strip-check HALF_WIDTH]
// --save-currents writes the solved [J; M] (raw complex<double>); --load-currents evaluates the
// diagnostics of saved currents of the same case without assembly and solve. --refine adds
// iterative-refinement steps x <- x + GMRES(Z, b - Z x) (left Jacobi) down to the listed true
// relative residuals and prints a STAGE line (R, A, phase of M against J) after each.
// --row-check compares the compressed operator with the exact Galerkin rows (op::assemble_sparse)
// of the central top-face basis functions on the solution and prints the exact residual there.
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/solver/gmres.hpp"
#include "specklebem/solver/preconditioner.hpp"

#include <chrono>
#include <cstdio>
#include <exception>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ag_absorption_support.hpp"
#include "fresnel_flat_support.hpp"

namespace {

using namespace fresnel_flat;
namespace aa = ag_absorption;

struct Options {
    Case c;
    std::string summary;
    bool force = false;
    std::string save;  ///< --save-currents FILE: raw [J; M] (complex<double>) after the solve
    std::string load;  ///< --load-currents FILE: diagnostics of saved currents, no solve
    /// --refine T1,T2,...: iterative refinement to the true relative residuals T1, T2, ...
    std::vector<Real> refine;
    /// --row-check RADIUS: exact system rows of the top-face basis functions within RADIUS of
    /// the centre against the compressed operator (op::assemble_sparse) [m]
    Real row_check_radius = 0;
    /// --profile BINS (with --load-currents): metrics in BINS strips along x and along y
    int profile_bins = 0;
    /// --spectrum (with --load-currents): spatial spectrum along x of the current error
    bool spectrum = false;
    /// --strip-check HALF_WIDTH (with --load-currents): compressed vs exact rows on the strip
    /// |y| <= HALF_WIDTH applied to the interpolated analytic currents [m]
    Real strip_half_width = 0;
};

Options parse(int argc, char** argv) {
    Options o;
    o.c.L = 2e-6;
    o.c.waist = 0.4e-6;
    o.c.tolerance = 1e-5;
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
        } else if (a == "--compression") {
            o.c.compression = value();
        } else if (a == "--digits" || a == "--d0") {
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
        } else if (a == "--save-currents") {
            o.save = value();
        } else if (a == "--load-currents") {
            o.load = value();
        } else if (a == "--strip-check") {
            o.strip_half_width = std::stod(value());
        } else if (a == "--spectrum") {
            o.spectrum = true;
        } else if (a == "--profile") {
            o.profile_bins = std::stoi(value());
        } else if (a == "--row-check") {
            o.row_check_radius = std::stod(value());
        } else if (a == "--refine") {
            std::stringstream ss(value());
            std::string item;
            while (std::getline(ss, item, ',')) o.refine.push_back(std::stod(item));
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

void save_currents(const std::string& path, const VectorXc& x) {
    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    f.write(reinterpret_cast<const char*>(x.data()),
            static_cast<std::streamsize>(static_cast<std::size_t>(x.size()) * sizeof(Complex)));
}

VectorXc load_currents(const std::string& path, Index n) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    VectorXc x(n);
    f.read(reinterpret_cast<char*>(x.data()),
           static_cast<std::streamsize>(static_cast<std::size_t>(n) * sizeof(Complex)));
    if (!f || f.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error(path + ": size does not match 2N = " + std::to_string(n));
    return x;
}

struct Diagnostics {
    aa::AbsorbedParts deg2, deg6;
    Real p_ext = 0;
    Real p_fwd_full = 0;
    Real a_beam = 0;   ///< 1 - R_beam - away share
    Real a_patch = 0;  ///< reference absorbed flux through the L x L patch / P_inc
    aa::TopMetrics centre, inner, top;
    std::size_t ref_waves = 0;
    Real seconds = 0;
};

std::string metrics_line(const char* name, const aa::TopMetrics& m, Real p_inc) {
    char buf[900];
    std::snprintf(
        buf, sizeof(buf),
        "DIAG %s | area %.4g um2 | A_sim %.6f | A_ref %.6f | A_int %.6f | sim/ref-1 %+.4f | "
        "int/ref-1 %+.4f | A(Msim,Jref)/ref-1 %+.4f | A(Mref,Jsim)/ref-1 %+.4f | |c_J| %.6f | "
        "arg c_J %+.3f mrad | |c_M| %.6f | arg c_M %+.3f mrad | err_J %.4e | err_M %.4e | "
        "err_J_int %.4e | err_M_int %.4e | |M|/|J| %.3f Ohm",
        name, m.area * 1e12, m.p_sim / p_inc, m.p_ref / p_inc, m.p_int / p_inc,
        m.p_sim / m.p_ref - 1.0, m.p_int / m.p_ref - 1.0, m.p_ms_jr / m.p_ref - 1.0,
        m.p_mr_js / m.p_ref - 1.0, std::abs(m.c_j), 1e3 * std::arg(m.c_j), std::abs(m.c_m),
        1e3 * std::arg(m.c_m), m.err_j, m.err_m, m.err_j_int, m.err_m_int, m.m_over_j);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        const Case& c = o.c;
        std::printf("%s\n", label(c).c_str());
        std::printf("available memory at start: %.1f GB of %.1f GB\n",
                    system_memory::available_memory_bytes() / 1e9,
                    system_memory::physical_memory_bytes() / 1e9);
        std::fflush(stdout);
        const Geometry geo = make_geometry(c);
        const auto tb = std::chrono::steady_clock::now();
        const auto beam = make_beam(c, geo.r_max);
        const Real build_s =
            std::chrono::duration<Real>(std::chrono::steady_clock::now() - tb).count();
        const Estimate e = estimate(geo, c, beam);
        std::printf("SETUP | %s\n", summary(e).c_str());
        std::fflush(stdout);
        const Real need = e.peak_setup_bytes + e.krylov_max_bytes;
        const Real avail = system_memory::available_memory_bytes();
        if (!o.force && !(avail > need + 10e9)) {
            std::fprintf(stderr,
                         "ag_absorption: estimated peak %.1f GB + 10 GB margin exceeds the "
                         "available %.1f GB (--force to run)\n",
                         need / 1e9, avail / 1e9);
            return 2;
        }
        const Complex eps2 = object_material(c.material).eps_r;
        const Real th = c.theta_deg * kDeg;
        Diagnostics d;
        const auto diagnose = [&](const post::SurfaceSolution& s) {
            const auto t0 = std::chrono::steady_clock::now();
            d.deg2 = aa::absorbed_power_parts(s, geo.spacing, 2);
            d.deg6 = aa::absorbed_power_parts(s, geo.spacing, 6);
            d.p_ext = aa::extinction_power(s, *beam, 6);
            const Real eta1 = material::vacuum().wave_impedance(beam->omega()).real();
            const Vec3 fwd(std::sin(th), 0.0, std::cos(th));
            d.p_fwd_full = hemisphere_power(s, hemisphere_grid(+1.0, c.ff_dtheta_deg), c.ff_degree,
                                            eta1, fwd, 0.1)
                               .total;
            const aa::InterfaceFields f(*beam, 1.0, eps2);
            d.ref_waves = f.plane_waves();
            const BeamReflectance br = beam_reflectance(*beam, 1.0, eps2);
            d.a_beam = 1.0 - br.reflectance - br.away_share;
            d.a_patch = aa::reference_absorbed_flux(f, 0.5 * c.L, kLambda / 16.0) / beam->power();
            const VectorXc interp = aa::interpolate_reference(*s.problem->space, f, geo.spacing);
            const Real a2 = 0.25 * c.waist * c.waist;
            const Real ct = std::cos(th);
            d.centre = aa::top_metrics(
                s, f, interp, geo.spacing,
                [&](const Vec3& p) { return p.x() * p.x() * ct * ct + p.y() * p.y() <= a2; }, 4);
            d.inner = aa::top_metrics(
                s, f, interp, geo.spacing,
                [&](const Vec3& p) {
                    return std::abs(p.x()) <= 0.25 * c.L && std::abs(p.y()) <= 0.25 * c.L;
                },
                4);
            d.top = aa::top_metrics(s, f, interp, geo.spacing, [](const Vec3&) { return true; }, 4);
            d.seconds = std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
        };
        // Exact rows of the system at the patch centre against the compressed operator.
        const auto do_row_check = [&](const Simulation& sim, const VectorXc& x) {
            if (!(o.row_check_radius > 0.0))
                return;
            const std::vector<Index> rows =
                aa::central_basis(*sim.problem().space, geo.spacing, o.row_check_radius);
            const aa::RowCheck rc =
                aa::row_check(sim.problem(), *sim.system_operator(), sim.rhs(), x, rows);
            std::printf(
                "ROWCHECK | %lld basis functions within %.3g um | op error J rows %.3e | "
                "op error M rows %.3e | residual exact %.3e | residual compressed %.3e | "
                "%.1f s\n",
                static_cast<long long>(rc.rows), o.row_check_radius * 1e6, rc.op_error_j,
                rc.op_error_m, rc.residual_exact, rc.residual_compr, rc.seconds);
            std::fflush(stdout);
        };
        if (!o.load.empty()) {
            // Diagnostics of saved currents (no solve; assembly only for --row-check).
            Simulation sim(geo.mesh, beam, make_config(c));
            const post::SurfaceSolution s{&sim.problem(),
                                          load_currents(o.load, sim.num_unknowns())};
            const Reference ref = reference(c, *beam, build_s);
            const Real eta1 = material::vacuum().wave_impedance(beam->omega()).real();
            const Vec3 specular(std::sin(th), 0.0, -std::cos(th));
            const Real p_refl = hemisphere_power(s, hemisphere_grid(-1.0, c.ff_dtheta_deg),
                                                 c.ff_degree, eta1, specular, 0.1)
                                    .total;
            diagnose(s);
            const Real P = ref.power;
            const Real p_sca = p_refl + d.p_fwd_full;
            std::printf(
                "LOADED %s | R_sim %.7f | R_beam %.7f | A_sim %.6f | A_ext %.6f | A_beam "
                "%.6f | A_patch %.6f | walls %.3e | bottom %.3e\n",
                o.load.c_str(), p_refl / P, ref.beam.reflectance, d.deg2.total() / P,
                (d.p_ext - p_sca) / P, d.a_beam, d.a_patch, d.deg2.walls / P, d.deg2.bottom / P);
            const std::string mc = metrics_line("centre", d.centre, P);
            const std::string mi = metrics_line("inner", d.inner, P);
            const std::string mt = metrics_line("top", d.top, P);
            std::printf("%s\n%s\n%s\n", mc.c_str(), mi.c_str(), mt.c_str());
            if (o.profile_bins > 0) {
                const aa::InterfaceFields f(*beam, 1.0, eps2);
                const VectorXc interp =
                    aa::interpolate_reference(*s.problem->space, f, geo.spacing);
                for (const int axis : {0, 1}) {
                    for (const aa::ProfileBin& b : aa::top_profile(
                             s, f, interp, geo.spacing, c.L, o.profile_bins, axis, c.L / 8.0)) {
                        std::printf(
                            "PROFILE %s %+.3f um | A_ref/P %.3e | sim/ref-1 %+.4f | "
                            "int/ref-1 %+.4f | |c_J| %.5f | |c_M| %.5f | dphi(M-J) "
                            "%+.3f mrad | err_J %.3e | err_M %.3e\n",
                            axis == 0 ? "x" : "y", b.centre * 1e6, b.m.p_ref / ref.power,
                            b.m.p_sim / b.m.p_ref - 1.0, b.m.p_int / b.m.p_ref - 1.0,
                            std::abs(b.m.c_j), std::abs(b.m.c_m),
                            1e3 * (std::arg(b.m.c_m) - std::arg(b.m.c_j)), b.m.err_j, b.m.err_m);
                    }
                }
                std::fflush(stdout);
            }
            if (o.spectrum) {
                // Spatial spectrum along x of the current error X_sim - X_ref on the strip
                // |y| <= L / 8 (top-triangle centroids, area weights): which tangential wavenumber
                // carries the error.
                const aa::InterfaceFields f(*beam, 1.0, eps2);
                const geometry::TriangleMesh& mesh = geo.mesh;
                const Real k = 2.0 * constants::pi / kLambda;
                std::vector<Real> xs, ws;
                std::vector<Vec3c> ej, em, rj, rm;
                for (Index t = 0; t < mesh.num_triangles(); ++t) {
                    const Vec3 cen = mesh.centroid(t);
                    if (aa::part_of(mesh, t, geo.spacing) != aa::Part::Top ||
                        std::abs(cen.y()) > c.L / 8.0)
                        continue;
                    const auto [Js, Ms] = aa::currents_at(s, t, cen);
                    const auto [Jr, Mr] = f.currents(cen.x(), cen.y());
                    xs.push_back(cen.x());
                    ws.push_back(mesh.area(t));
                    ej.push_back(Js - Jr);
                    em.push_back(Ms - Mr);
                    rj.push_back(Jr);
                    rm.push_back(Mr);
                }
                for (int i = -30; i <= 30; ++i) {
                    const Real kx = 0.05 * static_cast<Real>(i) * k;
                    Vec3c sej = Vec3c::Zero(), sem = Vec3c::Zero(), srj = Vec3c::Zero(),
                          srm = Vec3c::Zero();
                    for (std::size_t q = 0; q < xs.size(); ++q) {
                        const Complex ph = ws[q] * std::polar(1.0, kx * xs[q]);
                        sej += ph * ej[q];
                        sem += ph * em[q];
                        srj += ph * rj[q];
                        srm += ph * rm[q];
                    }
                    std::printf(
                        "SPECTRUM kx/k %+.2f | |J_err| %.3e | |M_err| %.3e | |J_ref| %.3e | "
                        "|M_ref| %.3e\n",
                        kx / k, sej.norm(), sem.norm(), srj.norm(), srm.norm());
                }
                std::fflush(stdout);
            }
            if (o.strip_half_width > 0.0) {
                // Compressed vs exact rows along the strip |y| <= half width, applied to the RWG
                // interpolant of the analytic currents (tangential wavenumber +k sin(theta) only):
                // spatial spectrum along x of the operator error on the J and M rows.
                sim.assemble();
                const basis::RwgSpace& space = *sim.problem().space;
                const geometry::TriangleMesh& mesh = space.mesh();
                const Index N = space.size();
                const aa::InterfaceFields f(*beam, 1.0, eps2);
                const VectorXc xpw = aa::interpolate_reference(space, f, geo.spacing);
                std::vector<Index> rows;
                std::vector<Real> xm;
                for (Index n = 0; n < N; ++n) {
                    if (aa::part_of(mesh, space.plus_triangle(n), geo.spacing) != aa::Part::Top ||
                        aa::part_of(mesh, space.minus_triangle(n), geo.spacing) != aa::Part::Top)
                        continue;
                    const Vec3 mid = 0.5 * (mesh.vertices().row(mesh.edges()(n, 0)).transpose() +
                                            mesh.vertices().row(mesh.edges()(n, 1)).transpose());
                    if (std::abs(mid.y()) <= o.strip_half_width) {
                        rows.push_back(n);
                        xm.push_back(mid.x());
                    }
                }
                op::BasisPattern pat;
                pat.group_of_row.assign(static_cast<std::size_t>(N), 0);
                pat.col_ptr = {0, 0, N};
                pat.cols.resize(static_cast<std::size_t>(N));
                for (Index n = 0; n < N; ++n) pat.cols[static_cast<std::size_t>(n)] = n;
                for (const Index m : rows) pat.group_of_row[static_cast<std::size_t>(m)] = 1;
                const auto Zex = op::assemble_sparse(sim.problem(), pat);
                VectorXc ye(2 * N), yc(2 * N);
                Zex->apply(xpw, ye);
                sim.system_operator()->apply(xpw, yc);
                const Real k = 2.0 * constants::pi / kLambda;
                Real nd = 0, ne = 0;
                for (const Index m : rows) {
                    for (const Index r : {m, N + m}) {
                        nd += std::norm(yc(r) - ye(r));
                        ne += std::norm(ye(r));
                    }
                }
                std::printf(
                    "STRIP | %zu basis functions with |y| <= %.3g um | relative operator "
                    "error on the interpolant %.3e\n",
                    rows.size(), o.strip_half_width * 1e6, std::sqrt(nd / ne));
                for (int i = -30; i <= 30; ++i) {
                    const Real kx = 0.05 * static_cast<Real>(i) * k;
                    Complex dj = 0, dm = 0, ej = 0, em = 0;
                    for (std::size_t q = 0; q < rows.size(); ++q) {
                        const Complex ph = std::polar(1.0, kx * xm[q]);
                        const Index m = rows[q];
                        dj += ph * (yc(m) - ye(m));
                        dm += ph * (yc(N + m) - ye(N + m));
                        ej += ph * ye(m);
                        em += ph * ye(N + m);
                    }
                    std::printf(
                        "STRIPSPEC kx/k %+.2f | J rows: |dZx| %.3e |Zx| %.3e | M rows: "
                        "|dZx| %.3e |Zx| %.3e\n",
                        kx / k, std::abs(dj), std::abs(ej), std::abs(dm), std::abs(em));
                }
                std::fflush(stdout);
            }
            if (o.row_check_radius > 0.0) {
                sim.assemble();
                do_row_check(sim, s.currents);
            }
            return 0;
        }
        // One summary line of a solution (after the solve or a refinement step).
        const auto stage = [&](const std::string& name, const post::SurfaceSolution& s,
                               Real true_res) {
            const Real eta1 = material::vacuum().wave_impedance(beam->omega()).real();
            const Vec3 specular(std::sin(th), 0.0, -std::cos(th));
            const Real p_refl = hemisphere_power(s, hemisphere_grid(-1.0, c.ff_dtheta_deg),
                                                 c.ff_degree, eta1, specular, 0.1)
                                    .total;
            diagnose(s);
            const Real P = beam->power();
            const Real r_beam = beam_reflectance(*beam, 1.0, eps2).reflectance;
            std::printf(
                "STAGE %s | true_res %.3e | R_sim/R_beam-1 %+.5f %% | A_sim %.6f | A_ext "
                "%.6f | A_beam %.6f | walls %+.2e | centre %+.4f | inner %+.4f | top "
                "%+.4f | dphi(M-J) centre %+.3f inner %+.3f top %+.3f mrad\n",
                name.c_str(), true_res, 100.0 * (p_refl / P / r_beam - 1.0), d.deg2.total() / P,
                (d.p_ext - p_refl - d.p_fwd_full) / P, d.a_beam, d.deg2.walls / P,
                d.centre.p_sim / d.centre.p_ref - 1.0, d.inner.p_sim / d.inner.p_ref - 1.0,
                d.top.p_sim / d.top.p_ref - 1.0,
                1e3 * (std::arg(d.centre.c_m) - std::arg(d.centre.c_j)),
                1e3 * (std::arg(d.inner.c_m) - std::arg(d.inner.c_j)),
                1e3 * (std::arg(d.top.c_m) - std::arg(d.top.c_j)));
            std::fflush(stdout);
        };
        const Result r = run(c, geo, beam, build_s, [&](const Simulation& sim) {
            if (!o.save.empty())
                save_currents(o.save, sim.solution().currents);
            do_row_check(sim, sim.solution().currents);
            if (!o.refine.empty()) {
                // Iterative refinement: x <- x + GMRES(Z, b - Z x) to the true residuals listed.
                const op::LinearOperator& Z = *sim.system_operator();
                const VectorXc& b = sim.rhs();
                const solver::DiagonalPreconditioner jac(op::assemble_diagonal(sim.problem()));
                VectorXc x = sim.solution().currents;
                VectorXc zx(x.size());
                Z.apply(x, zx);
                stage("solve", sim.solution(), (b - zx).norm() / b.norm());
                for (const Real target : o.refine) {
                    const VectorXc res = b - zx;
                    solver::GmresParams gp;
                    gp.tolerance = target * b.norm() / res.norm();
                    gp.max_iter = c.max_iter;
                    gp.verbose = true;
                    const solver::GmresResult g = solver::gmres(Z, res, jac, gp);
                    x += g.x;
                    Z.apply(x, zx);
                    const Real tr = (b - zx).norm() / b.norm();
                    char name[64];
                    std::snprintf(name, sizeof(name), "refine %.0e (%d it)", target, g.iterations);
                    stage(name, post::SurfaceSolution{&sim.problem(), x}, tr);
                }
            }
            diagnose(sim.solution());
        });
        const Real P = r.ref.power;
        std::printf("%s\n\n%s\n\n", summary(r).c_str(), r.report.c_str());
        std::printf("DIAG parts (degree 2) | top %.6f | walls %.3e | bottom %.3e | total %.6f\n",
                    d.deg2.top / P, d.deg2.walls / P, d.deg2.bottom / P, d.deg2.total() / P);
        std::printf("DIAG parts (degree 6) | top %.6f | walls %.3e | bottom %.3e | total %.6f\n",
                    d.deg6.top / P, d.deg6.walls / P, d.deg6.bottom / P, d.deg6.total() / P);
        const Real p_sca = r.p_refl + d.p_fwd_full;
        std::printf(
            "DIAG extinction | P_ext/P_inc %.7f | P_refl/P_inc %.7f | P_fwd/P_inc %.7f | "
            "(P_ext - P_sca)/P_inc %.6f\n",
            d.p_ext / P, r.p_refl / P, d.p_fwd_full / P, (d.p_ext - p_sca) / P);
        std::printf(
            "DIAG reference | %zu plane waves | A_beam = 1 - R_beam - away %.6f | "
            "A_ref over the L x L patch %.6f | Fresnel 1 - |r|^2 %.6f\n",
            d.ref_waves, d.a_beam, d.a_patch, 1.0 - r.ref.beam.fresnel_central);
        const std::string mc = metrics_line("centre", d.centre, P);
        const std::string mi = metrics_line("inner", d.inner, P);
        const std::string mt = metrics_line("top", d.top, P);
        std::printf("%s\n%s\n%s\nDIAG time %.1f s\n", mc.c_str(), mi.c_str(), mt.c_str(),
                    d.seconds);
        char buf[1200];
        std::snprintf(
            buf, sizeof(buf),
            "ABS | %s | L %.4g | w0 %.4g | theta %.4g | pol %s | h %.4g | %s%s | %s | tol %.1e "
            "| 2N %lld | it %d | R_sim %.7f | R_beam %.7f | err_R %+.4f %% | A_sim %.6f | A_beam "
            "%.6f | A_sim/A_beam-1 %+.4f | A_ext %.6f | A_top %.6f | A_walls %.2e | A_bottom "
            "%.2e | centre sim/ref-1 %+.4f int/ref-1 %+.4f dphi(M-J) %+.3f mrad | inner sim/ref-1 "
            "%+.4f | 1-R-A %+.3e | solve %.0f s",
            c.material.c_str(), c.L, c.waist, c.theta_deg,
            c.pol == excitation::Polarization::P ? "p" : "s", c.mesh_size, r.formulation.c_str(),
            r.jacobi ? "+J" : "", c.compression.c_str(), c.tolerance,
            static_cast<long long>(r.est.unknowns), r.iterations, r.r_sim(), r.ref.beam.reflectance,
            100.0 * r.error_beam(), d.deg2.total() / P, d.a_beam,
            d.deg2.total() / P / d.a_beam - 1.0, (d.p_ext - p_sca) / P, d.deg2.top / P,
            d.deg2.walls / P, d.deg2.bottom / P, d.centre.p_sim / d.centre.p_ref - 1.0,
            d.centre.p_int / d.centre.p_ref - 1.0,
            1e3 * (std::arg(d.centre.c_m) - std::arg(d.centre.c_j)),
            d.inner.p_sim / d.inner.p_ref - 1.0, 1.0 - r.r_sim() - d.deg2.total() / P, r.solve_s);
        std::printf("%s\n", buf);
        std::fflush(stdout);
        append_summary(o.summary, buf);
        return r.converged ? 0 : 1;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "ag_absorption: %s\n", ex.what());
        return 1;
    }
}
