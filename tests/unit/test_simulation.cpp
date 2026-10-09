// Unit tests of the Simulation driver (WP14a): solver / preconditioner / formulation selection,
// agreement with the low-level pipeline, idempotence, input checks and post-processing on the
// owned solution. Sphere of radius 0.5 um, icosphere n = 1 (2N = 240) at lambda = 1 um with
// cheap quadrature (far 1, near 2, singular 2, and the WP7 rules: no outer grading, fixed
// degrees, as wp7_options() in test_assembler.cpp), so that every case stays well below a second
// in release and a few seconds in the sanitizer build. Physics
// (eps_rr against Mie) is in tests/validation/test_simulation_mie.cpp.
#include "specklebem/simulation.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "assembler_test_support.hpp"

using namespace assembler_test;
using formulation::Kind;

namespace {

constexpr Real kLambdaFast = 1e-6;

kernels::OperatorOptions cheap_options() {
    kernels::OperatorOptions o;
    o.quad_degree_far = 1;
    o.quad_degree_near = 2;
    o.quad_degree_sing = 2;
    // WP7 rules: the WP7b defaults (graded touching pairs, k-aware degrees) cost 2 to 10x the
    // sanitizer time here and are not what these driver tests check.
    o.outer_grading_levels = 0;
    o.target_accuracy = 0.0;
    return o;
}

geometry::TriangleMesh sphere_mesh(int subdivisions = 1) {
    return geometry::make_icosphere(kRadius, subdivisions);
}

std::shared_ptr<excitation::Excitation> plane_wave(Real lambda = kLambdaFast,
                                                   material::Material bg = material::vacuum()) {
    return std::make_shared<excitation::PlaneWave>(lambda, Vec3::UnitZ(), Vec3c(1.0, 0.0, 0.0), bg);
}

SimulationConfig base_config(const material::Material& object) {
    SimulationConfig c;
    c.wavelength = kLambdaFast;
    c.object = object;
    c.kernels = cheap_options();
    c.gmres.verbose = false;
    return c;
}

Real rel_diff(const VectorXc& a, const VectorXc& b) {
    return (a - b).norm() / b.norm();
}

bool contains(const std::string& s, const std::string& what) {
    return s.find(what) != std::string::npos;
}

/// Low-level reference currents of the n = 1.5 sphere with PMCHWT: the same Problem ingredients
/// as the driver, DenseStrategy + assemble_rhs + solve_direct.
VectorXc reference_currents() {
    const geometry::TriangleMesh mesh = sphere_mesh();
    const basis::RwgSpace space(mesh);
    const auto wave = plane_wave();
    const auto form = formulation::make_formulation(Kind::PMCHWT);
    op::Problem p;
    p.space = &space;
    p.object = lossless_n15();
    p.formulation = form.get();
    p.excitation = wave.get();
    p.kernel_options = cheap_options();
    p.omega = wave->omega();
    const auto Z = std::dynamic_pointer_cast<op::DenseOperator>(op::DenseStrategy().build(p));
    REQUIRE(Z != nullptr);
    return solver::solve_direct(*Z, op::assemble_rhs(p));
}

SimulationConfig pmchwt_config(SolverKind kind) {
    SimulationConfig cfg = base_config(lossless_n15());
    cfg.formulation = Kind::PMCHWT;
    cfg.solver = kind;
    return cfg;
}

}  // namespace

TEST_CASE("simulation: direct solve equals the low-level pipeline", "[simulation]") {
    const VectorXc x_ref = reference_currents();
    Simulation direct(sphere_mesh(), plane_wave(), pmchwt_config(SolverKind::Direct));
    const solver::GmresResult rd = direct.solve();
    CHECK(rd.iterations == 0);
    CHECK(rd.converged);
    CHECK(rd.true_relative_residual < 1e-12);
    // gmres.hpp size rule: iterations + 1 entries.
    REQUIRE(rd.residual_history.size() == 1);
    CHECK(rd.residual_history[0] == rd.true_relative_residual);
    CHECK(rel_diff(direct.solution().currents, x_ref) < 1e-12);
    CHECK(rel_diff(rd.x, x_ref) < 1e-12);
    CHECK(contains(direct.report(), "unused by the direct solver"));
}

namespace {

struct GmresVariant {
    Kind kind;
    bool jacobi;
    solver::PreconditionerSide side;
};

/// GMRES through the driver against the low-level direct solution. The direct driver solve
/// equals x_ref to 1e-12 (previous case), so x_ref stands for it.
void check_gmres_variants(std::initializer_list<GmresVariant> variants) {
    const VectorXc x_ref = reference_currents();
    const auto wave = plane_wave();
    for (const GmresVariant v : variants) {
        SimulationConfig g = pmchwt_config(SolverKind::Gmres);
        g.formulation = v.kind;
        g.diagonal_preconditioner = v.jacobi;
        g.gmres.tolerance = 1e-10;
        g.gmres.side = v.side;
        Simulation sim(sphere_mesh(), wave, g);
        int calls = 0;
        const solver::GmresResult r = sim.solve([&calls](int, Real) { ++calls; });
        INFO("jacobi " << v.jacobi << ", iterations " << r.iterations);
        CHECK(r.converged);
        CHECK(calls == r.iterations);
        CHECK(sim.diagonal_preconditioner() == v.jacobi);
        CHECK(rel_diff(sim.solution().currents, x_ref) < 1e-7);
        CHECK(contains(sim.report(), v.jacobi ? "diagonal (Jacobi) (explicit)" : "none"));
    }
}

}  // namespace

// Split into three cases to keep each sanitizer run short (each GMRES solve takes ~1 s there).
TEST_CASE("simulation: unpreconditioned GMRES equals the direct solution", "[simulation]") {
    // ICTF (a block-row scaling of PMCHWT, so the same discrete solution): 55 iterations
    // instead of 149 for PMCHWT.
    check_gmres_variants({{Kind::ICTF, false, solver::PreconditionerSide::Left}});
}

TEST_CASE("simulation: GMRES with left Jacobi equals the direct solution", "[simulation]") {
    check_gmres_variants({{Kind::PMCHWT, true, solver::PreconditionerSide::Left}});
}

TEST_CASE("simulation: GMRES with right Jacobi equals the direct solution", "[simulation]") {
    check_gmres_variants({{Kind::PMCHWT, true, solver::PreconditionerSide::Right}});
}

TEST_CASE("simulation: automatic formulation and preconditioner choice", "[simulation]") {
    const auto wave = plane_wave();
    {
        const Simulation si(sphere_mesh(), wave, base_config(material::silicon_500nm()));
        CHECK(si.formulation_kind() == Kind::ICTF);
        CHECK_FALSE(si.diagonal_preconditioner());
        const std::string rep = si.report();
        CHECK(contains(rep, "ICTF (automatic)"));
        CHECK(contains(rep, "preconditioner: none (automatic)"));
        CHECK(contains(rep, "compression:    dense"));
        CHECK(contains(rep, "not assembled"));
    }
    {
        const Simulation ag(sphere_mesh(), wave, base_config(material::silver_500nm()));
        CHECK(ag.formulation_kind() == Kind::ICTF);
        CHECK(ag.diagonal_preconditioner());
        CHECK(contains(ag.report(), "ICTF (automatic)"));
        CHECK(contains(ag.report(), "diagonal (Jacobi) (automatic)"));
    }
    {
        // Explicit settings override the recommendation (independently of each other).
        SimulationConfig c = base_config(material::silver_500nm());
        c.formulation = Kind::MCTF;
        c.diagonal_preconditioner = false;
        const Simulation ag(sphere_mesh(), wave, c);
        CHECK(ag.formulation_kind() == Kind::MCTF);
        CHECK_FALSE(ag.diagonal_preconditioner());
        CHECK(contains(ag.report(), "MCTF (explicit)"));
        CHECK(contains(ag.report(), "preconditioner: none (explicit)"));
        c.formulation = Kind::PMCHWT;
        c.diagonal_preconditioner.reset();
        const Simulation ag2(sphere_mesh(), wave, c);
        CHECK(ag2.formulation_kind() == Kind::PMCHWT);
        CHECK(ag2.diagonal_preconditioner());
        CHECK(contains(ag2.report(), "PMCHWT (explicit)"));
        CHECK(contains(ag2.report(), "diagonal (Jacobi) (automatic)"));
    }
}

TEST_CASE("simulation: assemble is idempotent, solve can be repeated", "[simulation]") {
    SimulationConfig cfg = base_config(material::silver_500nm());  // ICTF + Jacobi
    cfg.gmres.tolerance = 1e-8;
    Simulation sim(sphere_mesh(), plane_wave(), cfg);
    CHECK(sim.num_unknowns() == 240);
    CHECK_THROWS_AS(sim.solution(), std::logic_error);
    CHECK_THROWS_AS(sim.system_operator(), std::logic_error);
    CHECK_THROWS_AS(sim.rhs(), std::logic_error);

    sim.assemble();
    const auto op1 = sim.system_operator();
    const VectorXc b1 = sim.rhs();
    sim.assemble();  // no-op: same operator object, same rhs
    CHECK(sim.system_operator() == op1);
    CHECK((sim.rhs().array() == b1.array()).all());
    CHECK(op1->rows() == 240);
    CHECK(contains(sim.report(), "dense 240x240"));
    CHECK(contains(sim.report(), "solve:          not run"));
    CHECK_THROWS_AS(sim.solution(), std::logic_error);

    const solver::GmresResult r1 = sim.solve();
    REQUIRE(r1.converged);
    const VectorXc x1 = sim.solution().currents;
    const solver::GmresResult r2 = sim.solve();
    CHECK(sim.system_operator() == op1);
    CHECK(r2.iterations == r1.iterations);
    CHECK(r2.residual_history == r1.residual_history);
    CHECK((sim.solution().currents.array() == x1.array()).all());
    CHECK(sim.solution().problem == &sim.problem());
    const std::string rep = sim.report();
    CHECK(contains(rep, "2N = 240"));
    CHECK(contains(rep, "iterations"));
    CHECK(contains(rep, "converged"));
    CHECK(contains(rep, "diagonal "));  // diagonal assembly time
    CHECK(contains(rep, "GMRES (tol 1e-08, max_iter 2000, restart 0)"));
}

TEST_CASE("simulation: constructor input checks", "[simulation]") {
    const auto wave = plane_wave();
    const SimulationConfig ok = base_config(lossless_n15());
    CHECK_NOTHROW(Simulation(sphere_mesh(0), wave, ok));

    // Open mesh: icosahedron without its last triangle.
    const geometry::TriangleMesh ico = sphere_mesh(0);
    const geometry::TriangleMesh open(ico.vertices(), ico.triangles().topRows(19));
    CHECK_THROWS_AS(Simulation(open, wave, ok), std::invalid_argument);

    CHECK_THROWS_AS(Simulation(sphere_mesh(0), nullptr, ok), std::invalid_argument);

    SimulationConfig c = ok;
    c.wavelength = kLambdaFast * (1 + 1e-9);
    CHECK_THROWS_AS(Simulation(sphere_mesh(0), wave, c), std::invalid_argument);
    c.wavelength = -1.0;
    CHECK_THROWS_AS(Simulation(sphere_mesh(0), wave, c), std::invalid_argument);

    c = ok;
    c.exterior = {Complex(1.77, 0.0), Complex(1.0, 0.0)};  // water-like, wave is in vacuum
    CHECK_THROWS_AS(Simulation(sphere_mesh(0), wave, c), std::invalid_argument);
    // A matching background is accepted.
    CHECK_NOTHROW(Simulation(sphere_mesh(0), plane_wave(kLambdaFast, c.exterior), c));

    for (const char* name : {"mlfmm", "aca", "hmatrix", "Dense", "fmm", ""}) {
        c = ok;
        c.compression = name;
        INFO("compression \"" << name << "\"");
        CHECK_THROWS_AS(Simulation(sphere_mesh(0), wave, c), std::invalid_argument);
    }

    c = ok;
    c.kernels.quad_degree_near = 3;  // not positive-interior
    CHECK_THROWS_AS(Simulation(sphere_mesh(0), wave, c), std::invalid_argument);
    c = ok;
    c.kernels.quad_degree_far = 21;
    CHECK_THROWS_AS(Simulation(sphere_mesh(0), wave, c), std::invalid_argument);

    c = ok;
    c.gmres.tolerance = 0.0;
    CHECK_THROWS_AS(Simulation(sphere_mesh(0), wave, c), std::invalid_argument);
    c = ok;
    c.object.eps_r = Complex(0.0, 0.0);
    CHECK_THROWS_AS(Simulation(sphere_mesh(0), wave, c), std::invalid_argument);
}

TEST_CASE("simulation: further constructor checks", "[simulation]") {
    const auto wave = plane_wave();
    const SimulationConfig ok = base_config(lossless_n15());

    SECTION("wavelength within the relative tolerance, NaN rejected") {
        SimulationConfig c = ok;
        c.wavelength = kLambdaFast * (1 + 1e-13);
        CHECK_NOTHROW(Simulation(sphere_mesh(0), wave, c));
        c.wavelength = kLambdaFast * (1 - 1e-13);
        const Simulation sim(sphere_mesh(0), wave, c);
        // omega is the excitation's (bitwise), not recomputed from config.wavelength.
        CHECK(sim.problem().omega == wave->omega());
        c.wavelength = std::nan("");
        CHECK_THROWS_AS(Simulation(sphere_mesh(0), wave, c), std::invalid_argument);
    }
    SECTION("mu_r of the background must match") {
        SimulationConfig c = ok;
        c.exterior.mu_r = Complex(2.0, 0.0);  // same eps_r as the wave's vacuum background
        CHECK_THROWS_AS(Simulation(sphere_mesh(0), wave, c), std::invalid_argument);
    }
    SECTION("GMRES parameters") {
        SimulationConfig c = ok;
        c.gmres.max_iter = 0;
        CHECK_THROWS_AS(Simulation(sphere_mesh(0), wave, c), std::invalid_argument);
        c = ok;
        c.gmres.restart = -1;
        CHECK_THROWS_AS(Simulation(sphere_mesh(0), wave, c), std::invalid_argument);
        c = ok;
        c.gmres.tolerance = std::nan("");
        CHECK_THROWS_AS(Simulation(sphere_mesh(0), wave, c), std::invalid_argument);
    }
    SECTION("JMCFIE is rejected by the constructor") {
        SimulationConfig c = ok;
        c.formulation = Kind::JMCFIE;
        CHECK_THROWS_AS(Simulation(sphere_mesh(0), wave, c), std::invalid_argument);
    }
    SECTION("inward-pointing normals are rejected") {
        geometry::TriangleMesh inward = sphere_mesh(0);
        REQUIRE(inward.signed_volume() > 0);
        inward.flip_normals();
        REQUIRE(inward.is_closed());
        REQUIRE(inward.signed_volume() < 0);
        CHECK_THROWS_AS(Simulation(inward, wave, ok), std::invalid_argument);
        inward.flip_normals();
        CHECK_NOTHROW(Simulation(inward, wave, ok));
    }
#ifdef NDEBUG
    SECTION("dense size limit") {
        // Icosphere n = 6: 122 880 edges, 2N = 245 760 > op::kMaxDenseUnknowns; rejected before
        // anything is assembled. Release only (the mesh construction of 81 920 triangles takes
        // seconds in the sanitizer build).
        const geometry::TriangleMesh big = sphere_mesh(6);
        REQUIRE(2 * big.num_edges() > op::kMaxDenseUnknowns);
        CHECK_THROWS_AS(Simulation(big, wave, ok), std::invalid_argument);
    }
#endif
}

TEST_CASE("simulation: per-call GMRES parameters", "[simulation]") {
    SimulationConfig cfg = base_config(lossless_n15());
    cfg.formulation = Kind::ICTF;
    cfg.diagonal_preconditioner = false;
    cfg.gmres.tolerance = 1e-3;
    Simulation sim(sphere_mesh(), plane_wave(), cfg);

    solver::GmresParams invalid = cfg.gmres;
    invalid.max_iter = 0;
    CHECK_THROWS_AS(sim.solve(invalid), std::invalid_argument);
    invalid = cfg.gmres;
    invalid.restart = -1;
    CHECK_THROWS_AS(sim.solve(invalid), std::invalid_argument);
    invalid = cfg.gmres;
    invalid.tolerance = 0.0;
    CHECK_THROWS_AS(sim.solve(invalid), std::invalid_argument);
    CHECK_THROWS_AS(sim.solution(), std::logic_error);

    const solver::GmresResult coarse = sim.solve();  // config().gmres: tolerance 1e-3
    REQUIRE(coarse.converged);
    CHECK(contains(sim.report(), "GMRES (tol 0.001,"));
    solver::GmresParams fine = cfg.gmres;
    fine.tolerance = 1e-10;
    int calls = 0;
    const solver::GmresResult r = sim.solve(fine, [&calls](int, Real) { ++calls; });
    INFO("iterations: tol 1e-3 " << coarse.iterations << ", tol 1e-10 " << r.iterations);
    REQUIRE(r.converged);
    CHECK(r.iterations > coarse.iterations);
    CHECK(calls == r.iterations);
    CHECK(r.residual_history.back() <= 1e-10);
    CHECK(sim.config().gmres.tolerance == 1e-3);  // the config is unchanged
    CHECK(contains(sim.report(), "GMRES (tol 1e-10,"));
    CHECK(rel_diff(sim.solution().currents, reference_currents()) < 1e-7);
}

TEST_CASE("simulation: non-converged GMRES keeps the solution", "[simulation]") {
    SimulationConfig cfg = base_config(lossless_n15());
    cfg.formulation = Kind::ICTF;
    cfg.diagonal_preconditioner = false;
    cfg.gmres.tolerance = 1e-10;
    cfg.gmres.max_iter = 1;
    Simulation sim(sphere_mesh(), plane_wave(), cfg);
    const solver::GmresResult r = sim.solve();
    CHECK_FALSE(r.converged);
    CHECK(r.iterations == 1);
    CHECK(r.residual_history.size() == 2);
    REQUIRE_NOTHROW(sim.solution());
    CHECK(sim.solution().currents.size() == sim.num_unknowns());
    CHECK(sim.solution().currents.allFinite());
    CHECK(sim.solution().currents.norm() > 0.0);
    CHECK(contains(sim.report(), "1 iterations, NOT converged"));
}

TEST_CASE("simulation: the direct solver assembles no diagonal", "[simulation]") {
    SimulationConfig cfg = pmchwt_config(SolverKind::Direct);
    cfg.diagonal_preconditioner = true;
    Simulation sim(sphere_mesh(), plane_wave(), cfg);
    CHECK(sim.diagonal_preconditioner());
    sim.assemble();
    const solver::GmresResult r = sim.solve();
    CHECK(r.converged);
    const std::string rep = sim.report();
    const auto pos = rep.find("  assembly:");
    REQUIRE(pos != std::string::npos);
    const std::string assembly_line = rep.substr(pos, rep.find('\n', pos) - pos);
    INFO(assembly_line);
    CHECK(contains(assembly_line, "rhs"));
    CHECK_FALSE(contains(assembly_line, "diagonal"));
    CHECK(contains(rep, "diagonal (Jacobi) (explicit), unused by the direct solver"));
}

TEST_CASE("simulation: far field and RCS on the owned solution", "[simulation]") {
    std::unique_ptr<Simulation> sim;
    {
        // The mesh and the excitation handle go out of scope: the Simulation owns its copies.
        SimulationConfig cfg = base_config(lossless_n15());
        cfg.solver = SolverKind::Direct;
        sim = std::make_unique<Simulation>(sphere_mesh(), plane_wave(), cfg);
    }
    sim->solve();
    const post::SurfaceSolution& sol = sim->solution();
    REQUIRE(sol.problem != nullptr);
    CHECK(sol.problem->space->size() * 2 == sim->num_unknowns());
    CHECK(&sol.problem->space->mesh() == &sim->mesh());

    Vertices dirs(2, 3);
    dirs.row(0) = Vec3::UnitZ().transpose();
    dirs.row(1) = Vec3::UnitX().transpose();
    Eigen::Matrix<Complex, Eigen::Dynamic, 3> F;
    post::far_field(sol, dirs, F);
    REQUIRE(F.rows() == 2);
    CHECK(F.allFinite());
    CHECK(F.row(0).norm() > 0.0);

    VectorXr angles(3);
    angles << 0.0, 0.5 * kPi, kPi;
    const VectorXr rcs = post::bistatic_rcs(sol, Vec3::UnitY(), angles);
    REQUIRE(rcs.size() == 3);
    CHECK(rcs.allFinite());
    CHECK((rcs.array() > 0.0).all());
    // Forward RCS from the far field: sigma = 4 pi |F|^2 / |E0|^2 with |E0| = 1 V/m.
    CHECK(std::abs(rcs(0) - 4 * kPi * F.row(0).squaredNorm()) <= 1e-12 * rcs(0));
}
