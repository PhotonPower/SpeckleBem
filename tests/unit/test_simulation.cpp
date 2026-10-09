// Unit tests of the Simulation driver (WP14a): solver / preconditioner / formulation selection,
// agreement with the low-level pipeline, idempotence, input checks and post-processing on the
// owned solution. Sphere of radius 0.5 um, icosphere n = 1 (2N = 240) at lambda = 1 um with
// cheap quadrature (far 1, near 2, singular 2), as in test_assembler.cpp, so that every case
// stays well below a second in release and a few seconds in the sanitizer build. Physics
// (eps_rr against Mie) is in tests/validation/test_simulation_mie.cpp.
#include "specklebem/simulation.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
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
    return o;
}

geometry::TriangleMesh sphere_mesh(int subdivisions = 1) {
    return geometry::make_icosphere(kRadius, subdivisions);
}

std::shared_ptr<excitation::Excitation> plane_wave(Real lambda = kLambdaFast,
                                                   material::Material bg = material::vacuum()) {
    return std::make_shared<excitation::PlaneWave>(lambda, Vec3::UnitZ(), Vec3c(1.0, 0.0, 0.0),
                                                   bg);
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

}  // namespace

TEST_CASE("simulation: direct and GMRES agree with the low-level pipeline", "[simulation]") {
    const material::Material mat = lossless_n15();

    // Low-level reference: same Problem ingredients, DenseStrategy + assemble_rhs + solve_direct.
    const geometry::TriangleMesh mesh = sphere_mesh();
    const basis::RwgSpace space(mesh);
    const auto wave = plane_wave();
    const auto form = formulation::make_formulation(Kind::PMCHWT);
    op::Problem p;
    p.space = &space;
    p.object = mat;
    p.formulation = form.get();
    p.excitation = wave.get();
    p.kernel_options = cheap_options();
    p.omega = wave->omega();
    const auto Z = std::dynamic_pointer_cast<op::DenseOperator>(op::DenseStrategy().build(p));
    REQUIRE(Z != nullptr);
    const VectorXc x_ref = solver::solve_direct(*Z, op::assemble_rhs(p));

    SimulationConfig cfg = base_config(mat);
    cfg.formulation = Kind::PMCHWT;
    cfg.solver = SolverKind::Direct;
    Simulation direct(sphere_mesh(), wave, cfg);
    const solver::GmresResult rd = direct.solve();
    CHECK(rd.iterations == 0);
    CHECK(rd.converged);
    CHECK(rd.true_relative_residual < 1e-12);
    REQUIRE(rd.residual_history.size() == 2);
    CHECK(rd.residual_history[0] == 1.0);
    CHECK(rd.residual_history[1] == rd.true_relative_residual);
    CHECK(rel_diff(direct.solution().currents, x_ref) < 1e-12);
    CHECK(rel_diff(rd.x, x_ref) < 1e-12);

    struct Variant {
        bool jacobi;
        solver::PreconditionerSide side;
    };
    for (const Variant v : {Variant{false, solver::PreconditionerSide::Left},
                            Variant{true, solver::PreconditionerSide::Left},
                            Variant{true, solver::PreconditionerSide::Right}}) {
        SimulationConfig g = cfg;
        g.solver = SolverKind::Gmres;
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
        CHECK(rel_diff(sim.solution().currents, direct.solution().currents) < 1e-7);
        CHECK(contains(sim.report(), v.jacobi ? "diagonal (Jacobi) (explicit)" : "none"));
    }
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
