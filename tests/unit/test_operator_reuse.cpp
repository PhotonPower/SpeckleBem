// Unit tests of the operator reuse across excitations (WP-KR1, ADR 0013 decision 1):
// Simulation::set_excitation (right-hand side re-assembled with the code of assemble(), operator
// and preconditioner kept, solution state reset, checks as the constructors) and the explicit
// initial guess of Simulation::solve (InitialGuess::Previous). Two cases at CI size: the tiny
// rough PML surface of test_incomplete_lu_preconditioner.cpp with compression "mlfmm" (the ILUT
// default of that route since WP-SC4) and a dense box-route sphere (GMRES, Jacobi). Every swap is
// compared with a fresh Simulation built with the new excitation: right-hand side, solution and
// post-processing bitwise (the same deterministic assembly, preconditioner and GMRES).
#include "specklebem/excitation/angular_spectrum_beam.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/geometry/open_surface.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/postprocessing/fields.hpp"
#include "specklebem/postprocessing/scattering.hpp"
#include "specklebem/simulation.hpp"
#include "specklebem/solver/gmres.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace specklebem;
using Catch::Matchers::ContainsSubstring;

namespace {

constexpr Real kLambda = 500e-9;
using FieldMatrix = Eigen::Matrix<Complex, Eigen::Dynamic, 3>;

VectorXc random_vector(Index n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<Real> nd(0.0, 1.0);
    VectorXc v(n);
    for (Index i = 0; i < n; ++i) v(i) = Complex(nd(rng), nd(rng));
    return v;
}

template <class A, class B>
bool bitwise_equal(const A& a, const B& b) {
    return a.rows() == b.rows() && a.cols() == b.cols() && (a.array() == b.array()).all();
}

/// Cheap kernels (as test_block_preconditioner.cpp / test_incomplete_lu_preconditioner.cpp).
kernels::OperatorOptions cheap_options() {
    kernels::OperatorOptions opt;
    opt.outer_grading_levels = 0;
    opt.target_accuracy = 0.0;
    opt.quad_degree_far = 1;
    opt.quad_degree_near = 2;
    opt.quad_degree_sing = 2;
    opt.stretched_touching_points = 4;
    return opt;
}

/// The tiny rough open surface of test_incomplete_lu_preconditioner.cpp: patch 0.2, taper 0.2
/// (Gaussian patch, sigma 20 nm, Lc 200 nm, seed 29), gap 0.1, PML 0.1 um at h = 100 nm
/// (pml_start 0.4 um).
geometry::OpenSurfaceMesh tiny_rough_open() {
    geometry::OpenSurfaceParams op;
    op.patch_size = 0.2e-6;
    op.taper_width = 0.2e-6;
    op.gap_width = 0.1e-6;
    op.pml_width = 0.1e-6;
    geometry::RoughSurfaceParams rp;
    rp.mesh_size = 100e-9;
    rp.edge_length_L = op.patch_size + 2.0 * op.taper_width;
    rp.rms_roughness = 20e-9;
    rp.correlation_length = 200e-9;
    rp.seed = 29;
    return geometry::make_open_surface_mesh(geometry::generate_gaussian_height_map(rp), op);
}

/// Angular-spectrum beam (vacuum background unless given).
std::shared_ptr<excitation::AngularSpectrumBeam> beam(
    Real theta_deg, excitation::Polarization pol, Real region_radius = 0.75e-6,
    Real wavelength = kLambda, const material::Material& background = material::vacuum()) {
    excitation::AngularSpectrumBeam::Params bp;
    bp.wavelength = wavelength;
    bp.waist_radius = 0.4e-6;
    bp.incidence_angle = theta_deg * constants::pi / 180.0;
    bp.polarization = pol;
    bp.region_radius = region_radius;
    return std::make_shared<excitation::AngularSpectrumBeam>(bp, background);
}

/// The tiny PML case: n = 1.5 object, cheap kernels, sigma_max 1, compression "mlfmm" with no
/// preconditioner setting (the automatic ILUT of the route), GMRES 1e-9. The tiny surface has
/// no dark band (relaxed tolerance 0.9, as the structural PML tests).
struct TinyPml {
    TinyPml() {
        cfg.wavelength = kLambda;
        cfg.object = material::Material{Complex(2.25, 0.0), Complex(1.0, 0.0)};
        cfg.kernels = cheap_options();
        cfg.gmres.tolerance = 1e-9;
        cfg.gmres.verbose = false;
        cfg.compression = "mlfmm";
        term.sigma_max = 1.0;
        term.dark_band_tolerance = 0.9;
        term.correlation_length = 200e-9;
    }
    [[nodiscard]] std::unique_ptr<Simulation> make(
        std::shared_ptr<excitation::Excitation> exc) const {
        return std::make_unique<Simulation>(tiny_rough_open(), term, std::move(exc), cfg);
    }

    SimulationConfig cfg;
    PmlTermination term;
};

/// Reflection-hemisphere directions of the far-field checks.
Vertices reflection_directions() {
    Vertices d(4, 3);
    d << 0.0, 0.0, -1.0, 0.4, 0.0, -0.9, 0.0, 0.5, -0.8, -0.3, 0.3, -0.9;
    return d;
}

/// Every post-processing quantity of the PML route that the swap must reproduce, compared
/// bitwise with a fresh Simulation (same deterministic code on equal inputs).
void check_pml_post_equal(const Simulation& reused, const Simulation& fresh) {
    CHECK(reused.incident_power() == fresh.incident_power());
    CHECK(reused.dark_band_ratio() == fresh.dark_band_ratio());
    const Vertices dirs = reflection_directions();
    CHECK(bitwise_equal(reused.far_field(dirs), fresh.far_field(dirs)));
    CHECK(bitwise_equal(reused.drc(dirs), fresh.drc(dirs)));
    const post::PolarizedIntensity a = reused.drc_polarized(dirs, Vec3::UnitY());
    const post::PolarizedIntensity b = fresh.drc_polarized(dirs, Vec3::UnitY());
    CHECK(bitwise_equal(a.co, b.co));
    CHECK(bitwise_equal(a.cross, b.cross));
    const Real Ra = reused.reflected_power(12, 24) / reused.incident_power();
    const Real Rb = fresh.reflected_power(12, 24) / fresh.incident_power();
    INFO("R reused " << Ra << ", fresh " << Rb);
    CHECK(Ra == Rb);
    CHECK(Ra > 0.0);
}

}  // namespace

TEST_CASE("operator_reuse: set_excitation on the mlfmm PML route equals a fresh Simulation",
          "[operator_reuse]") {
    const TinyPml t;
    const auto b0s = beam(0.0, excitation::Polarization::S);
    const auto b20p = beam(20.0, excitation::Polarization::P);
    const auto sim = t.make(b0s);
    REQUIRE(sim->preconditioner_kind() == PreconditionerKind::IncompleteLu);
    const solver::GmresResult r1 = sim->solve();
    REQUIRE(r1.converged);
    const VectorXc x1 = sim->solution().currents;
    const std::shared_ptr<const op::LinearOperator> Z = sim->system_operator();
    const solver::IncompleteLuPreconditioner* P = sim->incomplete_lu_preconditioner();
    REQUIRE(P != nullptr);
    const VectorXc v = random_vector(sim->num_unknowns(), 7);
    VectorXc zv_before;
    Z->apply(v, zv_before);
    VectorXc pv_before;
    P->apply(v, pv_before);
    const VectorXc rhs_before = sim->rhs();

    sim->set_excitation(b20p);
    // Operator and preconditioner kept (same objects, same results); the right-hand side is new.
    CHECK(sim->system_operator().get() == Z.get());
    CHECK(sim->incomplete_lu_preconditioner() == P);
    VectorXc zv_after;
    sim->system_operator()->apply(v, zv_after);
    CHECK(bitwise_equal(zv_after, zv_before));
    VectorXc pv_after;
    sim->incomplete_lu_preconditioner()->apply(v, pv_after);
    CHECK(bitwise_equal(pv_after, pv_before));
    CHECK(sim->excitation() == b20p);
    CHECK(sim->problem().excitation == b20p.get());
    CHECK(sim->assembled());
    CHECK_FALSE(bitwise_equal(sim->rhs(), rhs_before));
    // Solution state reset: post-processing that needs the currents throws until the next solve.
    CHECK_FALSE(sim->solved());
    CHECK_THROWS_AS(sim->solution(), std::logic_error);
    const Vertices dirs = reflection_directions();
    CHECK_THROWS_AS(sim->far_field(dirs), std::logic_error);
    CHECK_THROWS_AS(sim->drc(dirs), std::logic_error);
    CHECK_THROWS_AS(sim->reflected_power(12, 24), std::logic_error);
    CHECK_THROWS_AS(sim->radiated_power(dirs, VectorXr::Ones(dirs.rows())), std::logic_error);
    CHECK_THAT(sim->report(), ContainsSubstring("solve:          not run"));
    // incident_power needs no solve and reads the current excitation.
    CHECK(sim->incident_power() == b20p->power());

    // A fresh Simulation with the new excitation: right-hand side, solution and post-processing
    // bitwise equal (one deterministic assembly, preconditioner and GMRES on equal inputs).
    const auto fresh = t.make(b20p);
    fresh->assemble();
    CHECK(bitwise_equal(sim->rhs(), fresh->rhs()));
    CHECK(sim->dark_band_ratio() == fresh->dark_band_ratio());
    const solver::GmresResult r2 = sim->solve();
    const solver::GmresResult f2 = fresh->solve();
    INFO("iterations reused " << r2.iterations << ", fresh " << f2.iterations);
    CHECK(r2.converged);
    CHECK(r2.iterations == f2.iterations);
    CHECK(r2.residual_history == f2.residual_history);
    CHECK(sim->solved());
    CHECK(bitwise_equal(sim->solution().currents, fresh->solution().currents));
    check_pml_post_equal(*sim, *fresh);

    // InitialGuess::Previous: back to the first excitation, started from the second one's
    // currents. The first residual is that of x0 in the ILUT-preconditioned (left) system, the
    // result the first solution to the tolerance's level.
    sim->set_excitation(b0s);
    CHECK(bitwise_equal(sim->rhs(), rhs_before));
    const VectorXc x0 = fresh->solution().currents;
    std::vector<Real> seen;
    const solver::GmresResult r3 =
        sim->solve(sim->config().gmres, SolveOptions{.initial_guess = InitialGuess::Previous},
                   [&](int, Real res) { seen.push_back(res); });
    CHECK(r3.converged);
    CHECK(seen.size() == static_cast<std::size_t>(r3.iterations));
    VectorXc zx;
    Z->apply(x0, zx);
    const VectorXc res0 = rhs_before - zx;
    VectorXc mr;
    VectorXc mb;
    P->apply(res0, mr);
    P->apply(rhs_before, mb);
    const Real expected = mr.norm() / mb.norm();
    INFO("initial residual " << r3.residual_history.front() << ", expected " << expected
                             << ", iterations " << r3.iterations << " (x0 = 0: " << r1.iterations
                             << ")");
    CHECK(std::abs(r3.residual_history.front() - expected) <= 1e-12 * expected);
    CHECK((sim->solution().currents - x1).norm() <= 1e-7 * x1.norm());
    CHECK(r3.true_relative_residual <= 1e-7);
    // Previous right after a solve (no set_excitation in between): x0 = the solution itself,
    // whose residual (~1e-9) meets a looser tolerance: no iteration, the currents unchanged.
    const VectorXc x3 = sim->solution().currents;
    solver::GmresParams loose = sim->config().gmres;
    loose.tolerance = 1e-6;
    const solver::GmresResult r4 =
        sim->solve(loose, SolveOptions{.initial_guess = InitialGuess::Previous});
    CHECK(r4.iterations == 0);
    CHECK(r4.converged);
    CHECK(r4.residual_history.front() <= 1e-8);
    CHECK(bitwise_equal(sim->solution().currents, x3));
}

TEST_CASE("operator_reuse: set_excitation checks leave the Simulation unchanged",
          "[operator_reuse]") {
    const TinyPml t;
    const auto b0s = beam(0.0, excitation::Polarization::S);
    const auto sim = t.make(b0s);
    REQUIRE(sim->solve().converged);
    const VectorXc rhs = sim->rhs();
    const VectorXc x = sim->solution().currents;
    const Real ratio = sim->dark_band_ratio();
    const auto unchanged = [&] {
        CHECK(sim->excitation() == b0s);
        CHECK(sim->problem().excitation == b0s.get());
        CHECK(bitwise_equal(sim->rhs(), rhs));
        CHECK(sim->solved());
        CHECK(bitwise_equal(sim->solution().currents, x));
        CHECK(sim->dark_band_ratio() == ratio);
    };
    CHECK_THROWS_WITH(sim->set_excitation(nullptr), ContainsSubstring("excitation is null"));
    unchanged();
    CHECK_THROWS_WITH(sim->set_excitation(beam(0.0, excitation::Polarization::P, 0.75e-6, 600e-9)),
                      ContainsSubstring("wavelength"));
    unchanged();
    const material::Material glass{Complex(2.25, 0.0), Complex(1.0, 0.0)};
    CHECK_THROWS_WITH(
        sim->set_excitation(beam(0.0, excitation::Polarization::P, 0.75e-6, kLambda, glass)),
        ContainsSubstring("background"));
    unchanged();
    // The unmasked vertices reach sqrt(2) 0.4 um = 0.57 um from the focus.
    CHECK_THROWS_WITH(sim->set_excitation(beam(0.0, excitation::Polarization::P, 0.5e-6)),
                      ContainsSubstring("unmasked part of the mesh"));
    unchanged();
    // A plane wave has no dark band (ratio 1 > 0.9).
    CHECK_THROWS_WITH(sim->set_excitation(std::make_shared<excitation::PlaneWave>(
                          kLambda, Vec3(0.0, 0.0, 1.0), Vec3c(1.0, 0.0, 0.0))),
                      ContainsSubstring("no dark band"));
    unchanged();
    // The same excitation again: accepted, the right-hand side bitwise the same, state reset.
    sim->set_excitation(b0s);
    CHECK(bitwise_equal(sim->rhs(), rhs));
    CHECK_FALSE(sim->solved());
}

TEST_CASE("operator_reuse: set_excitation before assemble behaves like the constructor",
          "[operator_reuse]") {
    const TinyPml t;
    const auto b0s = beam(0.0, excitation::Polarization::S);
    const auto b20p = beam(20.0, excitation::Polarization::P);
    const auto sim = t.make(b0s);
    sim->set_excitation(b20p);
    CHECK_FALSE(sim->assembled());
    CHECK_FALSE(sim->solved());
    CHECK(sim->excitation() == b20p);
    CHECK(sim->problem().omega == b20p->omega());
    const auto fresh = t.make(b20p);
    CHECK(sim->dark_band_ratio() == fresh->dark_band_ratio());
    const solver::GmresResult r = sim->solve();
    const solver::GmresResult f = fresh->solve();
    CHECK(r.iterations == f.iterations);
    CHECK(bitwise_equal(sim->rhs(), fresh->rhs()));
    CHECK(bitwise_equal(sim->solution().currents, fresh->solution().currents));
    check_pml_post_equal(*sim, *fresh);
}

TEST_CASE("operator_reuse: InitialGuess::Previous needs a previous GMRES solve",
          "[operator_reuse]") {
    const TinyPml t;
    const auto sim = t.make(beam(0.0, excitation::Polarization::S));
    const SolveOptions prev{.initial_guess = InitialGuess::Previous};
    CHECK_THROWS_WITH(sim->solve(sim->config().gmres, prev),
                      ContainsSubstring("without a previous solve()"));
    CHECK_THROWS_AS(sim->solve(sim->config().gmres, prev), std::logic_error);
    sim->set_excitation(beam(20.0, excitation::Polarization::P));  // still no previous solve
    CHECK_THROWS_AS(sim->solve(sim->config().gmres, prev), std::logic_error);
    CHECK_THROWS_AS(sim->solve(sim->config().gmres,
                               SolveOptions{.initial_guess = static_cast<InitialGuess>(7)}),
                    std::invalid_argument);
    CHECK(std::string(to_string(InitialGuess::Zero)) == "zero");
    CHECK(std::string(to_string(InitialGuess::Previous)) == "previous");

    // The direct solver takes no initial guess.
    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.object = material::silicon_500nm();
    cfg.solver = SolverKind::Direct;
    Simulation direct(
        geometry::make_icosphere(0.3e-6, 1),
        std::make_shared<excitation::PlaneWave>(kLambda, Vec3(0.0, 0.0, 1.0), Vec3c(1.0, 0.0, 0.0)),
        cfg);
    (void)direct.solve();
    CHECK_THROWS_WITH(direct.solve(cfg.gmres, prev), ContainsSubstring("needs the GMRES solver"));
    CHECK_THROWS_AS(direct.solve(cfg.gmres, prev), std::invalid_argument);
}

TEST_CASE("operator_reuse: set_excitation on the dense box route equals a fresh Simulation",
          "[operator_reuse]") {
    // Si sphere r = 0.3 um, icosphere n = 1 (2N = 240), dense, GMRES with left Jacobi; two
    // plane waves (+z with E along x, then oblique with E along y).
    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.object = material::silicon_500nm();
    cfg.preconditioner = PreconditionerKind::Diagonal;
    cfg.gmres.tolerance = 1e-10;
    cfg.gmres.verbose = false;
    const auto mesh = [] { return geometry::make_icosphere(0.3e-6, 1); };
    const auto w1 =
        std::make_shared<excitation::PlaneWave>(kLambda, Vec3(0.0, 0.0, 1.0), Vec3c(1.0, 0.0, 0.0));
    const Vec3 k2 = Vec3(0.5, 0.0, std::sqrt(0.75));
    const auto w2 = std::make_shared<excitation::PlaneWave>(kLambda, k2, Vec3c(0.0, 1.0, 0.0));
    Simulation sim(mesh(), w1, cfg);
    REQUIRE(sim.solve().converged);
    const VectorXc x1 = sim.solution().currents;
    const op::LinearOperator* Z = sim.system_operator().get();
    sim.set_excitation(w2);
    CHECK(sim.system_operator().get() == Z);
    CHECK_FALSE(sim.solved());
    CHECK_THROWS_AS(sim.scattered_power(6, 12), std::logic_error);
    CHECK_THROWS_AS(sim.solution(), std::logic_error);
    CHECK_THROWS_AS(sim.incident_power(), std::logic_error);  // a plane wave: infinite power
    Simulation fresh(mesh(), w2, cfg);
    fresh.assemble();
    CHECK(bitwise_equal(sim.rhs(), fresh.rhs()));
    const solver::GmresResult r = sim.solve();
    const solver::GmresResult f = fresh.solve();
    CHECK(r.converged);
    CHECK(r.iterations == f.iterations);
    CHECK(bitwise_equal(sim.solution().currents, fresh.solution().currents));
    CHECK(sim.solution().problem->excitation == w2.get());
    // Post-processing: far field, scattered power and the bistatic RCS (|E_inc| of the current
    // excitation) bitwise.
    Vertices dirs(3, 3);
    dirs << 0.0, 0.0, 1.0, 0.6, 0.0, -0.8, 0.0, -1.0, 0.0;
    CHECK(bitwise_equal(sim.far_field(dirs), fresh.far_field(dirs)));
    CHECK(sim.scattered_power(6, 12) == fresh.scattered_power(6, 12));
    const VectorXr theta = VectorXr::LinSpaced(7, 0.0, constants::pi);
    CHECK(bitwise_equal(post::bistatic_rcs(sim.solution(), Vec3::UnitY(), theta),
                        post::bistatic_rcs(fresh.solution(), Vec3::UnitY(), theta)));
    // The controlled region is checked on the whole closed mesh.
    excitation::AngularSpectrumBeam::Params bp;
    bp.wavelength = kLambda;
    bp.waist_radius = 1e-6;
    bp.region_radius = 0.2e-6;
    CHECK_THROWS_WITH(sim.set_excitation(std::make_shared<excitation::AngularSpectrumBeam>(bp)),
                      ContainsSubstring("the mesh extends to"));
    CHECK(sim.solved());  // the failed swap changed nothing
    // Previous: back to w1, started from w2's currents; the first solution is reproduced.
    sim.set_excitation(w1);
    const solver::GmresResult p =
        sim.solve(cfg.gmres, SolveOptions{.initial_guess = InitialGuess::Previous});
    INFO("iterations with x0 = previous " << p.iterations);
    CHECK(p.converged);
    CHECK((sim.solution().currents - x1).norm() <= 1e-8 * x1.norm());
}

TEST_CASE("operator_reuse: set_excitation after assemble refuses an omega off the operator's",
          "[operator_reuse]") {
    // Two plane waves at lambda (1 -+ 0.8e-12): each passes the constructor's wavelength check
    // against config.wavelength (relative 1e-12), but they differ from each other by 1.6e-12.
    // After assemble() the operator's omega is kept and op::validate refuses the second one;
    // before assemble() it is accepted (its omega replaces the first one's, as a constructor).
    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.object = material::silicon_500nm();
    cfg.preconditioner = PreconditionerKind::Diagonal;
    cfg.gmres.tolerance = 1e-10;
    cfg.gmres.verbose = false;
    const auto mesh = [] { return geometry::make_icosphere(0.3e-6, 1); };
    const auto wave = [](Real wavelength) {
        return std::make_shared<excitation::PlaneWave>(wavelength, Vec3(0.0, 0.0, 1.0),
                                                       Vec3c(1.0, 0.0, 0.0));
    };
    const auto lo = wave(kLambda * (1.0 - 0.8e-12));
    const auto hi = wave(kLambda * (1.0 + 0.8e-12));
    REQUIRE(lo->omega() != hi->omega());
    REQUIRE(std::abs(hi->omega() - lo->omega()) > 1e-12 * lo->omega());

    Simulation before(mesh(), lo, cfg);  // the constructor accepts either one
    before.set_excitation(hi);
    CHECK(before.excitation() == hi);
    CHECK(before.problem().omega == hi->omega());

    Simulation sim(mesh(), lo, cfg);
    REQUIRE(sim.solve().converged);
    const VectorXc rhs = sim.rhs();
    const VectorXc x = sim.solution().currents;
    const Real omega = sim.problem().omega;
    CHECK(omega == lo->omega());
    CHECK_THROWS_AS(sim.set_excitation(hi), std::invalid_argument);
    CHECK_THROWS_WITH(sim.set_excitation(hi), ContainsSubstring("omega differs"));
    // Unchanged: excitation, problem, right-hand side, solution state and currents.
    CHECK(sim.excitation() == lo);
    CHECK(sim.problem().excitation == lo.get());
    CHECK(sim.problem().omega == omega);
    CHECK(bitwise_equal(sim.rhs(), rhs));
    CHECK(sim.solved());
    CHECK(bitwise_equal(sim.solution().currents, x));
    CHECK(sim.solution().problem->excitation == lo.get());
}
