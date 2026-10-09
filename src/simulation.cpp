#include "specklebem/simulation.hpp"

#include "specklebem/basis/rwg.hpp"
#include "specklebem/core/logging.hpp"
#include "specklebem/core/timer.hpp"
#include "specklebem/kernels/operators.hpp"
#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/solver/direct.hpp"
#include "specklebem/solver/preconditioner.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace specklebem {

namespace {

constexpr Real kRelTol = 1e-12;

bool close(Complex a, Complex b) {
    return std::abs(a - b) <= kRelTol * std::max(std::abs(a), std::abs(b));
}

/// Accepts "dense"; rejects the planned strategies and unknown names.
void check_compression(const std::string& name) {
    if (name == "dense")
        return;
    if (name == "mlfmm") {
        throw std::invalid_argument(
            "Simulation: compression \"mlfmm\" is not available before Phase 4; use \"dense\"");
    }
    if (name == "aca" || name == "hmatrix") {
        throw std::invalid_argument("Simulation: compression \"" + name +
                                    "\" is not available before Phase 8; use \"dense\"");
    }
    throw std::invalid_argument("Simulation: unknown compression \"" + name +
                                "\" (expected \"dense\", \"mlfmm\", \"aca\" or \"hmatrix\")");
}

void check_gmres_params(const solver::GmresParams& g) {
    if (!(std::isfinite(g.tolerance) && g.tolerance > 0) || g.max_iter < 1 || g.restart < 0) {
        throw std::invalid_argument(
            "Simulation: GMRES parameters need a finite tolerance > 0, max_iter >= 1 and "
            "restart >= 0");
    }
}

const char* side_name(solver::PreconditionerSide s) {
    return s == solver::PreconditionerSide::Left ? "left" : "right";
}

}  // namespace

struct Simulation::Impl {
    Impl(geometry::TriangleMesh m, std::shared_ptr<excitation::Excitation> exc,
         SimulationConfig cfg)
        : config(std::move(cfg)), mesh(std::move(m)), space(mesh), excitation(std::move(exc)) {}

    SimulationConfig config;
    geometry::TriangleMesh mesh;
    basis::RwgSpace space;  ///< refers to mesh (Impl is heap-allocated and never moved)
    std::shared_ptr<excitation::Excitation> excitation;
    std::unique_ptr<formulation::Formulation> form;
    formulation::Kind kind = formulation::Kind::ICTF;
    bool kind_auto = false;
    bool jacobi = false;
    bool jacobi_auto = false;
    op::Problem problem;

    std::shared_ptr<op::LinearOperator> op;  ///< null until assemble()
    VectorXc rhs;
    VectorXc diagonal;  ///< empty unless the Jacobi preconditioner is used
    double assembly_s = 0, rhs_s = 0, diagonal_s = 0, solve_s = 0;

    bool solved = false;
    post::SurfaceSolution solution;
    solver::GmresResult result;  ///< last result (x moved into solution.currents)
    solver::DirectSolveInfo direct_info;
};

Simulation::Simulation(geometry::TriangleMesh mesh,
                       std::shared_ptr<excitation::Excitation> excitation, SimulationConfig config) {
    if (!mesh.is_closed() || !mesh.is_consistently_oriented()) {
        throw std::invalid_argument(
            "Simulation: the mesh must be closed and consistently oriented (" +
            std::to_string(mesh.num_boundary_edges()) + " boundary edges)");
    }
    if (excitation == nullptr) {
        throw std::invalid_argument("Simulation: excitation is null");
    }
    const Real lambda = config.wavelength;
    if (!(std::isfinite(lambda) && lambda > 0)) {
        throw std::invalid_argument("Simulation: wavelength must be finite and > 0");
    }
    if (!(std::abs(excitation->wavelength() - lambda) <= kRelTol * lambda)) {
        throw std::invalid_argument("Simulation: config.wavelength differs from the excitation's "
                                    "wavelength");
    }
    const material::Material& bg = excitation->background();
    if (!close(bg.eps_r, config.exterior.eps_r) || !close(bg.mu_r, config.exterior.mu_r)) {
        throw std::invalid_argument(
            "Simulation: the excitation's background differs from config.exterior");
    }
    check_compression(config.compression);
    kernels::validate(config.kernels);
    check_gmres_params(config.gmres);

    impl_ = std::make_unique<Impl>(std::move(mesh), std::move(excitation), std::move(config));
    Impl& s = *impl_;
    const formulation::Recommendation rec = formulation::recommend(s.config.object.eps_r);
    s.kind_auto = !s.config.formulation.has_value();
    s.kind = s.config.formulation.value_or(rec.kind);
    s.jacobi_auto = !s.config.diagonal_preconditioner.has_value();
    s.jacobi = s.config.diagonal_preconditioner.value_or(rec.diagonal_preconditioner);
    s.form = formulation::make_formulation(s.kind);

    s.problem.space = &s.space;
    s.problem.exterior = s.config.exterior;
    s.problem.object = s.config.object;
    s.problem.formulation = s.form.get();
    s.problem.excitation = s.excitation.get();
    s.problem.kernel_options = s.config.kernels;
    // Same expression as excitation::PlaneWave / GaussianBeam, so equal wavelengths give
    // bitwise equal omegas.
    s.problem.omega = 2 * constants::pi * constants::c0 / lambda;
    op::validate(s.problem);
}

Simulation::~Simulation() = default;

void Simulation::assemble() {
    Impl& s = *impl_;
    if (s.op != nullptr)
        return;
    // Built into locals and committed at the end, so that an exception leaves the object
    // unassembled (and a later call retries).
    std::shared_ptr<op::LinearOperator> Z;
    VectorXc b, d;
    {
        const ScopedTimer t("Simulation: dense assembly of Z");
        Z = op::DenseStrategy().build(s.problem);
        s.assembly_s = t.elapsed_seconds();
    }
    {
        const ScopedTimer t("Simulation: right-hand side");
        b = op::assemble_rhs(s.problem);
        s.rhs_s = t.elapsed_seconds();
    }
    if (s.config.solver == SolverKind::Gmres && s.jacobi) {
        const ScopedTimer t("Simulation: Jacobi diagonal");
        d = op::assemble_diagonal(s.problem);
        s.diagonal_s = t.elapsed_seconds();
    }
    s.rhs = std::move(b);
    s.diagonal = std::move(d);
    s.op = std::move(Z);
}

solver::GmresResult Simulation::solve(const solver::IterationCallback& cb) {
    assemble();
    Impl& s = *impl_;
    solver::GmresResult res;
    if (s.config.solver == SolverKind::Direct) {
        const auto* dense = dynamic_cast<const op::DenseOperator*>(s.op.get());
        if (dense == nullptr) {
            throw std::logic_error("Simulation: the direct solver needs a dense operator");
        }
        const ScopedTimer t("Simulation: direct LU solve");
        res.x = solver::solve_direct(*dense, s.rhs, &s.direct_info);
        res.wall_seconds = t.elapsed_seconds();
        res.iterations = 0;
        res.converged = true;
        res.true_relative_residual = s.direct_info.residual;
        res.residual_history = {1.0, s.direct_info.residual};
    } else if (s.jacobi) {
        const solver::DiagonalPreconditioner M(s.diagonal);
        res = solver::gmres(*s.op, s.rhs, M, s.config.gmres, cb);
    } else {
        res = solver::gmres(*s.op, s.rhs, solver::IdentityPreconditioner(), s.config.gmres, cb);
    }
    s.solve_s = res.wall_seconds;
    s.solution.problem = &s.problem;
    s.solution.currents = res.x;
    s.result = res;
    s.result.x = VectorXc();  // the currents are kept once, in s.solution
    s.solved = true;
    SBEM_INFO("Simulation: {} {}{}, 2N = {}: {} it, converged {}, true residual {:.3e}, "
              "assembly {:.3f} s, solve {:.3f} s",
              s.form->name(),
              s.config.solver == SolverKind::Direct ? "direct LU" : "GMRES",
              s.config.solver == SolverKind::Gmres && s.jacobi ? " + Jacobi" : "",
              num_unknowns(), res.iterations, res.converged, res.true_relative_residual,
              s.assembly_s + s.rhs_s + s.diagonal_s, s.solve_s);
    return res;
}

const post::SurfaceSolution& Simulation::solution() const {
    if (!impl_->solved) {
        throw std::logic_error("Simulation: solution() called before solve()");
    }
    return impl_->solution;
}

std::string Simulation::report() const {
    const Impl& s = *impl_;
    const bool direct = s.config.solver == SolverKind::Direct;
    std::ostringstream o;
    o << "SpeckleBem simulation\n";
    o << "  mesh:           " << s.mesh.num_triangles() << " triangles, N = " << s.space.size()
      << " RWG functions, 2N = " << num_unknowns() << " unknowns\n";
    o << "  wavelength:     " << s.config.wavelength << " m, object eps_r = "
      << s.config.object.eps_r << ", exterior eps_r = " << s.config.exterior.eps_r << "\n";
    o << "  formulation:    " << s.form->name() << (s.kind_auto ? " (automatic)" : " (explicit)")
      << "\n";
    o << "  preconditioner: " << (s.jacobi ? "diagonal (Jacobi)" : "none")
      << (s.jacobi_auto ? " (automatic)" : " (explicit)");
    if (direct)
        o << ", unused by the direct solver";
    else if (s.jacobi)
        o << ", " << side_name(s.config.gmres.side) << " side";
    o << "\n";
    o << "  solver:         "
      << (direct ? std::string("direct LU")
                 : "GMRES (tol " + std::to_string(s.config.gmres.tolerance) + ", restart " +
                       std::to_string(s.config.gmres.restart) + ")")
      << "\n";
    o << "  compression:    " << s.config.compression << "\n";
    if (s.op == nullptr) {
        o << "  operator:       not assembled\n";
        return o.str();
    }
    o << "  operator:       " << s.op->describe() << ", "
      << static_cast<double>(s.op->memory_bytes()) / 1e6 << " MB\n";
    o << "  assembly:       Z " << s.assembly_s << " s, rhs " << s.rhs_s << " s";
    if (s.diagonal.size() > 0)
        o << ", diagonal " << s.diagonal_s << " s";
    o << "\n";
    if (!s.solved) {
        o << "  solve:          not run\n";
        return o.str();
    }
    o << "  solve:          " << s.solve_s << " s, " << s.result.iterations << " iterations, "
      << (s.result.converged ? "converged" : "NOT converged") << "\n";
    o << "  residuals:      monitored " << s.result.residual_history.back() << ", true "
      << s.result.true_relative_residual;
    if (direct)
        o << ", rcond " << s.direct_info.rcond;
    o << "\n";
    return o.str();
}

const geometry::TriangleMesh& Simulation::mesh() const {
    return impl_->mesh;
}

const SimulationConfig& Simulation::config() const {
    return impl_->config;
}

const op::Problem& Simulation::problem() const {
    return impl_->problem;
}

formulation::Kind Simulation::formulation_kind() const {
    return impl_->kind;
}

bool Simulation::diagonal_preconditioner() const {
    return impl_->jacobi;
}

Index Simulation::num_unknowns() const {
    return 2 * impl_->space.size();
}

std::shared_ptr<const op::LinearOperator> Simulation::system_operator() const {
    if (impl_->op == nullptr) {
        throw std::logic_error("Simulation: system_operator() called before assemble()");
    }
    return impl_->op;
}

const VectorXc& Simulation::rhs() const {
    if (impl_->op == nullptr) {
        throw std::logic_error("Simulation: rhs() called before assemble()");
    }
    return impl_->rhs;
}

}  // namespace specklebem
