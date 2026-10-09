#pragma once
/// @file simulation.hpp
/// High-level driver that wires everything together (top layer, docs/02). This is also the
/// surface exposed to Python (docs/10): build a Simulation from a mesh, an excitation and a
/// config, assemble, solve, post-process with post::* on solution().
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/geometry/mesh.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/postprocessing/fields.hpp"
#include "specklebem/solver/gmres.hpp"

#include <memory>
#include <optional>
#include <string>

namespace specklebem {

/// Linear solver of Simulation::solve().
enum class SolverKind {
    Gmres,  ///< solver::gmres with config.gmres or the per-call parameters (incl. the side)
    Direct  ///< solver::solve_direct (dense LU) on the dense operator
};

struct SimulationConfig {
    /// Vacuum wavelength [m]; must equal the excitation's wavelength (relative 1e-12).
    Real wavelength = 500e-9;
    /// Region R1; must equal the excitation's background (the incident field lives in R1).
    material::Material exterior = material::vacuum();
    material::Material object = material::silicon_500nm();  ///< region R2
    /// Empty: automatic, formulation::recommend(object.eps_r).kind.
    std::optional<formulation::Kind> formulation;
    /// Left/right Jacobi preconditioner (op::assemble_diagonal) of the GMRES solve; empty: the
    /// recommendation's choice, formulation::recommend(object.eps_r).diagonal_preconditioner
    /// (also when the formulation is set explicitly). Ignored by the direct solver.
    std::optional<bool> diagonal_preconditioner;
    /// "dense" (default until Phase 4). "mlfmm" (Phase 4), "aca" and "hmatrix" (Phase 8) are
    /// rejected with std::invalid_argument until they are implemented, as is any other name.
    std::string compression = "dense";
    mlfmm::MlfmmParams mlfmm;  ///< unused until Phase 4
    solver::GmresParams gmres;
    kernels::OperatorOptions kernels;
    SolverKind solver = SolverKind::Gmres;
};

/// Owns everything a solve needs: the mesh, the RWG space, the formulation, the excitation
/// (shared), the op::Problem (omega = 2 pi c0 / wavelength), the operator, the right-hand side,
/// the Jacobi diagonal (if used) and the solution. solution().problem points into this object,
/// so a SurfaceSolution obtained from solution() is valid while the Simulation lives. Not
/// copyable or movable (the RWG space and the Problem hold pointers into it).
class Simulation {
public:
    /// Cheap (no assembly): validates and resolves the configuration. The angular frequency is
    /// the excitation's, excitation->omega() (its wavelength is checked against
    /// config.wavelength).
    /// @throws std::invalid_argument for a mesh that is not closed or whose normals point
    ///         inward (signed_volume() <= 0; they must point out of R2 into R1, docs/06), a
    ///         null excitation, config.wavelength not finite and > 0 or different from the
    ///         excitation's wavelength (relative 1e-12), an excitation background (eps_r or
    ///         mu_r) different from config.exterior, an unknown or unavailable compression,
    ///         2N > op::kMaxDenseUnknowns for the dense strategy, a formulation that is not
    ///         implemented (formulation::Kind::JMCFIE), invalid kernel options or GMRES
    ///         parameters (tolerance not finite or <= 0, max_iter < 1, restart < 0), and
    ///         everything op::validate rejects.
    Simulation(geometry::TriangleMesh mesh, std::shared_ptr<excitation::Excitation> excitation,
               SimulationConfig config);
    ~Simulation();
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;
    Simulation(Simulation&&) = delete;
    Simulation& operator=(Simulation&&) = delete;

    /// Assembles the operator, the right-hand side and (GMRES with the Jacobi preconditioner
    /// only) the diagonal, and times them. Idempotent: what is already assembled is kept, so
    /// later calls do nothing (after an exception a later call retries the missing parts).
    void assemble();
    /// Same as solve(config().gmres, cb).
    solver::GmresResult solve(const solver::IterationCallback& cb = {});
    /// Assembles if needed and solves with the GMRES parameters `params` for this call only
    /// (config().gmres is not changed); the solution replaces any previous one, also when GMRES
    /// does not converge (converged = false, logged as a warning, shown by report()). The
    /// Jacobi diagonal is built here if the GMRES solve uses it and it is not assembled yet.
    ///
    /// GMRES: the result of solver::gmres (callback cb per iteration). Direct: params are
    /// validated but unused; the dense LU result is mapped onto GmresResult following the
    /// size rule of residual_history (iterations + 1 entries): iterations 0, converged true,
    /// residual_history {residual} and true_relative_residual = residual, the LU residual
    /// |b - Z x| / |b| (DirectSolveInfo::residual; {0} for b = 0), wall_seconds the LU time;
    /// cb is not called.
    /// @throws std::invalid_argument for invalid params (tolerance not finite or <= 0,
    ///         max_iter < 1, restart < 0). std::runtime_error from the solvers (singular
    ///         system, non-finite values).
    solver::GmresResult solve(const solver::GmresParams& params,
                              const solver::IterationCallback& cb = {});
    /// @throws std::logic_error before the first solve().
    [[nodiscard]] const post::SurfaceSolution& solution() const;
    /// Multi-line report: mesh size, formulation / preconditioner used (and whether automatic),
    /// solver (with the GMRES parameters of the last solve, config().gmres before the first),
    /// compression, operator memory, timings, iterations and residuals.
    [[nodiscard]] std::string report() const;

    [[nodiscard]] const geometry::TriangleMesh& mesh() const;
    [[nodiscard]] const SimulationConfig& config() const;
    /// The problem (space, materials, formulation, excitation, omega) owned by this object.
    [[nodiscard]] const op::Problem& problem() const;
    /// Formulation actually used (config.formulation or the recommendation).
    [[nodiscard]] formulation::Kind formulation_kind() const;
    /// Whether the Jacobi preconditioner is used by the GMRES solve (resolved setting).
    [[nodiscard]] bool diagonal_preconditioner() const;
    /// Number of unknowns 2N.
    [[nodiscard]] Index num_unknowns() const;
    /// System operator and right-hand side. @throws std::logic_error before assemble().
    [[nodiscard]] std::shared_ptr<const op::LinearOperator> system_operator() const;
    [[nodiscard]] const VectorXc& rhs() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace specklebem
