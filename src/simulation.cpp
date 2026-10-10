#include "specklebem/simulation.hpp"

#include "specklebem/basis/rwg.hpp"
#include "specklebem/compression/mlfmm/far_operator.hpp"
#include "specklebem/compression/mlfmm/interpolation.hpp"
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
#include <string>
#include <utility>

namespace specklebem {

namespace {

constexpr Real kRelTol = 1e-12;

bool close(Complex a, Complex b) {
    return std::abs(a - b) <= kRelTol * std::max(std::abs(a), std::abs(b));
}

/// Accepts "dense" and "mlfmm"; rejects the planned strategies and unknown names.
void check_compression(const std::string& name) {
    if (name == "dense" || name == "mlfmm")
        return;
    if (name == "aca" || name == "hmatrix") {
        throw std::invalid_argument("Simulation: compression \"" + name +
                                    "\" is not available before Phase 8; use \"dense\" or "
                                    "\"mlfmm\"");
    }
    throw std::invalid_argument("Simulation: unknown compression \"" + name +
                                "\" (expected \"dense\", \"mlfmm\", \"aca\" or \"hmatrix\")");
}

/// Cheap checks of the MLFMM parameters (the octree and far-operator constructors repeat them at
/// assembly), so that a bad configuration fails at construction.
void check_mlfmm_params(const mlfmm::MlfmmParams& m) {
    const mlfmm::OctreeParams& o = m.octree;
    if (o.max_elements_per_leaf < 1 || o.max_levels < 1 || o.max_levels > mlfmm::kMaxOctreeLevels ||
        !(std::isfinite(o.min_box_size_lambda) && o.min_box_size_lambda >= 0.0)) {
        throw std::invalid_argument(
            "Simulation: MLFMM octree parameters need max_elements_per_leaf >= 1, max_levels in "
            "[1, " +
            std::to_string(mlfmm::kMaxOctreeLevels) + "] and a finite min_box_size_lambda >= 0");
    }
    if (m.truncation_L != 0 || !m.precompute_translators || m.use_fft_interpolation) {
        throw std::invalid_argument(
            "Simulation: unsupported MLFMM parameters (truncation_L must be 0, "
            "precompute_translators true, use_fft_interpolation false)");
    }
    (void)mlfmm::interpolation_order(m.accuracy_digits);  // d0 in (0, 5]
}

/// The incident field is evaluated at surface quadrature points, which lie in the convex hull
/// of the vertices; every vertex must lie in the ball where the excitation is controlled
/// (AngularSpectrumBeam: |r - focus| <= region_radius; infinite for the other sources).
void check_controlled_region(const geometry::TriangleMesh& mesh,
                             const excitation::Excitation& exc) {
    const Real radius = exc.controlled_radius();
    if (std::isnan(radius) || !(radius > 0)) {
        throw std::invalid_argument("Simulation: the excitation's controlled_radius() must be > 0");
    }
    if (std::isinf(radius))
        return;
    const Vec3 center = exc.controlled_center();
    Real extent = 0;
    const auto& v = mesh.vertices();
    for (Index i = 0; i < v.rows(); ++i) {
        extent = std::max(extent, (v.row(i).transpose() - center).norm());
    }
    if (extent > radius) {
        // Suggested radius: the extent rounded up to 4 significant digits.
        const Real scale = std::pow(10.0, std::floor(std::log10(extent)) - 3);
        const Real suggested = std::ceil(extent / scale) * scale;
        std::ostringstream msg;
        msg.precision(4);
        msg << "Simulation: the mesh extends to " << extent << " m from the excitation's centre ("
            << center.x() << ", " << center.y() << ", " << center.z()
            << ") m, but its fields are controlled only within " << radius
            << " m; set AngularSpectrumBeam region_radius >= " << suggested << " m";
        throw std::invalid_argument(msg.str());
    }
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

namespace {

/// Field decay length of the object, finite (decaying field) or std::invalid_argument.
Real decaying_field_length(const material::Material& object, Real wavelength, const char* caller) {
    if (object.eps_r.imag() > 0.0 || object.mu_r.imag() > 0.0) {
        throw std::invalid_argument(std::string(caller) +
                                    ": Im(eps_r) > 0 or Im(mu_r) > 0 is an active medium in the "
                                    "exp(+jwt) convention (docs/06)");
    }
    const Real delta = material::field_decay_length(object, wavelength);
    if (!std::isfinite(delta)) {
        throw std::invalid_argument(
            std::string(caller) +
            ": the object's field does not decay (lossless dielectric); ADR 0006 covers only "
            "absorbing substrates, pass the box depth explicitly");
    }
    return delta;
}

}  // namespace

Real default_box_depth(const material::Material& object, Real wavelength) {
    const Real delta = decaying_field_length(object, wavelength, "default_box_depth");
    const Real intensity_absorption_length = 0.5 * delta;  // 1 / alpha
    const Real depth = std::max(10.0 * intensity_absorption_length, kMinBoxDepth);
    if (depth > kLargeBoxDepthWarning) {
        SBEM_WARN(
            "default_box_depth: {:.3g} m for a weakly absorbing object (delta = {:.3g} m); "
            "ADR 0006 assumes absorbing substrates, check the box size",
            depth, delta);
    }
    return depth;
}

Real default_box_fine_depth(const material::Material& object, Real wavelength, Real sigma) {
    if (!(std::isfinite(sigma) && sigma >= 0.0)) {
        throw std::invalid_argument("default_box_fine_depth: sigma must be finite and >= 0");
    }
    const Real delta = decaying_field_length(object, wavelength, "default_box_fine_depth");
    return 3.0 * delta + 3.0 * sigma;
}

Real exterior_wavelength(const material::Material& background, Real wavelength) {
    if (!(std::isfinite(wavelength) && wavelength > 0.0)) {
        throw std::invalid_argument("exterior_wavelength: the wavelength must be finite and > 0");
    }
    if (!(std::isfinite(background.eps_r.real()) && std::isfinite(background.eps_r.imag()) &&
          std::isfinite(background.mu_r.real()) && std::isfinite(background.mu_r.imag()))) {
        throw std::invalid_argument("exterior_wavelength: eps_r and mu_r must be finite");
    }
    if (background.eps_r.imag() > 0.0 || background.mu_r.imag() > 0.0) {
        throw std::invalid_argument(
            "exterior_wavelength: Im(eps_r) > 0 or Im(mu_r) > 0 is an active medium in the "
            "exp(+jwt) convention (docs/06)");
    }
    if (!(background.eps_r.real() > 0.0)) {
        throw std::invalid_argument(
            "exterior_wavelength: a metallic background (Re(eps_r) <= 0) has no propagating "
            "exterior and is not covered by the closing box (ADR 0006)");
    }
    const Complex n = background.refractive_index();
    if (!(n.real() > 0.0)) {
        throw std::invalid_argument(
            "exterior_wavelength: the background needs Re(n) > 0 (propagating exterior)");
    }
    // |n_1| >= Re(n_1): conservative (shorter lambda_1) for a lossy background.
    return wavelength / std::abs(n);
}

Real default_box_mesh_size(const material::Material& background, Real wavelength, Real mesh_size) {
    if (!(std::isfinite(mesh_size) && mesh_size > 0.0)) {
        throw std::invalid_argument("default_box_mesh_size: mesh_size must be finite and > 0");
    }
    return geometry::box_spacing_for_exterior_wavelength(
        exterior_wavelength(background, wavelength), mesh_size);
}

void RoughBoxParams::apply_to(geometry::RoughSurfaceParams& p) const {
    p.box_depth = box_depth;
    p.box_fine_depth = box_fine_depth;
    p.box_mesh_size = box_mesh_size;
    p.exterior_wavelength = exterior_wavelength;
}

RoughBoxParams rough_surface_box_params(const material::Material& object,
                                        const material::Material& background, Real wavelength,
                                        Real sigma, Real mesh_size) {
    if (!(std::isfinite(mesh_size) && mesh_size > 0.0)) {
        throw std::invalid_argument("rough_surface_box_params: mesh_size must be finite and > 0");
    }
    RoughBoxParams b;
    b.box_depth = default_box_depth(object, wavelength);
    const Real fine_depth = default_box_fine_depth(object, wavelength, sigma);
    // box_mesh_size stays unset: the generator caps its automatic rule at lambda_1 / 5 with
    // the actual grid spacing L / round(L / h), which mesh_size does not know.
    b.exterior_wavelength = exterior_wavelength(background, wavelength);
    const Real delta = material::field_decay_length(object, wavelength);
    const bool weakly_absorbing = 3.0 * delta > geometry::kNoBandCoarseningDepth * mesh_size;
    if (!weakly_absorbing)
        return b;
    if (fine_depth < b.box_depth) {
        b.box_fine_depth = fine_depth;
    } else {
        SBEM_WARN(
            "rough_surface_box_params: the fine band 3 delta + 3 sigma = {:.3g} m reaches the "
            "bottom plate at {:.3g} m; using the uniform box (box_mesh_size = mesh_size), "
            "several times the cost of the graded box",
            fine_depth, b.box_depth);
        b.box_mesh_size = mesh_size;
        b.uniform_fallback = true;
    }
    return b;
}

void check_beam_waist(Real patch_length, Real waist, const BeamWaistOptions& options) {
    if (!(std::isfinite(patch_length) && patch_length > 0.0) ||
        !(std::isfinite(waist) && waist > 0.0)) {
        throw std::invalid_argument(
            "check_beam_waist: patch_length and waist must be finite and > 0");
    }
    const Real theta = options.incidence_angle;
    if (!(std::isfinite(theta) && std::abs(theta) < 0.5 * constants::pi)) {
        std::ostringstream os;
        os << "check_beam_waist: incidence_angle must be finite with |theta_in| < pi / 2, got "
           << theta << " rad";
        throw std::invalid_argument(os.str());
    }
    const Real limit = kMaxBeamWaistFraction * patch_length * std::cos(theta);
    if (!(waist > limit * (1.0 + 1e-12)))
        return;
    std::ostringstream os;
    os << "beam waist " << waist << " m exceeds L cos(theta_in) / 4 = " << limit
       << " m for the patch edge L = " << patch_length << " m and theta_in = " << theta
       << " rad (ADR 0006 amendments 2026-10-10/11: w0 <= L cos(theta_in) / 4; past the patch "
          "edges the beam carries 0.6-0.9 % of its power at L / 3 and 0 deg, 0.63 % at L / 4 "
          "and 45 deg)";
    if (!options.allow_wide)
        throw std::invalid_argument("check_beam_waist: " + os.str() +
                                    "; set allow_wide to accept it");
    SBEM_WARN("check_beam_waist: {} (allowed by the caller)", os.str());
}

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

    /// GMRES with the Jacobi preconditioner (the only solve that needs the diagonal).
    [[nodiscard]] bool uses_diagonal() const;
    /// Assembles the diagonal once if uses_diagonal(); no-op otherwise.
    void ensure_diagonal();

    bool solved = false;
    solver::GmresParams last_gmres;  ///< parameters of the last solve (for report())
    post::SurfaceSolution solution;
    solver::GmresResult result;  ///< last result (x moved into solution.currents)
    solver::DirectSolveInfo direct_info;
};

Simulation::Simulation(geometry::TriangleMesh mesh,
                       std::shared_ptr<excitation::Excitation> excitation,
                       SimulationConfig config) {
    // A constructed TriangleMesh is always consistently oriented (the constructor repairs the
    // orientation) and its closed components point outward, but flip_normals() can turn them
    // inward afterwards; the SIE needs normals out of R2 into R1 (docs/06).
    if (!mesh.is_closed()) {
        throw std::invalid_argument("Simulation: the mesh must be closed (" +
                                    std::to_string(mesh.num_boundary_edges()) + " boundary edges)");
    }
    if (!(mesh.signed_volume() > 0)) {
        throw std::invalid_argument(
            "Simulation: the mesh normals point inward (signed volume " +
            std::to_string(mesh.signed_volume()) +
            " m^3); they must point out of the object R2 into R1 (flip_normals())");
    }
    if (excitation == nullptr) {
        throw std::invalid_argument("Simulation: excitation is null");
    }
    const Real lambda = config.wavelength;
    if (!(std::isfinite(lambda) && lambda > 0)) {
        throw std::invalid_argument("Simulation: wavelength must be finite and > 0");
    }
    if (!(std::abs(excitation->wavelength() - lambda) <= kRelTol * lambda)) {
        throw std::invalid_argument(
            "Simulation: config.wavelength differs from the excitation's "
            "wavelength");
    }
    const material::Material& bg = excitation->background();
    if (!close(bg.eps_r, config.exterior.eps_r) || !close(bg.mu_r, config.exterior.mu_r)) {
        throw std::invalid_argument(
            "Simulation: the excitation's background differs from config.exterior");
    }
    check_controlled_region(mesh, *excitation);
    check_compression(config.compression);
    // Closed mesh: every edge is interior and carries one RWG function (N = num_edges()).
    const Index unknowns = 2 * mesh.num_edges();
    if (config.compression == "mlfmm")
        check_mlfmm_params(config.mlfmm);
    if (config.compression == "mlfmm" && config.solver == SolverKind::Direct) {
        throw std::invalid_argument(
            "Simulation: the direct solver needs the dense operator; use compression \"dense\" "
            "or the GMRES solver with \"mlfmm\"");
    }
    if (config.compression == "dense" && unknowns > op::kMaxDenseUnknowns) {
        throw std::invalid_argument("Simulation: 2N = " + std::to_string(unknowns) +
                                    " unknowns exceed the dense limit of " +
                                    std::to_string(op::kMaxDenseUnknowns) +
                                    " (16 (2N)^2 bytes of matrix storage)");
    }
    kernels::validate(config.kernels);
    check_gmres_params(config.gmres);

    const formulation::Recommendation rec = formulation::recommend(config.object.eps_r);
    const formulation::Kind kind = config.formulation.value_or(rec.kind);
    std::unique_ptr<formulation::Formulation> form;
    try {
        form = formulation::make_formulation(kind);
    } catch (const std::logic_error& e) {  // JMCFIE (reserved) or an unknown Kind
        throw std::invalid_argument(std::string("Simulation: formulation not available: ") +
                                    e.what());
    }

    impl_ = std::make_unique<Impl>(std::move(mesh), std::move(excitation), std::move(config));
    Impl& s = *impl_;
    s.kind_auto = !s.config.formulation.has_value();
    s.kind = kind;
    s.jacobi_auto = !s.config.diagonal_preconditioner.has_value();
    s.jacobi = s.config.diagonal_preconditioner.value_or(rec.diagonal_preconditioner);
    s.form = std::move(form);

    s.problem.space = &s.space;
    s.problem.exterior = s.config.exterior;
    s.problem.object = s.config.object;
    s.problem.formulation = s.form.get();
    s.problem.excitation = s.excitation.get();
    s.problem.kernel_options = s.config.kernels;
    // The excitation's omega (its wavelength equals config.wavelength, checked above), so the
    // Problem and the incident field use the same frequency.
    s.problem.omega = s.excitation->omega();
    op::validate(s.problem);
}

Simulation::~Simulation() = default;

bool Simulation::Impl::uses_diagonal() const {
    return config.solver == SolverKind::Gmres && jacobi;
}

void Simulation::Impl::ensure_diagonal() {
    if (!uses_diagonal() || diagonal.size() > 0)
        return;
    const ScopedTimer t("Simulation: Jacobi diagonal");
    diagonal = op::assemble_diagonal(problem);
    diagonal_s = t.elapsed_seconds();
}

void Simulation::assemble() {
    Impl& s = *impl_;
    if (s.op == nullptr) {
        // Built into locals and committed together, so that an exception leaves the operator
        // unassembled (and a later call retries).
        std::shared_ptr<op::LinearOperator> Z;
        VectorXc b;
        if (s.config.compression == "mlfmm") {
            const ScopedTimer t("Simulation: MLFMM operator");
            try {
                Z = mlfmm::MlfmmStrategy(s.config.mlfmm).build(s.problem);
            } catch (const mlfmm::TruncationOrderError& e) {
                // Neither a usable expansion nor enough decay for the ADR 0008 §6 fallback: say
                // what to change, per cause, instead of only the far operator's diagnosis.
                std::ostringstream os;
                os << "Simulation: compression \"mlfmm\" cannot represent region R"
                   << e.region() + 1 << " to 10^-" << s.config.mlfmm.accuracy_digits << ": ";
                switch (e.cause()) {
                    case mlfmm::TruncationOrderError::Cause::mesh_or_leaf_size:
                        os << "the mesh is too coarse for the octree leaves; refine the mesh or "
                              "raise mlfmm.min_box_size_lambda";
                        break;
                    case mlfmm::TruncationOrderError::Cause::lossy_region:
                        os << "the region is too lossy for the expansion at this box size but its "
                              "decay is too weak for truncation; change mlfmm.min_box_size_lambda, "
                              "lower mlfmm.accuracy_digits or use compression \"dense\"";
                        break;
                    case mlfmm::TruncationOrderError::Cause::digits:
                        os << "the requested digits are not reachable at this leaf size; lower "
                              "mlfmm.accuracy_digits or raise mlfmm.min_box_size_lambda";
                        break;
                    case mlfmm::TruncationOrderError::Cause::exact_part_too_large:
                        os << "the exactly evaluated far interactions of the lossy region would "
                              "exceed the memory budget mlfmm.max_exact_far_bytes; lower "
                              "mlfmm.accuracy_digits, raise mlfmm.max_exact_far_bytes or use "
                              "compression \"dense\"";
                        break;
                }
                os << ". Details: " << e.what();
                throw std::runtime_error(os.str());
            }
            s.assembly_s = t.elapsed_seconds();
        } else {
            const ScopedTimer t("Simulation: dense assembly of Z");
            Z = op::DenseStrategy().build(s.problem);
            s.assembly_s = t.elapsed_seconds();
        }
        {
            const ScopedTimer t("Simulation: right-hand side");
            b = op::assemble_rhs(s.problem);
            s.rhs_s = t.elapsed_seconds();
        }
        s.rhs = std::move(b);
        s.op = std::move(Z);
    }
    s.ensure_diagonal();
}

solver::GmresResult Simulation::solve(const solver::IterationCallback& cb) {
    return solve(impl_->config.gmres, cb);
}

solver::GmresResult Simulation::solve(const solver::GmresParams& params,
                                      const solver::IterationCallback& cb) {
    check_gmres_params(params);
    assemble();  // whatever is missing: Z and b, and the diagonal if this solve uses it
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
        // GmresResult size rule: residual_history has iterations + 1 entries, the last one
        // the final residual (here the LU residual; 0 for b = 0).
        res.iterations = 0;
        res.converged = true;
        res.true_relative_residual = s.direct_info.residual;
        res.residual_history = {s.direct_info.residual};
    } else if (s.jacobi) {
        const solver::DiagonalPreconditioner M(s.diagonal);
        res = solver::gmres(*s.op, s.rhs, M, params, cb);
    } else {
        res = solver::gmres(*s.op, s.rhs, solver::IdentityPreconditioner(), params, cb);
    }
    s.solve_s = res.wall_seconds;
    s.solution.problem = &s.problem;
    s.solution.currents = res.x;
    s.result = res;
    s.result.x = VectorXc();  // the currents are kept once, in s.solution
    s.last_gmres = params;
    s.solved = true;
    if (!res.converged) {
        SBEM_WARN(
            "Simulation: GMRES did not converge in {} iterations (monitored residual {:.3e}, "
            "tolerance {:.3e}, true residual {:.3e}); the solution is stored anyway",
            res.iterations, res.residual_history.back(), params.tolerance,
            res.true_relative_residual);
    }
    SBEM_INFO(
        "Simulation: {} {}{}, 2N = {}: {} it, converged {}, true residual {:.3e}, "
        "assembly {:.3f} s, solve {:.3f} s",
        s.form->name(), s.config.solver == SolverKind::Direct ? "direct LU" : "GMRES",
        s.config.solver == SolverKind::Gmres && s.jacobi ? " + Jacobi" : "", num_unknowns(),
        res.iterations, res.converged, res.true_relative_residual,
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
    const solver::GmresParams& g = s.solved ? s.last_gmres : s.config.gmres;
    std::ostringstream o;
    o << "SpeckleBem simulation\n";
    o << "  mesh:           " << s.mesh.num_triangles() << " triangles, N = " << s.space.size()
      << " RWG functions, 2N = " << num_unknowns() << " unknowns\n";
    o << "  wavelength:     " << s.config.wavelength
      << " m, object eps_r = " << s.config.object.eps_r
      << ", exterior eps_r = " << s.config.exterior.eps_r << "\n";
    o << "  formulation:    " << s.form->name() << (s.kind_auto ? " (automatic)" : " (explicit)")
      << "\n";
    o << "  preconditioner: " << (s.jacobi ? "diagonal (Jacobi)" : "none")
      << (s.jacobi_auto ? " (automatic)" : " (explicit)");
    if (direct)
        o << ", unused by the direct solver";
    else if (s.jacobi)
        o << ", " << side_name(g.side) << " side";
    o << "\n";
    o << "  solver:         ";
    if (direct)
        o << "direct LU\n";
    else
        o << "GMRES (tol " << g.tolerance << ", max_iter " << g.max_iter << ", restart "
          << g.restart << ")\n";
    o << "  compression:    " << s.config.compression;
    if (s.config.compression == "mlfmm") {
        const mlfmm::MlfmmParams& m = s.config.mlfmm;
        o << " (accuracy_digits " << m.accuracy_digits << ", max_elements_per_leaf "
          << m.octree.max_elements_per_leaf << ", min_box_size_lambda "
          << m.octree.min_box_size_lambda << ", max_levels " << m.octree.max_levels
          << ", automatic_leaf_size " << (m.automatic_leaf_size ? "true" : "false")
          << ", max_exact_far_bytes " << m.max_exact_far_bytes << ")";
    }
    o << "\n";
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
