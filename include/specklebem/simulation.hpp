#pragma once
/// @file simulation.hpp
/// High-level driver that wires everything together (top layer, docs/02). This is also the
/// surface exposed to Python (docs/10): build a Simulation from a mesh, an excitation and a
/// config, assemble, solve, post-process with post::* on solution().
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/geometry/mesh.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/postprocessing/fields.hpp"
#include "specklebem/solver/gmres.hpp"

#include <memory>
#include <optional>
#include <string>

namespace specklebem {

/// Minimum depth [m] of the closing box of a rough patch (ADR 0006): 2 um.
inline constexpr Real kMinBoxDepth = 2e-6;
/// default_box_depth() logs a warning above this depth (weakly absorbing objects, ADR 0006).
inline constexpr Real kLargeBoxDepthWarning = 50e-6;

/// Default depth [m] of the closing box below the mean plane z = 0 of a truncated rough
/// surface (ADR 0006, coordinator decision 2026-10-09), to be passed as
/// geometry::RoughSurfaceParams::box_depth:
///
///   depth = max(10 / alpha, kMinBoxDepth),   1 / alpha = delta / 2,
///
/// with 1 / alpha the intensity absorption length and delta = material::field_decay_length(
/// object, wavelength) the field (amplitude) decay length lambda_0 / (2 pi |Im n_2|). At
/// 500 nm: Si delta = 1.129 um -> 5.65 um; Ag delta = 25.4 nm -> 127 nm -> 2 um (the minimum).
/// A lossless metal (eps_r real and < 0) has a finite, evanescent delta and is accepted.
/// @throws std::invalid_argument for an object whose field does not decay (lossless
///         dielectric, delta = +infinity: not covered by ADR 0006), and for what
///         material::field_decay_length rejects (non-finite eps_r or mu_r, wavelength not
///         finite and > 0).
[[nodiscard]] Real default_box_depth(const material::Material& object, Real wavelength);

/// Default fine band [m] of the graded closing box (ADR 0006), to be passed as
/// geometry::RoughSurfaceParams::box_fine_depth:
///
///   box_fine_depth = 3 delta + 3 sigma,
///
/// delta = material::field_decay_length(object, wavelength), sigma the rms roughness [m]
/// (the band is measured from the mean plane z = 0, so 3 sigma covers rim points below it).
/// Si at 500 nm with sigma = 50 nm: 3.54 um. The fine band matters for weakly absorbing
/// objects (Si); for strongly absorbing ones (Ag: 0.23 um) the graded box without a fine band
/// already starts coarsening at about 4 delta. The generator requires the result to lie below
/// the box depth (check against default_box_depth() or the depth actually used). Since the
/// ADR 0006 amendment (2026-10-10) the fine band is mandatory for weakly absorbing objects;
/// rough_surface_box_params() decides when it is needed.
/// @throws std::invalid_argument for sigma not finite or < 0, a lossless dielectric object
///         (delta = +infinity) and what material::field_decay_length rejects.
[[nodiscard]] Real default_box_fine_depth(const material::Material& object, Real wavelength,
                                          Real sigma);

/// Wavelength lambda_1 = lambda_0 / |n_1| [m] in the exterior medium R1 (background),
/// n_1 = sqrt(eps_r mu_r). For a lossless background this is the wavelength in R1; for a
/// lossy one it is shorter than lambda_0 / Re(n_1), so the box resolution derived from it
/// (lambda_1 / 5) is conservative.
/// @throws std::invalid_argument for a wavelength not finite and > 0, non-finite eps_r or
///         mu_r, an active background (Im(eps_r) > 0 or Im(mu_r) > 0, docs/06), a metallic
///         background (Re(eps_r) <= 0: no propagating exterior, not covered by ADR 0006) or
///         Re(n_1) <= 0.
[[nodiscard]] Real exterior_wavelength(const material::Material& background, Real wavelength);

/// Nominal coarse spacing [m] of the graded closing box (ADR 0006 amendment 2026-10-10) for
/// the nominal top-face spacing h = mesh_size: the largest 2^M h <= lambda_1 / 5 (M >= 0,
/// lambda_1 = exterior_wavelength(background, wavelength); computed by
/// geometry::box_spacing_for_exterior_wavelength()). M = 0 (lambda_1 / 5 < 2 h) returns h:
/// the uniform box. Examples: 500 nm in vacuum, h = 50 nm -> 100 nm (M = 1); h = 20 nm ->
/// 80 nm (M = 2); h = 60 nm -> 60 nm (uniform). This is informational: the generator's grid
/// spacing L / round(L / h) differs from mesh_size for most L (and the box levels use
/// max(dx, dy)), so passing this value as an explicit box_mesh_size can give a coarse
/// spacing slightly above lambda_1 / 5 (e.g. L = 1.07 um, h = 50 nm: 101.9 nm). To build a
/// box, set geometry::RoughSurfaceParams::exterior_wavelength and leave box_mesh_size unset
/// (as rough_surface_box_params() does): the generator then applies the same rule with the
/// actual spacing. lambda_1 / 5 is necessary under illumination, not shown sufficient
/// (WP-V2).
/// @throws std::invalid_argument for mesh_size not finite and > 0 and what
///         exterior_wavelength() rejects.
[[nodiscard]] Real default_box_mesh_size(const material::Material& background, Real wavelength,
                                         Real mesh_size);

/// Closing-box parameters of a truncated rough patch chosen from the materials (ADR 0006 with
/// the 2026-10-10 amendment); see rough_surface_box_params().
struct RoughBoxParams {
    Real box_depth = 0.0;                ///< default_box_depth(object, wavelength) [m]
    std::optional<Real> box_fine_depth;  ///< 3 delta + 3 sigma for weakly absorbing objects [m]
    /// Explicit coarse spacing [m]: unset (the generator caps its automatic rule at
    /// lambda_1 / 5 with the actual grid spacing), mesh_size for the uniform-box fallback.
    std::optional<Real> box_mesh_size;
    Real exterior_wavelength = 0.0;  ///< lambda_1 = lambda_0 / |n_1| [m]
    /// The fine band would reach the bottom plate (3 delta + 3 sigma >= box_depth): uniform
    /// box (box_mesh_size = mesh_size, no band) chosen instead, logged with SBEM_WARN.
    bool uniform_fallback = false;
    /// Sets the four box fields of p (box_depth, box_fine_depth, box_mesh_size,
    /// exterior_wavelength); the other fields are left as they are.
    void apply_to(geometry::RoughSurfaceParams& p) const;
};

/// Closing-box parameters for an object (R2) under a background (R1) at the vacuum wavelength
/// `wavelength`, rms roughness sigma and top-face spacing mesh_size [m]:
///  * box_depth = default_box_depth(object, wavelength) (unchanged by the amendment);
///  * exterior_wavelength = lambda_1 = exterior_wavelength(background, wavelength);
///  * box_mesh_size unset: the generator caps its automatic coarse spacing at the largest
///    2^M h_b <= lambda_1 / 5 with the actual level spacing h_b = max(dx, dy) of the grid
///    (geometry::RoughSurfaceParams::exterior_wavelength), so the coarse spacing never
///    exceeds lambda_1 / 5 (up to a relative 1e-9) and no exterior-resolution warning is
///    logged;
///  * box_fine_depth = default_box_fine_depth(object, wavelength, sigma) = 3 delta + 3 sigma
///    for weakly absorbing objects, unset otherwise. "Weakly absorbing" mirrors the grading
///    contract (geometry::kNoBandCoarseningDepth): the graded walls without a fine band
///    start to coarsen 2 top-face spacings below a smooth rim, so an object is weakly
///    absorbing if 3 delta > 2 mesh_size, i.e. if its field has not decayed to e^-3 there.
///    At 500 nm and h = 50 nm (threshold delta > 33 nm): Si (delta = 1.13 um) gets the band
///    3.54 um (sigma = 50 nm), Ag (delta = 25 nm, first coarse row at ~4 delta) none. The
///    band is mandatory where it applies (ADR 0006 amendment).
///  * Uniform-box fallback: if the band would reach the bottom plate (3 delta + 3 sigma >=
///    box_depth, very rough surfaces), the uniform box is chosen instead (box_mesh_size =
///    mesh_size, no band, uniform_fallback = true; logged with SBEM_WARN since it costs
///    several times the graded box).
/// The beam waist is not part of the box: rough-surface drivers call check_beam_waist().
/// @throws std::invalid_argument for what default_box_depth(), default_box_fine_depth(),
///         exterior_wavelength() reject and for mesh_size not finite and > 0.
[[nodiscard]] RoughBoxParams rough_surface_box_params(const material::Material& object,
                                                      const material::Material& background,
                                                      Real wavelength, Real sigma, Real mesh_size);

/// Largest beam waist relative to the patch edge L at normal incidence (ADR 0006 amendment
/// 2026-10-10: w0 <= L / 4; at L / 3 the rigorous beam carries 0.6-0.9 % of its power past the
/// edges, at L / 4 0.02 %). At oblique incidence the limit is L cos(theta_in) / 4 (amendment
/// 2026-10-11).
inline constexpr Real kMaxBeamWaistFraction = 0.25;

/// Options of check_beam_waist().
struct BeamWaistOptions {
    /// Incidence angle theta_in [rad] of the beam axis from +z, |theta_in| < pi / 2.
    Real incidence_angle = 0.0;
    /// Accept a waist above the limit with one SBEM_WARN (deliberate studies of edge effects).
    bool allow_wide = false;
};

/// Beam-waist rule of truncated rough patches (ADR 0006 amendments 2026-10-10 and 2026-10-11):
/// waist <= kMaxBeamWaistFraction * patch_length * cos(incidence_angle) (relative slack 1e-12).
/// The footprint on z = 0 is 1 / cos(theta_in) longer along the plane of incidence, so the
/// limit keeps the edge loss of L / 4 at 0 deg (w0 = L / 4 at 45 deg loses 0.63 %).
/// Simulation cannot determine L from an arbitrary mesh, so every driver that illuminates a
/// rough patch of edge L with a Gaussian beam of waist w0 must call this, e.g.
/// check_beam_waist(L, w0, {.incidence_angle = theta}). A wider waist throws unless
/// options.allow_wide is true, in which case it logs one SBEM_WARN.
/// @throws std::invalid_argument for patch_length or waist not finite and > 0, an incidence
///         angle not finite or with |theta_in| >= pi / 2, and for a waist above the limit
///         without allow_wide.
void check_beam_waist(Real patch_length, Real waist, const BeamWaistOptions& options = {});

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
    /// "dense" (default) or "mlfmm" (mlfmm::MlfmmOperator with the `mlfmm` parameters; GMRES
    /// only, the Jacobi diagonal still comes from op::assemble_diagonal, which equals the near
    /// field's diagonal). "aca" and "hmatrix" (Phase 8) are rejected with
    /// std::invalid_argument until they are implemented, as is any other name.
    std::string compression = "dense";
    /// MLFMM parameters (compression "mlfmm" only): octree (leaf floor min_box_size_lambda in
    /// wavelengths of the exterior), accuracy_digits d0.
    mlfmm::MlfmmParams mlfmm;
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
    ///         mu_r) different from config.exterior, a mesh vertex farther than the
    ///         excitation's controlled_radius() from its controlled_center()
    ///         (AngularSpectrumBeam: region_radius around the focus), an unknown or unavailable
    ///         compression,
    ///         2N > op::kMaxDenseUnknowns for the dense strategy, compression "mlfmm" with
    ///         SolverKind::Direct, a formulation that is not
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
    /// @throws std::invalid_argument / std::runtime_error from the operator construction; for
    ///         "mlfmm" a std::runtime_error when no expansion order meets 10^-d0 (lossy
    ///         interior such as Ag: the ADR 0008 §6 policy is WP21), with the remedies.
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
