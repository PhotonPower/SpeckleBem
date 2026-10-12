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

/// Closing box of a truncated rough patch (ADR 0006 amendment 2026-10-12b).
enum class BoxGradingMode {
    /// Box cells = top-face cells (box_mesh_size = mesh_size, M = 0): the default for
    /// quantitative rough-surface runs since the amendment 2026-10-12b (the graded lambda_1 / 5
    /// box failed the depth check for Si at w0 = L / 4 and costs more at MLFMM sizes).
    Uniform,
    /// Graded box of the amendment 2026-10-10: coarse walls and bottom with the largest
    /// 2^M h <= lambda_1 / 5 (cheaper dense studies, convergence checks).
    Graded
};

/// Name of a grading mode: "uniform" or "graded".
[[nodiscard]] const char* to_string(BoxGradingMode mode);

/// Nominal coarse spacing [m] of the graded closing box (BoxGradingMode::Graded, ADR 0006
/// amendment 2026-10-10) for the nominal top-face spacing h = mesh_size: the largest
/// 2^M h <= lambda_1 / 5 (M >= 0, lambda_1 = exterior_wavelength(background, wavelength);
/// computed by geometry::box_spacing_for_exterior_wavelength()). M = 0 (lambda_1 / 5 < 2 h)
/// returns h: the uniform box. Examples: 500 nm in vacuum, h = 50 nm -> 100 nm (M = 1);
/// h = 20 nm -> 80 nm (M = 2); h = 60 nm -> 60 nm (uniform). Renamed from
/// default_box_mesh_size in WP-B3: the default box is the uniform one since the ADR 0006
/// amendment 2026-10-12b, and this value describes only the graded option. It is
/// informational: the generator's grid spacing L / round(L / h) differs from mesh_size for most
/// L (and the box levels use max(dx, dy)), so passing this value as an explicit box_mesh_size
/// can give a coarse spacing slightly above lambda_1 / 5 (e.g. L = 1.07 um, h = 50 nm:
/// 101.9 nm). To build a graded box, set geometry::RoughSurfaceParams::exterior_wavelength and
/// leave box_mesh_size unset (as rough_surface_box_params(..., BoxGradingMode::Graded) does):
/// the generator then applies the same rule with the actual spacing.
/// @throws std::invalid_argument for mesh_size not finite and > 0 and what
///         exterior_wavelength() rejects.
[[nodiscard]] Real graded_box_mesh_size(const material::Material& background, Real wavelength,
                                        Real mesh_size);

/// "Weakly absorbing" object of ADR 0006 (amendments 2026-10-10 and 2026-10-12b):
/// 3 delta > geometry::kNoBandCoarseningDepth * mesh_size = 2 h, delta =
/// material::field_decay_length(object, wavelength), i.e. the transmitted field has not
/// decayed to e^-3 at the first coarsened wall row of a graded box without fine band. Such
/// objects need the fine band and the tighter beam waist (beam_waist_fraction()). At 500 nm
/// and h = 50 nm (threshold delta > 33 nm): Si (delta = 1.13 um) yes, Ag (25 nm) no; Ag at
/// h = 20 nm yes. A lossless dielectric (delta = +infinity) counts as weakly absorbing.
/// @throws std::invalid_argument for mesh_size not finite and > 0, an active object
///         (Im(eps_r) > 0 or Im(mu_r) > 0, docs/06) and what material::field_decay_length
///         rejects.
[[nodiscard]] bool is_weakly_absorbing(const material::Material& object, Real wavelength,
                                       Real mesh_size);

/// Plasmonic metal: Re(eps_r) < 0 (eps_r of the Material, i.e. at the wavelength it was
/// evaluated for; Ag at 500 nm: -9.794). ADR 0006 amendment 2026-10-12b: no quantitative
/// rough-surface results for such objects until WP-V3 (the docs/05 box checks fail for Ag);
/// flat reflectance and A_ext remain validated. rough_surface_box_params() warns for them.
/// @throws std::invalid_argument for a non-finite eps_r.
[[nodiscard]] bool is_plasmonic(const material::Material& object);

/// Largest beam waist relative to the patch edge L at normal incidence (ADR 0006 amendment
/// 2026-10-10: w0 <= L / 4; at L / 3 the rigorous beam carries 0.6-0.9 % of its power past the
/// edges, at L / 4 0.02 %). At oblique incidence the limit is L cos(theta_in) / 4 (amendment
/// 2026-10-11). Strongly absorbing objects; the upper bound of BeamWaistOptions::max_fraction.
inline constexpr Real kMaxBeamWaistFraction = 0.25;

/// Largest beam waist relative to L cos(theta_in) for weakly absorbing objects (ADR 0006
/// amendment 2026-10-12b, provisional: Si at w0 = L / 4 fails the L x 1.2 check with 1.28 %, at
/// L / 6 both docs/05 checks pass).
inline constexpr Real kWeakAbsorberBeamWaistFraction = 1.0 / 6.0;

/// Beam-waist fraction of an object (ADR 0006 amendment 2026-10-12b), to be passed as
/// BeamWaistOptions::max_fraction: kWeakAbsorberBeamWaistFraction = 1/6 for weakly absorbing
/// objects (is_weakly_absorbing(): Si at 500 nm, h = 50 nm), kMaxBeamWaistFraction = 1/4
/// otherwise (Ag; no validated tighter rule, see is_plasmonic()).
/// @throws std::invalid_argument for what is_weakly_absorbing() rejects.
[[nodiscard]] Real beam_waist_fraction(const material::Material& object, Real wavelength,
                                       Real mesh_size);

/// Closing-box parameters of a truncated rough patch chosen from the materials (ADR 0006 with
/// the amendments 2026-10-10 and 2026-10-12b); see rough_surface_box_params().
struct RoughBoxParams {
    Real box_depth = 0.0;  ///< default_box_depth(object, wavelength) [m]
    /// 3 delta + 3 sigma for weakly absorbing objects [m]; unset for the others and whenever
    /// the band would reach the bottom plate (3 delta + 3 sigma >= box_depth). With the uniform
    /// box the band does not change the mesh, so BoxGradingMode::Uniform drops it there without
    /// a fallback or warning: a caller who takes Uniform parameters and then sets a coarse
    /// box_mesh_size must re-check the fine band itself (default_box_fine_depth(); request
    /// BoxGradingMode::Graded for a coarse box instead, which handles this case).
    std::optional<Real> box_fine_depth;
    /// Explicit coarse spacing [m]: mesh_size for the uniform box (the default, and the
    /// fallback of the graded one); unset for the graded box (the generator caps its automatic
    /// rule at lambda_1 / 5 with the actual grid spacing).
    std::optional<Real> box_mesh_size;
    Real exterior_wavelength = 0.0;  ///< lambda_1 = lambda_0 / |n_1| [m]
    /// Graded only: the fine band would reach the bottom plate (3 delta + 3 sigma >=
    /// box_depth), so the uniform box (box_mesh_size = mesh_size, no band) was chosen instead,
    /// logged with SBEM_WARN. Always false for BoxGradingMode::Uniform.
    bool uniform_fallback = false;
    /// The grading requested from rough_surface_box_params().
    BoxGradingMode grading = BoxGradingMode::Uniform;
    /// beam_waist_fraction(object, wavelength, mesh_size): pass it as
    /// BeamWaistOptions::max_fraction to check_beam_waist().
    Real beam_waist_fraction = kMaxBeamWaistFraction;
    /// Sets the four box fields of p (box_depth, box_fine_depth, box_mesh_size,
    /// exterior_wavelength); the other fields are left as they are.
    void apply_to(geometry::RoughSurfaceParams& p) const;
};

/// Closing-box parameters for an object (R2) under a background (R1) at the vacuum wavelength
/// `wavelength`, rms roughness sigma and top-face spacing mesh_size [m]:
///  * box_depth = default_box_depth(object, wavelength) (unchanged by the amendments);
///  * exterior_wavelength = lambda_1 = exterior_wavelength(background, wavelength);
///  * grading BoxGradingMode::Uniform (default, ADR 0006 amendment 2026-10-12b): box_mesh_size
///    = mesh_size, the box cells equal the top-face cells (M = 0);
///  * grading BoxGradingMode::Graded (amendment 2026-10-10, explicit option): box_mesh_size
///    unset, so the generator caps its automatic coarse spacing at the largest
///    2^M h_b <= lambda_1 / 5 with the actual level spacing h_b = max(dx, dy) of the grid
///    (geometry::RoughSurfaceParams::exterior_wavelength); the coarse spacing never exceeds
///    lambda_1 / 5 (up to a relative 1e-9) and no exterior-resolution warning is logged;
///  * box_fine_depth = default_box_fine_depth(object, wavelength, sigma) = 3 delta + 3 sigma
///    for weakly absorbing objects (is_weakly_absorbing(): 3 delta > 2 mesh_size, mirroring
///    the grading contract geometry::kNoBandCoarseningDepth), unset otherwise. At 500 nm and
///    h = 50 nm: Si gets the band 3.54 um (sigma = 50 nm), Ag none. The band is mandatory
///    where it applies (amendment 2026-10-10); with the uniform box it does not change the
///    mesh but is kept so that both gradings carry the same parameters;
///  * band reaching the bottom plate (3 delta + 3 sigma >= box_depth, very rough surfaces):
///    no band. Graded: the uniform box is chosen instead (box_mesh_size = mesh_size,
///    uniform_fallback = true, logged with SBEM_WARN since it costs several times the graded
///    box). Uniform: the box already has the top-face spacing everywhere (no log);
///  * beam_waist_fraction = beam_waist_fraction(object, wavelength, mesh_size) (1/6 for weakly
///    absorbing objects, 1/4 otherwise).
/// Plasmonic objects (is_plasmonic(): Re(eps_r) < 0, e.g. Ag) under a rough surface (sigma >
/// 0) log one SBEM_WARN per call: no quantitative rough-surface results until WP-V3 (ADR 0006
/// amendment 2026-10-12b). sigma = 0 (a flat patch, whose reflectance and A_ext are validated,
/// e.g. the WP22c Fresnel cases) logs nothing. Not an exception: the parameters are returned.
/// The beam waist is not part of the box: rough-surface drivers call check_beam_waist() with
/// max_fraction = beam_waist_fraction.
/// @throws std::invalid_argument for what default_box_depth(), default_box_fine_depth(),
///         exterior_wavelength() reject, for mesh_size not finite and > 0 and for an invalid
///         grading value.
[[nodiscard]] RoughBoxParams rough_surface_box_params(
    const material::Material& object, const material::Material& background, Real wavelength,
    Real sigma, Real mesh_size, BoxGradingMode grading = BoxGradingMode::Uniform);

/// Options of check_beam_waist(). max_fraction is the last member so that a positional
/// initialiser {theta_in, allow_wide} written before it existed keeps its meaning; prefer
/// designated initialisers (in declaration order: .incidence_angle, .allow_wide, .max_fraction).
struct BeamWaistOptions {
    /// Incidence angle theta_in [rad] of the beam axis from +z, |theta_in| < pi / 2.
    Real incidence_angle = 0.0;
    /// Accept a waist above the limit with one SBEM_WARN (deliberate studies of edge effects).
    bool allow_wide = false;
    /// Largest waist relative to L cos(theta_in), in (0, kMaxBeamWaistFraction]: 1/4 (default,
    /// strongly absorbing objects) or beam_waist_fraction() of the object (1/6 for weakly
    /// absorbing objects, ADR 0006 amendment 2026-10-12b). Larger values are rejected: a wider
    /// waist needs allow_wide.
    Real max_fraction = kMaxBeamWaistFraction;
};

/// Beam-waist rule of truncated rough patches (ADR 0006 amendments 2026-10-10, 2026-10-11 and
/// 2026-10-12b): waist <= max_fraction * patch_length * cos(incidence_angle) (relative slack
/// 1e-12), max_fraction = 1/4 by default and 1/6 for weakly absorbing objects
/// (beam_waist_fraction()). The footprint on z = 0 is 1 / cos(theta_in) longer along the plane
/// of incidence, so the limit keeps the edge loss of L / 4 at 0 deg (w0 = L / 4 at 45 deg loses
/// 0.63 %). Simulation cannot determine L from an arbitrary mesh, so every driver that
/// illuminates a rough patch of edge L with a Gaussian beam of waist w0 must call this, e.g.
/// check_beam_waist(L, w0, {.incidence_angle = theta, .max_fraction =
/// box.beam_waist_fraction}). A wider waist throws unless options.allow_wide is true, in which case
/// it logs one SBEM_WARN; the message names the fraction (e.g. "L cos(theta_in) / 6").
/// @throws std::invalid_argument for patch_length or waist not finite and > 0, max_fraction
///         not finite or outside (0, kMaxBeamWaistFraction], an incidence angle not finite or
///         with |theta_in| >= pi / 2, and for a waist above the limit without allow_wide.
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
