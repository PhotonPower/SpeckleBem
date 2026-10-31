#pragma once
/// @file simulation.hpp
/// High-level driver that wires everything together (top layer, docs/02). This is also the
/// surface exposed to Python (docs/10): build a Simulation from a mesh, an excitation and a
/// config, assemble, solve, post-process with post::* on solution().
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/geometry/mesh.hpp"
#include "specklebem/geometry/open_surface.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/postprocessing/fields.hpp"
#include "specklebem/postprocessing/huygens.hpp"
#include "specklebem/postprocessing/scattering.hpp"
#include "specklebem/solver/gmres.hpp"
#include "specklebem/solver/incomplete_lu_preconditioner.hpp"
#include "specklebem/solver/schur_preconditioner.hpp"
#include "specklebem/solver/sparse_lu_preconditioner.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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

/// Preconditioner of the GMRES solve (SimulationConfig::preconditioner; WP-PC1, ADR 0011).
enum class PreconditionerKind {
    None,           ///< solver::IdentityPreconditioner
    Diagonal,       ///< Jacobi, solver::DiagonalPreconditioner (the diagonal of Z)
    BlockDiagonal,  ///< near-field blocks of the octree boxes, make_block_preconditioner()
    /// Near-field Schur complement over the J and M blocks of the octree boxes,
    /// make_schur_preconditioner() (WP-SC2, ADR 0012 decision 2): explicit option only, never
    /// automatic; needs compression "mlfmm" (the near-field matrix of the operator).
    Schur,
    /// Incomplete LU (ILUT) of the near-field blocks A = Z_n[J, J] and D = Z_n[M, M],
    /// make_incomplete_lu_preconditioner() with SimulationConfig::ilut (WP-SC3, ADR 0012
    /// decision 6): explicit option only, never automatic; needs compression "mlfmm".
    IncompleteLu
};

/// "none", "diagonal", "block_diagonal", "schur" or "ilut".
[[nodiscard]] const char* to_string(PreconditionerKind kind);

struct SimulationConfig {
    /// Vacuum wavelength [m]; must equal the excitation's wavelength (relative 1e-12).
    Real wavelength = 500e-9;
    /// Region R1; must equal the excitation's background (the incident field lives in R1).
    material::Material exterior = material::vacuum();
    material::Material object = material::silicon_500nm();  ///< region R2
    /// Empty: automatic, formulation::recommend(object.eps_r).kind.
    std::optional<formulation::Kind> formulation;
    /// Left/right Jacobi preconditioner (the diagonal of Z: op::assemble_diagonal, or the near
    /// parts of the MLFMM operator for "mlfmm") of the GMRES solve; empty: the
    /// recommendation's choice, formulation::recommend(object.eps_r).diagonal_preconditioner
    /// (also when the formulation is set explicitly). Ignored by the direct solver. Kept for
    /// compatibility; `preconditioner` below is the general option.
    std::optional<bool> diagonal_preconditioner;
    /// Preconditioner of the GMRES solve (WP-PC1, ADR 0011): None, Diagonal (Jacobi) or
    /// BlockDiagonal (one LU-factorised block per octree box over the J and M unknowns of its
    /// basis functions, from the near-field entries; make_block_preconditioner()). Empty: from
    /// diagonal_preconditioner (true -> Diagonal, false -> None), or, if that is empty too, the
    /// automatic choice: BlockDiagonal on the PML route with compression "mlfmm" (the GMRES
    /// solver; WP-PC1, ADR 0011 decision 3: 3.2x fewer iterations than Jacobi on rough Ag up to
    /// 3e5 unknowns and 2.5x fewer than none on rough Si; benchmarks/results/preconditioner.md),
    /// otherwise the recommendation (Diagonal for metals, None for dielectrics, as before WP-PC1),
    /// also on the dense PML route (not measured) and the box route (on the Ag sphere the blocks
    /// need more iterations than Jacobi). The automatic BlockDiagonal logs an SBEM_WARN at
    /// assembly when its largest block holds more than half of the unknowns (a tiny or coarse
    /// tree whose block level is the root: a dense LU); an explicit BlockDiagonal does not. Both
    /// set: they must agree (Diagonal with true, None with false; BlockDiagonal with an empty
    /// diagonal_preconditioner only), otherwise the constructors throw std::invalid_argument.
    /// Ignored by the direct solver. An explicit BlockDiagonal is allowed on every route.
    /// Schur (WP-SC2, ADR 0012 decision 2; make_schur_preconditioner()) is an explicit option
    /// only (never automatic), with an empty diagonal_preconditioner, and needs compression
    /// "mlfmm" (otherwise the constructors throw std::invalid_argument); its J and M blocks are
    /// the boxes of block_levels_above_leaf. IncompleteLu (WP-SC3, ADR 0012 decision 6;
    /// make_incomplete_lu_preconditioner(), parameters `ilut` below) has the same rules as Schur
    /// (explicit only, empty diagonal_preconditioner, compression "mlfmm"); it does not use
    /// block_levels_above_leaf.
    std::optional<PreconditionerKind> preconditioner;
    /// IncompleteLu only: the ILUT parameters (solver::IlutParams: drop tolerance relative to
    /// the row norm, default 1e-4; fill factor, nnz(L + U) <= fill_factor x nnz of each block
    /// row, default 3; pivot guard min_pivot_rel, default 1e-10; factor memory budget, default
    /// half of the available memory; ordering, default AMD). Validated by the constructors when
    /// IncompleteLu is set (solver::validate_ilut_params), ignored otherwise.
    solver::IlutParams ilut;
    /// BlockDiagonal and Schur only: the block level, 0 = the octree leaves, 1 = one level above
    /// them (the two levels of ADR 0011), 2 = two levels above them (WP-SC1, ADR 0012: a
    /// measurement point between one level up and the exact near-field inverse; with compression
    /// "mlfmm" these blocks are those of the near-field matrix, same-box far pairs are zero, see
    /// make_block_preconditioner); other values are rejected. Default 1, the faster of the two
    /// ADR 0011 levels at every measured PML size (it also needs the fewer iterations). On the
    /// dense route the octree is mlfmm::make_octree with the `mlfmm` parameters below (the tree
    /// "mlfmm" would build, so they are validated by the constructors whenever BlockDiagonal is
    /// set); on a tree of two levels (e.g. the tiny PML meshes of the tests) one level up is the
    /// root, i.e. one block = the dense LU of Z and GMRES converges in one iteration: use 0 to
    /// exercise the block path.
    int block_levels_above_leaf = 1;
    /// "dense" (default) or "mlfmm" (mlfmm::MlfmmOperator with the `mlfmm` parameters; GMRES
    /// only, the Jacobi diagonal is read from the operator's near parts,
    /// mlfmm::MlfmmOperator::diagonal(), equal to op::assemble_diagonal; WP-PML5c). "aca" and
    /// "hmatrix" (Phase 8) are rejected with std::invalid_argument until they are implemented, as
    /// is any other name.
    std::string compression = "dense";
    /// MLFMM parameters (compression "mlfmm" only): octree (leaf floor min_box_size_lambda in
    /// wavelengths of the exterior), accuracy_digits d0.
    mlfmm::MlfmmParams mlfmm;
    solver::GmresParams gmres;
    kernels::OperatorOptions kernels;
    SolverKind solver = SolverKind::Gmres;
};

/// Near-field block-diagonal preconditioner of an assembled system operator Z (WP-PC1, ADR
/// 0011): one block per octree box on level max(0, leaf_level - levels_above_leaf), over the J
/// and M unknowns of the box's basis functions (mlfmm::level_unknown_blocks; box index as the
/// block id), entries taken exactly once from
///  * mlfmm::MlfmmOperator: its octree, near_operator() plus, on the PML route, the near ring
///    (PmlRingOperator::near_operator(); the two partition the near pattern). Levels 0 and 1:
///    every same-box pair is near, so the blocks equal those of the dense matrix (checked:
///    solver::MissingBlockEntries::Throw). Level 2 (WP-SC1, ADR 0012): same-box pairs can be far
///    pairs, which the operator does not store; the blocks are those of the near-field matrix
///    Z_n (far pairs zero, solver::MissingBlockEntries::Zero; their number is
///    BlockDiagonalPreconditioner::missing_entries()), i.e. a coarser block Jacobi of Z_n between
///    the blocks one level up and Z_n^{-1} (make_near_field_lu_preconditioner);
///  * op::DenseOperator: the octree mlfmm::make_octree(*problem.space, lambda_1, mlfmm)
///    (lambda_1 = 2 pi / Re k_1, the reference of the MLFMM operator), blocks of the matrix (on
///    every level the exact blocks of Z, far pairs included).
/// Simulation uses it for PreconditionerKind::BlockDiagonal; drivers can use it to compare
/// preconditioners on one assembled operator.
/// @throws std::invalid_argument for levels_above_leaf not 0, 1 or 2, another operator type, an
///         operator whose size is not 2N of problem.space, and as solver::extract_diagonal_blocks
///         / mlfmm::make_octree; std::runtime_error for a singular or ill-conditioned block
///         (solver::BlockDiagonalPreconditioner, the message names the box).
[[nodiscard]] std::unique_ptr<solver::BlockDiagonalPreconditioner> make_block_preconditioner(
    const op::LinearOperator& Z, const op::Problem& problem, const mlfmm::MlfmmParams& mlfmm,
    int levels_above_leaf);

/// Near-field Schur-complement preconditioner of an MLFMM operator Z (WP-SC2, ADR 0012 decision
/// 2): solver::SchurComplementPreconditioner of the near-field matrix Z_n = near_operator() plus,
/// on the PML route, the near ring (borrowed, not copied: Z must outlive the preconditioner),
/// with the J and M unknowns of the octree boxes on level max(0, leaf_level - levels_above_leaf)
/// as its J and M blocks (mlfmm::level_unknown_blocks split at N, box index as the block id;
/// the same boxes as make_block_preconditioner). An explicit option, never a default. Logs the
/// describe() line (SBEM_INFO).
/// @throws std::invalid_argument for levels_above_leaf not 0, 1 or 2, an operator that is not an
///         mlfmm::MlfmmOperator (the dense route has no near-field matrix), an operator whose
///         size is not 2N of problem.space, and as solver::SchurComplementPreconditioner;
///         std::runtime_error for a singular or ill-conditioned J block of A or M block of the
///         Schur complement (the message names the box).
[[nodiscard]] std::unique_ptr<solver::SchurComplementPreconditioner> make_schur_preconditioner(
    const op::LinearOperator& Z, const op::Problem& problem, int levels_above_leaf,
    const solver::SchurParams& params = {});

/// Incomplete LU of the near-field blocks of an MLFMM operator Z (WP-SC3, ADR 0012 decision 6):
/// solver::IncompleteLuPreconditioner of A = Z_n[J, J] and D = Z_n[M, M], Z_n = near_operator()
/// plus, on the PML route, the near ring (the entries of A and D are read from the two sparse
/// parts directly, without forming Z_n; nothing is borrowed after the constructor). An explicit
/// option, never a default. Logs the describe() line (SBEM_INFO).
/// @throws std::invalid_argument for an operator that is not an mlfmm::MlfmmOperator (the dense
///         route has no near-field matrix), an operator whose size is not 2N of problem.space,
///         and as solver::IncompleteLuPreconditioner (invalid params); std::runtime_error for an
///         ILUT breakdown (zero row, zero or tiny pivot) or the factor memory above the budget.
[[nodiscard]] std::unique_ptr<solver::IncompleteLuPreconditioner> make_incomplete_lu_preconditioner(
    const op::LinearOperator& Z, const op::Problem& problem, const solver::IlutParams& params = {});

/// The near-field matrix Z_n of an MLFMM operator (WP-SC1, ADR 0012): near_operator() plus, on
/// the PML route, the near ring (PmlRingOperator::near_operator()), summed; the two hold every
/// near entry exactly once (ADR 0011), so the sum has nnz(Z_rr near) + nnz(ring near) entries.
/// Without a stretch a copy of near_operator(). 2N x 2N, RowMajor, compressed.
/// @throws std::logic_error if the two patterns overlap (an entry stored twice).
[[nodiscard]] op::SparseOperator::Matrix near_field_matrix(const mlfmm::MlfmmOperator& op);

/// Exact inverse of the near-field matrix as a preconditioner, M^{-1} = Z_n^{-1} (WP-SC1, ADR
/// 0012 decision 1): solver::SparseLuPreconditioner of near_field_matrix(op), the ceiling of
/// every near-field-based preconditioner (block-diagonal, Schur complement, SPAI / ILU of Z_n).
/// A measurement tool, never a default. Logs (SBEM_INFO) nnz(Z_n), the memory estimate and the
/// budget before the factorisation, and nnz(L + U), the fill factor nnz(L + U) / nnz(Z_n), the
/// factor memory and the setup time after it. The summed Z_n is released before returning.
/// @param budget_bytes, expected_fill as solver::SparseLuPreconditioner (empty: half of the
///        available physical memory; expected_fill: nnz(L + U) / nnz(Z_n) measured on a smaller
///        case of the same family, 1 = no fill).
/// Memory: the budget bounds the used factor entries (nnz(L + U) x 24 B + permutations; the
/// estimate counts the same), not the process peak. Eigen reserves about ceil(expected_fill) x
/// nnz(Z_n) x 42 B up front, grows its factor arrays by 1.5x and never shrinks them (on Windows
/// the reservation counts against the commit limit), and Z_n, its copies and the work arrays
/// coexist with the factors during the factorisation: measured on the 16 um rough-Ag near field
/// (factors 66.3 GB) a peak of about 1.3x the factors plus the input copies (84.7 GB working set
/// under a 40 GB budget, 86.4 GB with 75 GB). The factorisation can exhaust the machine before
/// the post-check fires; see solver::SparseLuPreconditioner.
/// @throws as near_field_matrix and solver::SparseLuPreconditioner (std::runtime_error when the
///         estimate or the factor memory exceeds the budget, or for a singular Z_n).
[[nodiscard]] std::unique_ptr<solver::SparseLuPreconditioner> make_near_field_lu_preconditioner(
    const mlfmm::MlfmmOperator& op, std::optional<std::size_t> budget_bytes = std::nullopt,
    Real expected_fill = 1.0);

/// Far-field source of the PML route (ADR 0010, WP-PML6).
enum class FarFieldSource {
    /// Radiation of the surface currents of the real part with the raised-cosine window across
    /// the gap (ADR 0009 decision 5; post::far_field with FieldOptions::window).
    Currents,
    /// Radiation of the windowed equivalent currents of a Huygens plane in R1 at the distance
    /// z_h below the surface (post::huygens_far_field; FieldOptions::huygens), which filters
    /// the bound surface-plasmon field by distance instead of truncating it.
    Huygens
};

/// PML termination of an open rough surface (ADR 0009, Simulation's second constructor). The
/// stretch is geometry::make_pml_stretch(open, sigma_max, order) (start = pml_start, width =
/// outer_half - pml_start, the same profile on x and y).
struct PmlTermination {
    /// sigma_max > 0 of the profile sigma(t) = sigma_max ((t - x_p) / W_pml)^order
    /// (dimensionless; ADR 0009 lead decision 2: 3-5).
    Real sigma_max = 4.0;
    /// Polynomial order m >= 0 (lead decision 2: 2). m < 1 needs a gap W_g >= the longest mesh
    /// edge (WP-PML2 review W5: non-coplanar touching pairs at the PML start have no ridge split).
    Real order = 2.0;
    /// Correlation length Lc [m] of the patch's height map: the taper must be at least Lc wide
    /// (W_t = taper_end - patch_half >= Lc, ADR 0009 decision 1). Required (> 0) for a rough
    /// patch; a flat patch (every patch vertex at z = 0) may leave it at 0.
    Real correlation_length = 0.0;
    /// Dark band (ADR 0009 decision 4): the largest |E_inc| on the PML-start square (z = 0,
    /// max(|x|, |y|) = pml_start) relative to the largest |E_inc| on the patch vertices must not
    /// exceed this (> 0; the PML target 1e-4 by default).
    Real dark_band_tolerance = 1e-4;
    /// Source of every far-field observable of the Simulation (far_field(directions,
    /// quad_degree), drc, drc_polarized, radiated_power, reflected_power; WP-PML6, ADR 0010).
    FarFieldSource far_field_source = FarFieldSource::Currents;
    /// Huygens plane of FarFieldSource::Huygens (post::HuygensOptions: z_h default lambda_1,
    /// spacing default lambda_1 / 4, window default the current window (taper_end, pml_start)).
    /// Checked by the constructor whatever the source: z_h finite and > 0, spacing finite, > 0
    /// and <= lambda_1 / 2, a window valid and ending at or before pml_start.
    post::HuygensOptions huygens;
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
    ///         parameters (tolerance not finite or <= 0, max_iter < 1, restart < 0),
    ///         preconditioner and diagonal_preconditioner set inconsistently, an invalid
    ///         PreconditionerKind, PreconditionerKind::Schur or IncompleteLu without compression
    ///         "mlfmm", invalid ILUT parameters (SimulationConfig::ilut) with IncompleteLu,
    ///         block_levels_above_leaf not 0, 1 or 2, invalid MLFMM
    ///         parameters (SimulationConfig::mlfmm) with compression "mlfmm" or with an explicit
    ///         BlockDiagonal on the dense route (its octree), and everything op::validate
    ///         rejects.
    Simulation(geometry::TriangleMesh mesh, std::shared_ptr<excitation::Excitation> excitation,
               SimulationConfig config);
    /// Open rough surface terminated by a complex-coordinate-stretching PML (ADR 0009, WP-PML3a):
    /// the PML route, dense (prototype, validated by 7a-7e) or with compression "mlfmm" (fast
    /// path, WP-PML5a / 5b: MLFMM on the real functions plus the PML ring, near exact, far ACA,
    /// mlfmm::MlfmmOperator; GMRES with the near-field block-diagonal preconditioner by default
    /// since WP-PC1 (the "mlfmm" route only, the dense route keeps the recommendation;
    /// SimulationConfig::preconditioner), the masked test rows stay in the system with
    /// right-hand side 0). Cheap (no assembly), like the box constructor.
    ///
    /// Builds the stretch geometry::make_pml_stretch(open, pml.sigma_max, pml.order) and sets
    /// op::Problem::stretch, so that the dense operator uses the stretched kernels, the
    /// right-hand side is masked in the PML (op::assemble_rhs) and far_field_options() carries
    /// the default current window inner_half = open.taper_end, outer_half = open.pml_start
    /// (across the gap; pass it to post::far_field, which requires a window for the stretched
    /// problem). The physical model is a rough spot in a polished surface with absorbing
    /// surroundings (ADR 0009, lead decision 1).
    ///
    /// Logs one SBEM_WARN each (warnings, not errors; rules after validation 7a, ADR 0009
    /// decision 4) for a weakly absorbing object (is_weakly_absorbing(), e.g. Si) at oblique
    /// incidence with a patch mesh size h (largest in-plane extent of a Patch triangle) above
    /// lambda_1 / 6, and for a beam (AngularSpectrumBeam, GaussianBeam) with
    /// w0 cos(theta_in) < lambda_1 or pml_start < 3.5 w0 / cos(theta_in).
    ///
    /// @throws std::invalid_argument (each message names the violated rule) for
    ///  * an inconsistent open mesh: zone.size() != triangles, half widths not finite with
    ///    0 < patch_half < taper_end <= pml_start < outer_half, a vertex beyond outer_half;
    ///  * a wrong orientation: a triangle of the flat zones (Gap, Pml) with n_z >= 0 (normals
    ///    must point out of R2 (z > xi) into R1, n_z < 0, docs/06; no closedness check: the
    ///    open boundary is the outer square, which carries no RWG function);
    ///  * PmlTermination: sigma_max not finite and > 0, order not finite and >= 0,
    ///    correlation_length not finite and >= 0, dark_band_tolerance not finite and > 0;
    ///  * compression other than "dense" or "mlfmm" (as the box constructor; "mlfmm" needs the
    ///    GMRES solver, the direct solver needs "dense"), or "mlfmm" with
    ///    mlfmm.leaf_radius_quantile < 1 (the local leaf rule is not supported by the PML ring);
    ///  * gap width W_g = pml_start - taper_end <= 0 (the far-field window needs it), or, for
    ///    order < 1, W_g < the longest mesh edge;
    ///  * a rough patch (some patch vertex with z != 0) with correlation_length = 0, or a taper
    ///    W_t = taper_end - patch_half < correlation_length (relative slack 1e-9);
    ///  * a vertex of a triangle that carries an unmasked test function (op::pml_masked_functions)
    ///    outside the excitation's controlled ball (masked PML vertices may lie outside it);
    ///  * the dark band: max |E_inc| on the PML-start square (z = 0, sampled at spacing
    ///    <= lambda_1 / 8) over max |E_inc| on the patch vertices > dark_band_tolerance (the
    ///    message reports the ratio and suggests a smaller waist or a wider gap);
    ///  * everything the box constructor checks that applies here (null excitation, wavelength,
    ///    exterior material, unknown compression, dense limit, formulation, kernel options,
    ///    GMRES parameters, preconditioner settings incl. the MLFMM octree parameters of explicit
    ///    BlockDiagonal blocks on the dense route, op::validate).
    Simulation(geometry::OpenSurfaceMesh open, PmlTermination pml,
               std::shared_ptr<excitation::Excitation> excitation, SimulationConfig config);
    ~Simulation();
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;
    Simulation(Simulation&&) = delete;
    Simulation& operator=(Simulation&&) = delete;

    /// Assembles the operator, the right-hand side and (GMRES only) the preconditioner's data:
    /// the Jacobi diagonal, the block-diagonal preconditioner (make_block_preconditioner) or the
    /// Schur-complement preconditioner (make_schur_preconditioner) or the ILUT preconditioner
    /// (make_incomplete_lu_preconditioner), and times them. Idempotent:
    /// what is already assembled is kept, so later calls do nothing (after an exception a later
    /// call retries the missing parts).
    /// @throws std::invalid_argument / std::runtime_error from the operator construction; for
    ///         "mlfmm" a std::runtime_error when no expansion order meets 10^-d0 (lossy
    ///         interior such as Ag: the ADR 0008 §6 policy is WP21), with the remedies;
    ///         std::runtime_error for a singular block of the block-diagonal or the
    ///         Schur-complement preconditioner, an ILUT breakdown (zero or tiny pivot) or its
    ///         factor memory above SimulationConfig::ilut.budget_bytes.
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
    ///         system, non-finite values; Direct: std::invalid_argument for a non-finite
    ///         matrix or right-hand side, see solver::DenseLu).
    solver::GmresResult solve(const solver::GmresParams& params,
                              const solver::IterationCallback& cb = {});
    /// @throws std::logic_error before the first solve().
    [[nodiscard]] const post::SurfaceSolution& solution() const;
    /// True once assemble() has completed (operator and right-hand side, committed together),
    /// i.e. system_operator() and rhs() do not throw. Never throws.
    [[nodiscard]] bool assembled() const;
    /// True once a solve() has completed (also a non-converged GMRES solve, whose solution is
    /// stored), i.e. solution() does not throw. Never throws.
    [[nodiscard]] bool solved() const;
    /// Multi-line report: mesh size, formulation / preconditioner used (and whether automatic;
    /// block-diagonal and Schur complement: the block level, and after assembly the blocks,
    /// setup time and memory),
    /// solver (with the GMRES parameters of the last solve, config().gmres before the first),
    /// compression, operator memory (PML route with "mlfmm": the stretched / real split and the
    /// ring's near / far bytes), timings, iterations and residuals.
    [[nodiscard]] std::string report() const;

    [[nodiscard]] const geometry::TriangleMesh& mesh() const;
    [[nodiscard]] const SimulationConfig& config() const;
    /// The problem (space, materials, formulation, excitation, omega) owned by this object.
    [[nodiscard]] const op::Problem& problem() const;
    /// Formulation actually used (config.formulation or the recommendation).
    [[nodiscard]] formulation::Kind formulation_kind() const;
    /// Whether the Jacobi preconditioner is used by the GMRES solve (resolved setting).
    [[nodiscard]] bool diagonal_preconditioner() const;
    /// Preconditioner of the GMRES solve (resolved setting, SimulationConfig::preconditioner).
    [[nodiscard]] PreconditionerKind preconditioner_kind() const;
    /// The block-diagonal preconditioner once built (assemble() with PreconditionerKind::
    /// BlockDiagonal and the GMRES solver), nullptr otherwise.
    [[nodiscard]] const solver::BlockDiagonalPreconditioner* block_preconditioner() const;
    /// The Schur-complement preconditioner once built (assemble() with PreconditionerKind::Schur
    /// and the GMRES solver), nullptr otherwise.
    [[nodiscard]] const solver::SchurComplementPreconditioner* schur_preconditioner() const;
    /// The ILUT preconditioner once built (assemble() with PreconditionerKind::IncompleteLu and
    /// the GMRES solver), nullptr otherwise.
    [[nodiscard]] const solver::IncompleteLuPreconditioner* incomplete_lu_preconditioner() const;
    /// Number of unknowns 2N.
    [[nodiscard]] Index num_unknowns() const;
    /// System operator and right-hand side. @throws std::logic_error before assemble().
    [[nodiscard]] std::shared_ptr<const op::LinearOperator> system_operator() const;
    [[nodiscard]] const VectorXc& rhs() const;

    /// True for the PML constructor (open surface with Problem::stretch).
    [[nodiscard]] bool has_pml() const;
    /// The PML route's default far-field window (taper_end, pml_start); empty for the box route.
    [[nodiscard]] const std::optional<post::CurrentWindow>& far_field_window() const;
    /// FieldOptions for post::far_field on solution(): quad_degree, the default
    /// min_distance_factor and far_field_window() (none for the box route), and on the PML route
    /// with far_field_source() == FarFieldSource::Huygens also FieldOptions::huygens =
    /// pml_termination().huygens (the plane window defaults to far_field_window()).
    [[nodiscard]] post::FieldOptions far_field_options(int quad_degree = 6) const;
    /// Same for an explicit source. @throws std::logic_error for FarFieldSource::Huygens on the
    /// box route (a closed mesh has no Huygens plane; valid call, wrong state).
    [[nodiscard]] post::FieldOptions far_field_options(int quad_degree,
                                                       FarFieldSource source) const;
    /// Far-field source of the observables: PmlTermination::far_field_source on the PML route,
    /// FarFieldSource::Currents on the box route.
    [[nodiscard]] FarFieldSource far_field_source() const;
    /// Far-field amplitude F [V] per row of `directions` (E_s ~ F exp(-j k_1 r) / r in R1):
    /// post::far_field on solution() with `opt` (e.g. far_field_options() with another window).
    /// On the PML route the rows must lie in the reflection hemisphere, as for drc() below; the
    /// box route accepts every direction.
    /// @throws std::logic_error before solve(); std::invalid_argument for k_z > 0 rows on the
    ///         PML route and everything post::far_field throws for.
    [[nodiscard]] Eigen::Matrix<Complex, Eigen::Dynamic, 3> far_field(
        const Vertices& directions, const post::FieldOptions& opt) const;
    /// Same with far_field_options(quad_degree) (the PML route's window, or its Huygens plane for
    /// FarFieldSource::Huygens, applied automatically).
    [[nodiscard]] Eigen::Matrix<Complex, Eigen::Dynamic, 3> far_field(const Vertices& directions,
                                                                      int quad_degree = 6) const;

    /// Absolute far-field observables (WP25). All of them evaluate post::far_field on
    /// solution() with far_field_options(quad_degree), so the PML route's current window (or
    /// its Huygens plane, far_field_source()) is applied automatically. On the PML route R1 is the
    /// half space below the surface: every direction must point into the reflection hemisphere, k_z
    /// <= 1e-12 |k_hat| (rows with k_z > 0 throw std::invalid_argument, as far_field() does), and
    /// scattered_power() is refused (std::logic_error: a valid call in the wrong state; the same
    /// rule as the Python binding's bistatic_rcs and field on that route).
    ///
    /// Incident power P_inc [W] of the excitation: AngularSpectrumBeam::power() (through any
    /// plane normal to the beam axis, the normalisation of the WP22c / PML reflectances) or
    /// GaussianBeam::power() (paraxial, pi w0^2 / (4 eta_1)). Needs no solve.
    /// @throws std::logic_error for a plane wave (infinite power: normalise by the intensity
    ///         |E_0|^2 / (2 eta_1) instead, i.e. use cross sections) and for any other excitation
    ///         type without a known finite power.
    [[nodiscard]] Real incident_power() const;
    /// Absolute DRC D(k_hat) = |F|^2 / (2 eta_1 P_inc) [1/sr] per row of `directions`
    /// (post::drc with P_inc = incident_power()); its integral over the reflection hemisphere
    /// is the reflectance.
    /// @throws std::logic_error before solve() and from incident_power(); std::invalid_argument
    ///         for k_z > 0 rows on the PML route and everything post::drc throws for.
    [[nodiscard]] VectorXr drc(const Vertices& directions, int quad_degree = 6) const;
    /// Co- and cross-polarised parts of drc() (post::drc_polarized, fixed Cartesian co_pol
    /// vector, e.g. the incident polarisation). Same exceptions as drc() plus a zero or
    /// non-finite co_pol_direction.
    [[nodiscard]] post::PolarizedIntensity drc_polarized(const Vertices& directions,
                                                         const Vec3& co_pol_direction,
                                                         int quad_degree = 6) const;
    /// Power [W] radiated into the solid angle of a quadrature rule (post::radiated_power:
    /// (1 / 2 eta_1) sum w |F|^2, weights in sr). Needs no incident power (works for a plane
    /// wave). @throws std::logic_error before solve(); std::invalid_argument for k_z > 0 rows on
    /// the PML route and everything post::radiated_power throws for.
    [[nodiscard]] Real radiated_power(const Vertices& directions, const VectorXr& weights,
                                      int quad_degree = 6) const;
    /// Power [W] radiated into the reflection hemisphere k_z < 0 (towards the source side of a
    /// beam travelling in +z; for a sphere the backward hemisphere): radiated_power over
    /// post::hemisphere_grid(-1, n_theta, n_phi). reflected_power() / incident_power() is the
    /// reflectance R (1 deg grid: n_theta = 90, n_phi = 180).
    /// @throws std::logic_error before solve(); std::invalid_argument for n_theta < 1 or
    ///         n_phi < 4 and everything post::radiated_power throws for.
    [[nodiscard]] Real reflected_power(int n_theta, int n_phi, int quad_degree = 6) const;
    /// Total scattered power [W] over the full sphere (post::sphere_grid(n_theta, n_phi)); the
    /// scattering cross section of a plane wave is scattered_power() / (|E_0|^2 / (2 eta_1)).
    /// Box route only (closed mesh in R1).
    /// @throws std::logic_error before solve() and on the PML route (the forward hemisphere lies
    ///         in R2 there: no far field); std::invalid_argument as reflected_power().
    [[nodiscard]] Real scattered_power(int n_theta, int n_phi, int quad_degree = 6) const;

    /// PML route only: the termination, the open mesh's zone tags and half widths (mesh moved
    /// into mesh()), the stretch and the measured dark-band ratio.
    /// @throws std::logic_error for the box route.
    [[nodiscard]] const PmlTermination& pml_termination() const;
    [[nodiscard]] const std::vector<geometry::SurfaceZone>& zones() const;
    [[nodiscard]] const geometry::CoordinateStretch& stretch() const;
    /// Half widths a_0 .. a_3 (patch_half, taper_end, pml_start, outer_half) [m].
    [[nodiscard]] std::array<Real, 4> zone_half_widths() const;
    [[nodiscard]] Real dark_band_ratio() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace specklebem
