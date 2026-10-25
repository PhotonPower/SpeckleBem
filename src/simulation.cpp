#include "specklebem/simulation.hpp"

#include "specklebem/basis/rwg.hpp"
#include "specklebem/compression/mlfmm/far_operator.hpp"
#include "specklebem/compression/mlfmm/interpolation.hpp"
#include "specklebem/core/logging.hpp"
#include "specklebem/core/timer.hpp"
#include "specklebem/excitation/angular_spectrum_beam.hpp"
#include "specklebem/kernels/operators.hpp"
#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/solver/direct.hpp"
#include "specklebem/solver/preconditioner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
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
    const Real q = m.leaf_radius_quantile;
    if (!(q > 0.0 && q <= 1.0) || (q < 1.0 && (m.accuracy_digits > mlfmm::kLocalLeafRuleMaxDigits ||
                                               !m.automatic_leaf_size))) {
        throw std::invalid_argument(
            "Simulation: leaf_radius_quantile must lie in (0, 1]; values < 1 (local leaf rule) "
            "need "
            "accuracy_digits <= 3 and automatic_leaf_size (ADR 0008 amendment 2026-10-11)");
    }
    const mlfmm::PmlRingParams& r = m.pml_ring;
    if (!(r.eta > 0.0 && std::isfinite(r.eta)) || !(r.aca_epsilon >= 0.0 && r.aca_epsilon < 1.0) ||
        r.max_rank < 0) {
        throw std::invalid_argument(
            "Simulation: MLFMM pml_ring parameters need eta > 0 (finite), aca_epsilon in [0, 1) "
            "(0: automatic 10^-(d0 + 1)) and max_rank >= 0 (0: break-even)");
    }
    if (!r.assemble_near) {
        throw std::invalid_argument(
            "Simulation: MLFMM pml_ring.assemble_near = false is a diagnostic of a standalone "
            "PmlRingOperator; a solve needs the near ring (assemble_near = true)");
    }
}

/// The incident field is evaluated at surface quadrature points, which lie in the convex hull
/// of the vertices; every vertex must lie in the ball where the excitation is controlled
/// (AngularSpectrumBeam: |r - focus| <= region_radius; infinite for the other sources).
/// use_vertex (optional, one entry per vertex): only the vertices with a non-zero entry must lie
/// in the ball (PML route: the vertices of the triangles the masked right-hand side integrates).
void check_controlled_region(const geometry::TriangleMesh& mesh, const excitation::Excitation& exc,
                             const std::vector<std::uint8_t>* use_vertex = nullptr) {
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
        if (use_vertex != nullptr && (*use_vertex)[static_cast<std::size_t>(i)] == 0)
            continue;
        extent = std::max(extent, (v.row(i).transpose() - center).norm());
    }
    if (extent > radius) {
        // Suggested radius: the extent rounded up to 4 significant digits.
        const Real scale = std::pow(10.0, std::floor(std::log10(extent)) - 3);
        const Real suggested = std::ceil(extent / scale) * scale;
        std::ostringstream msg;
        msg.precision(4);
        msg << (use_vertex != nullptr
                    ? "Simulation: the unmasked part of the mesh (triangles carrying an unmasked "
                      "test function, outside the PML) extends to "
                    : "Simulation: the mesh extends to ")
            << extent << " m from the excitation's centre (" << center.x() << ", " << center.y()
            << ", " << center.z() << ") m, but its fields are controlled only within " << radius
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

const char* to_string(BoxGradingMode mode) {
    switch (mode) {
        case BoxGradingMode::Uniform:
            return "uniform";
        case BoxGradingMode::Graded:
            return "graded";
    }
    throw std::invalid_argument("BoxGradingMode: invalid value");
}

Real graded_box_mesh_size(const material::Material& background, Real wavelength, Real mesh_size) {
    if (!(std::isfinite(mesh_size) && mesh_size > 0.0)) {
        throw std::invalid_argument("graded_box_mesh_size: mesh_size must be finite and > 0");
    }
    return geometry::box_spacing_for_exterior_wavelength(
        exterior_wavelength(background, wavelength), mesh_size);
}

bool is_weakly_absorbing(const material::Material& object, Real wavelength, Real mesh_size) {
    if (!(std::isfinite(mesh_size) && mesh_size > 0.0)) {
        throw std::invalid_argument("is_weakly_absorbing: mesh_size must be finite and > 0");
    }
    if (object.eps_r.imag() > 0.0 || object.mu_r.imag() > 0.0) {
        throw std::invalid_argument(
            "is_weakly_absorbing: Im(eps_r) > 0 or Im(mu_r) > 0 is an active medium in the "
            "exp(+jwt) convention (docs/06)");
    }
    // +infinity for a lossless dielectric: weakly absorbing (the comparison holds).
    const Real delta = material::field_decay_length(object, wavelength);
    return 3.0 * delta > geometry::kNoBandCoarseningDepth * mesh_size;
}

bool is_plasmonic(const material::Material& object) {
    if (!(std::isfinite(object.eps_r.real()) && std::isfinite(object.eps_r.imag()))) {
        throw std::invalid_argument("is_plasmonic: eps_r must be finite");
    }
    return object.eps_r.real() < 0.0;
}

Real beam_waist_fraction(const material::Material& object, Real wavelength, Real mesh_size) {
    return is_weakly_absorbing(object, wavelength, mesh_size) ? kWeakAbsorberBeamWaistFraction
                                                              : kMaxBeamWaistFraction;
}

void RoughBoxParams::apply_to(geometry::RoughSurfaceParams& p) const {
    p.box_depth = box_depth;
    p.box_fine_depth = box_fine_depth;
    p.box_mesh_size = box_mesh_size;
    p.exterior_wavelength = exterior_wavelength;
}

RoughBoxParams rough_surface_box_params(const material::Material& object,
                                        const material::Material& background, Real wavelength,
                                        Real sigma, Real mesh_size, BoxGradingMode grading) {
    if (!(std::isfinite(mesh_size) && mesh_size > 0.0)) {
        throw std::invalid_argument("rough_surface_box_params: mesh_size must be finite and > 0");
    }
    if (grading != BoxGradingMode::Uniform && grading != BoxGradingMode::Graded) {
        throw std::invalid_argument("rough_surface_box_params: invalid grading value");
    }
    RoughBoxParams b;
    b.grading = grading;
    b.box_depth = default_box_depth(object, wavelength);
    const Real fine_depth = default_box_fine_depth(object, wavelength, sigma);
    b.exterior_wavelength = exterior_wavelength(background, wavelength);
    b.beam_waist_fraction = beam_waist_fraction(object, wavelength, mesh_size);
    // Uniform: box cells = top-face cells. Graded: box_mesh_size stays unset, the generator
    // caps its automatic rule at lambda_1 / 5 with the actual grid spacing L / round(L / h),
    // which mesh_size does not know.
    if (grading == BoxGradingMode::Uniform)
        b.box_mesh_size = mesh_size;
    if (sigma > 0.0 && is_plasmonic(object)) {
        SBEM_WARN(
            "rough_surface_box_params: plasmonic object (Re(eps_r) = {:.4g} < 0) under a rough "
            "surface: no quantitative rough-surface results for plasmonic metals until WP-V3 "
            "(ADR 0006 amendment 2026-10-12b: the docs/05 box checks fail for Ag); flat "
            "reflectance and A_ext remain validated",
            object.eps_r.real());
    }
    if (!is_weakly_absorbing(object, wavelength, mesh_size))
        return b;
    if (fine_depth < b.box_depth) {
        b.box_fine_depth = fine_depth;
    } else if (grading == BoxGradingMode::Graded) {
        SBEM_WARN(
            "rough_surface_box_params: the fine band 3 delta + 3 sigma = {:.3g} m reaches the "
            "bottom plate at {:.3g} m; using the uniform box (box_mesh_size = mesh_size), "
            "several times the cost of the graded box",
            fine_depth, b.box_depth);
        b.box_mesh_size = mesh_size;
        b.uniform_fallback = true;
    }
    // Uniform with a band reaching the bottom: no band, the box has the top-face spacing.
    return b;
}

namespace {

/// "/ 4" for a fraction 1 / n with integer n (relative 1e-9), "* f" otherwise.
std::string fraction_text(Real f) {
    std::ostringstream os;
    const Real inv = 1.0 / f;
    const Real n = std::round(inv);
    if (n >= 1.0 && std::abs(inv - n) <= 1e-9 * n)
        os << "/ " << static_cast<long long>(n);
    else
        os << "* " << f;
    return os.str();
}

}  // namespace

void check_beam_waist(Real patch_length, Real waist, const BeamWaistOptions& options) {
    if (!(std::isfinite(patch_length) && patch_length > 0.0) ||
        !(std::isfinite(waist) && waist > 0.0)) {
        throw std::invalid_argument(
            "check_beam_waist: patch_length and waist must be finite and > 0");
    }
    const Real fraction = options.max_fraction;
    if (!(std::isfinite(fraction) && fraction > 0.0 && fraction <= kMaxBeamWaistFraction)) {
        std::ostringstream os;
        os << "check_beam_waist: max_fraction must lie in (0, 1/4], got " << fraction
           << " (a wider waist needs allow_wide)";
        throw std::invalid_argument(os.str());
    }
    const Real theta = options.incidence_angle;
    if (!(std::isfinite(theta) && std::abs(theta) < 0.5 * constants::pi)) {
        std::ostringstream os;
        os << "check_beam_waist: incidence_angle must be finite with |theta_in| < pi / 2, got "
           << theta << " rad";
        throw std::invalid_argument(os.str());
    }
    const Real limit = fraction * patch_length * std::cos(theta);
    if (!(waist > limit * (1.0 + 1e-12)))
        return;
    std::ostringstream os;
    os << "beam waist " << waist << " m exceeds L cos(theta_in) " << fraction_text(fraction)
       << " = " << limit << " m for the patch edge L = " << patch_length
       << " m and theta_in = " << theta
       << " rad (ADR 0006 amendments 2026-10-10/11/12b: w0 <= L cos(theta_in) / 4, / 6 for "
          "weakly absorbing objects such as Si; past the patch edges the beam carries 0.6-0.9 % "
          "of its power at L / 3 and 0 deg, 0.63 % at L / 4 and 45 deg; Si at L / 4 fails the "
          "L x 1.2 check with 1.28 %)";
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

    /// PML route (second constructor) only.
    struct Pml {
        PmlTermination termination;
        std::vector<geometry::SurfaceZone> zone;
        std::array<Real, 4> half{};  ///< patch_half, taper_end, pml_start, outer_half
        geometry::CoordinateStretch stretch;
        Real dark_band_ratio = 0;
    };
    std::optional<Pml> pml;
    std::optional<post::CurrentWindow> window;

    /// Formulation, preconditioner and Problem (both constructors; the stretch is set by the
    /// PML constructor afterwards).
    void setup(const formulation::Recommendation& rec, formulation::Kind k,
               std::unique_ptr<formulation::Formulation> f);
};

namespace {

/// Excitation checks of both constructors: non-null, wavelength, background.
void check_excitation(const std::shared_ptr<excitation::Excitation>& excitation,
                      const SimulationConfig& config) {
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
}

/// Operator / solver checks of both constructors (compression already checked by name).
void check_operator_config(const SimulationConfig& config, Index unknowns) {
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
}

std::unique_ptr<formulation::Formulation> checked_formulation(formulation::Kind kind) {
    try {
        return formulation::make_formulation(kind);
    } catch (const std::logic_error& e) {  // JMCFIE (reserved) or an unknown Kind
        throw std::invalid_argument(std::string("Simulation: formulation not available: ") +
                                    e.what());
    }
}

}  // namespace

void Simulation::Impl::setup(const formulation::Recommendation& rec, formulation::Kind k,
                             std::unique_ptr<formulation::Formulation> f) {
    kind_auto = !config.formulation.has_value();
    kind = k;
    jacobi_auto = !config.diagonal_preconditioner.has_value();
    jacobi = config.diagonal_preconditioner.value_or(rec.diagonal_preconditioner);
    form = std::move(f);

    problem.space = &space;
    problem.exterior = config.exterior;
    problem.object = config.object;
    problem.formulation = form.get();
    problem.excitation = excitation.get();
    problem.kernel_options = config.kernels;
    // The excitation's omega (its wavelength equals config.wavelength, checked above), so the
    // Problem and the incident field use the same frequency.
    problem.omega = excitation->omega();
}

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
    check_excitation(excitation, config);
    check_controlled_region(mesh, *excitation);
    check_compression(config.compression);
    // Closed mesh: every edge is interior and carries one RWG function (N = num_edges()).
    check_operator_config(config, 2 * mesh.num_edges());

    const formulation::Recommendation rec = formulation::recommend(config.object.eps_r);
    const formulation::Kind kind = config.formulation.value_or(rec.kind);
    std::unique_ptr<formulation::Formulation> form = checked_formulation(kind);

    impl_ = std::make_unique<Impl>(std::move(mesh), std::move(excitation), std::move(config));
    impl_->setup(rec, kind, std::move(form));
    op::validate(impl_->problem);
}

namespace {

/// Consistency of an OpenSurfaceMesh (as built by geometry::make_open_surface_mesh).
void check_open_mesh(const geometry::OpenSurfaceMesh& open) {
    const geometry::TriangleMesh& mesh = open.mesh;
    if (static_cast<Index>(open.zone.size()) != mesh.num_triangles()) {
        throw std::invalid_argument("Simulation (PML): OpenSurfaceMesh::zone has " +
                                    std::to_string(open.zone.size()) + " entries for " +
                                    std::to_string(mesh.num_triangles()) + " triangles");
    }
    const Real a0 = open.patch_half, a1 = open.taper_end, a2 = open.pml_start, a3 = open.outer_half;
    if (!(std::isfinite(a0) && std::isfinite(a3) && a0 > 0.0 && a0 < a1 && a1 <= a2 && a2 < a3)) {
        throw std::invalid_argument(
            "Simulation (PML): OpenSurfaceMesh half widths must be finite with 0 < patch_half < "
            "taper_end <= pml_start < outer_half");
    }
    const Vertices& V = mesh.vertices();
    for (Index i = 0; i < V.rows(); ++i) {
        if (std::max(std::abs(V(i, 0)), std::abs(V(i, 1))) > a3 * (1.0 + 1e-9)) {
            throw std::invalid_argument(
                "Simulation (PML): a mesh vertex lies beyond OpenSurfaceMesh::outer_half");
        }
    }
    // Orientation (docs/06): the object occupies z > xi, light comes from z < 0, so the normal
    // points out of R2 into R1, n_z < 0; the flat zones show it unambiguously.
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        const geometry::SurfaceZone z = open.zone[static_cast<std::size_t>(t)];
        if ((z == geometry::SurfaceZone::Gap || z == geometry::SurfaceZone::Pml) &&
            !(mesh.normal(t).z() < 0.0)) {
            throw std::invalid_argument(
                "Simulation (PML): triangle " + std::to_string(t) +
                " of the flat zones has n_z >= 0; the normals must point out of the object R2 "
                "(z > xi) into R1 (n_z < 0, docs/06; make_open_surface_mesh emits this "
                "orientation)");
        }
    }
}

void check_pml_termination(const PmlTermination& p) {
    if (!(std::isfinite(p.sigma_max) && p.sigma_max > 0.0)) {
        throw std::invalid_argument(
            "Simulation (PML): PmlTermination::sigma_max must be finite and > 0 (0 is no PML)");
    }
    if (!(std::isfinite(p.order) && p.order >= 0.0)) {
        throw std::invalid_argument(
            "Simulation (PML): PmlTermination::order must be finite and >= 0");
    }
    if (!(std::isfinite(p.correlation_length) && p.correlation_length >= 0.0)) {
        throw std::invalid_argument(
            "Simulation (PML): PmlTermination::correlation_length must be finite and >= 0");
    }
    if (!(std::isfinite(p.dark_band_tolerance) && p.dark_band_tolerance > 0.0)) {
        throw std::invalid_argument(
            "Simulation (PML): PmlTermination::dark_band_tolerance must be finite and > 0");
    }
    if (p.far_field_source != FarFieldSource::Currents &&
        p.far_field_source != FarFieldSource::Huygens) {
        throw std::invalid_argument("Simulation (PML): unknown PmlTermination::far_field_source");
    }
}

/// HuygensOptions of the PML route (ADR 0010): z_h > 0, 0 < spacing <= lambda_1 / 2, a valid
/// window ending at or before pml_start (plane points over the PML are not real observation
/// points).
void check_huygens_options(const post::HuygensOptions& h, Real lambda1, Real pml_start) {
    if (h.z_h && !(std::isfinite(*h.z_h) && *h.z_h > 0.0)) {
        throw std::invalid_argument(
            "Simulation (PML): PmlTermination::huygens.z_h must be finite and > 0");
    }
    if (h.spacing &&
        !(std::isfinite(*h.spacing) && *h.spacing > 0.0 && *h.spacing <= 0.5 * lambda1)) {
        std::ostringstream os;
        os << "Simulation (PML): PmlTermination::huygens.spacing must be finite, > 0 and <= "
              "lambda_1 / 2 = "
           << 0.5 * lambda1 << " m";
        throw std::invalid_argument(os.str());
    }
    if (h.window) {
        (void)post::current_window_value(*h.window, 0.0);  // validates the window
        if (!(h.window->outer_half <= pml_start)) {
            std::ostringstream os;
            os << "Simulation (PML): PmlTermination::huygens.window ends at "
               << h.window->outer_half << " m, beyond pml_start = " << pml_start
               << " m (plane points over the PML are not real observation points)";
            throw std::invalid_argument(os.str());
        }
    }
}

const char* source_name(FarFieldSource s) {
    return s == FarFieldSource::Huygens ? "Huygens plane" : "windowed currents";
}

Real longest_edge(const geometry::TriangleMesh& mesh) {
    const Vertices& V = mesh.vertices();
    const Triangles& T = mesh.triangles();
    Real h = 0;
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        for (Index c = 0; c < 3; ++c) {
            const Vec3 a = V.row(T(t, c)).transpose();
            const Vec3 b = V.row(T(t, (c + 1) % 3)).transpose();
            h = std::max(h, (a - b).norm());
        }
    }
    return h;
}

/// Dark band of ADR 0009 decision 4: max |E_inc| on the PML-start square (z = 0) over max
/// |E_inc| on the patch vertices.
Real measure_dark_band(const geometry::TriangleMesh& mesh, Real patch_half, Real pml_start,
                       const excitation::Excitation& exc, Real lambda1) {
    Real patch_max = 0;
    const Vertices& V = mesh.vertices();
    for (Index i = 0; i < V.rows(); ++i) {
        if (std::max(std::abs(V(i, 0)), std::abs(V(i, 1))) <= patch_half * (1.0 + 1e-9)) {
            patch_max = std::max(patch_max, exc.electric_field(V.row(i).transpose()).norm());
        }
    }
    if (!(patch_max > 0.0) || !std::isfinite(patch_max)) {
        throw std::invalid_argument(
            "Simulation (PML): the incident field vanishes (or is not finite) on the patch");
    }
    const Real a = pml_start;
    const auto n = static_cast<Index>(std::ceil(2.0 * a / (0.125 * lambda1)));
    Real edge_max = 0;
    for (Index i = 0; i <= n; ++i) {
        const Real u = -a + 2.0 * a * static_cast<Real>(i) / static_cast<Real>(n);
        const Vec3 pts[4] = {Vec3(u, -a, 0.0), Vec3(u, a, 0.0), Vec3(-a, u, 0.0), Vec3(a, u, 0.0)};
        for (const Vec3& p : pts) edge_max = std::max(edge_max, exc.electric_field(p).norm());
    }
    if (!std::isfinite(edge_max)) {
        throw std::invalid_argument(
            "Simulation (PML): the incident field is not finite on the PML-start square");
    }
    return edge_max / patch_max;
}

/// Patch mesh size h: the largest in-plane extent max(|dx|, |dy|) over the vertices of a Patch
/// triangle (the grid spacing of make_open_surface_mesh, whose cells are split along a diagonal).
Real patch_mesh_size(const geometry::TriangleMesh& mesh,
                     const std::vector<geometry::SurfaceZone>& zone) {
    const Vertices& V = mesh.vertices();
    const Triangles& T = mesh.triangles();
    Real h = 0;
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        if (zone[static_cast<std::size_t>(t)] != geometry::SurfaceZone::Patch)
            continue;
        for (Index a = 0; a < 3; ++a) {
            for (Index b = a + 1; b < 3; ++b) {
                h = std::max({h, std::abs(V(T(t, a), 0) - V(T(t, b), 0)),
                              std::abs(V(T(t, a), 1) - V(T(t, b), 1))});
            }
        }
    }
    return h;
}

/// Configuration warnings of the PML route (ADR 0009 decision 4, rules after validation 7a,
/// benchmarks/results/pml_validation.md); warnings only, the dark band is the enforced check.
void warn_pml_configuration(const SimulationConfig& cfg, const excitation::Excitation& exc,
                            Real lambda1, Real h, Real pml_start, Real ratio, Real tolerance) {
    Real theta = 0;
    std::optional<Real> waist;
    // AngularSpectrumBeam (spectrum cut at k_t = k) or the paraxial GaussianBeam.
    bool rigorous = false;
    if (const auto* b = dynamic_cast<const excitation::AngularSpectrumBeam*>(&exc)) {
        theta = b->params().incidence_angle;
        waist = b->params().waist_radius;
        rigorous = true;
    } else if (const auto* g = dynamic_cast<const excitation::GaussianBeam*>(&exc)) {
        theta = g->params().incidence_angle;
        waist = g->params().waist_radius;
    } else if (const auto* p = dynamic_cast<const excitation::PlaneWave*>(&exc)) {
        theta = std::acos(std::clamp(p->direction().z(), -1.0, 1.0));
    }
    const Real deg = std::abs(theta) * 180.0 / constants::pi;
    const bool oblique = std::abs(theta) > 1e-12;
    if (oblique && h > lambda1 / 6.0 * (1.0 + 1e-9) &&
        is_weakly_absorbing(cfg.object, cfg.wavelength, h)) {
        SBEM_WARN(
            "Simulation (PML): weakly absorbing object at oblique incidence theta_in = {:.4g} deg "
            "with the patch mesh size h = {:.4g} m > lambda_1 / 6 = {:.4g} m: validation 7a "
            "(ADR 0009 decision 4, benchmarks/results/pml_validation.md) found flat Si at 45 deg "
            "mesh-converged only at h <= lambda_1 / 6 (lambda / 5: +5.9 % in R; Ag needs h <= "
            "lambda / 4)",
            deg, h, lambda1 / 6.0);
    }
    if (!waist)
        return;
    const Real w0 = *waist;
    const Real c = std::cos(theta);
    std::string why;
    if (w0 * c < lambda1 * (1.0 - 1e-9)) {
        std::ostringstream os;
        os << "w0 cos(theta_in) = " << w0 * c << " m < lambda_1 = " << lambda1 << " m";
        if (rigorous)
            os << " (the spectrum cut at k_t = k puts grazing waves of relative amplitude "
                  "exp(-(k w0 cos theta_in)^2 / 4) into the surface)";
        else
            os << " (the paraxial GaussianBeam violates Maxwell's equations at "
                  "O((lambda_1 / (pi w0))^2); AngularSpectrumBeam is the rigorous beam)";
        why = os.str();
    }
    if (pml_start < 3.5 * w0 / c * (1.0 - 1e-9)) {
        std::ostringstream os;
        os << (why.empty() ? "" : "; ") << "pml_start = " << pml_start
           << " m < 3.5 w0 / cos(theta_in) = " << 3.5 * w0 / c
           << " m (0 deg: pml_start >~ 3.5 w0; oblique: the footprint grows faster than "
              "1 / cos(theta_in), so this bound is necessary, not sufficient)";
        why += os.str();
    }
    if (!why.empty()) {
        SBEM_WARN(
            "Simulation (PML): beam waist w0 = {:.4g} m at theta_in = {:.4g} deg violates the "
            "waist rule of ADR 0009 decision 4 (rule after 7a, "
            "benchmarks/results/pml_validation.md: w0 cos(theta_in) >~ lambda_1, pml_start >~ "
            "3.5 w0 at 0 deg): {}; measured dark-band ratio {:.3e} (tolerance {:.3g})",
            w0, deg, why, ratio, tolerance);
    }
}

}  // namespace

Simulation::Simulation(geometry::OpenSurfaceMesh open, PmlTermination pml,
                       std::shared_ptr<excitation::Excitation> excitation,
                       SimulationConfig config) {
    check_open_mesh(open);
    check_pml_termination(pml);
    check_excitation(excitation, config);
    check_compression(config.compression);  // "dense" or "mlfmm" (fast PML ring, WP-PML5a)
    const Real gap = open.pml_start - open.taper_end;
    if (!(gap > 0.0)) {
        throw std::invalid_argument(
            "Simulation (PML): the gap W_g = pml_start - taper_end must be > 0 (the far-field "
            "window falls to zero across it); set OpenSurfaceParams::gap_width");
    }
    if (pml.order < 1.0) {
        const Real h = longest_edge(open.mesh);
        if (gap < h) {
            std::ostringstream os;
            os << "Simulation (PML): profile order " << pml.order
               << " < 1 needs a gap W_g = " << gap << " m >= the longest mesh edge " << h
               << " m (non-coplanar touching pairs at the PML start have no ridge split, "
                  "WP-PML2 review W5)";
            throw std::invalid_argument(os.str());
        }
    }
    const Real taper = open.taper_end - open.patch_half;
    const Vertices& V0 = open.mesh.vertices();
    bool rough = false;
    for (Index i = 0; i < V0.rows() && !rough; ++i) rough = V0(i, 2) != 0.0;
    if (rough && !(pml.correlation_length > 0.0)) {
        throw std::invalid_argument(
            "Simulation (PML): the patch is rough; set PmlTermination::correlation_length (Lc "
            "of the height map) so that the taper W_t >= Lc can be checked (ADR 0009 decision "
            "1)");
    }
    if (rough && taper < pml.correlation_length * (1.0 - 1e-9)) {
        std::ostringstream os;
        os << "Simulation (PML): the taper W_t = taper_end - patch_half = " << taper
           << " m is narrower than the correlation length " << pml.correlation_length
           << " m (ADR 0009 decision 1: W_t >= Lc); widen OpenSurfaceParams::taper_width";
        throw std::invalid_argument(os.str());
    }
    const Real lambda1 = exterior_wavelength(config.exterior, config.wavelength);
    check_huygens_options(pml.huygens, lambda1, open.pml_start);

    geometry::CoordinateStretch stretch =
        geometry::make_pml_stretch(open, pml.sigma_max, pml.order);
    impl_ = std::make_unique<Impl>(std::move(open.mesh), std::move(excitation), std::move(config));
    Impl& s = *impl_;
    s.pml.emplace(Impl::Pml{pml,
                            std::move(open.zone),
                            {open.patch_half, open.taper_end, open.pml_start, open.outer_half},
                            stretch,
                            0.0});

    // Controlled region: the vertices of the triangles the masked right-hand side integrates
    // (those carrying an unmasked test function); masked PML vertices may lie outside.
    const std::vector<std::uint8_t> masked = op::pml_masked_functions(s.space, s.pml->stretch);
    std::vector<std::uint8_t> use_vertex(static_cast<std::size_t>(s.mesh.num_vertices()), 0);
    for (Index t = 0; t < s.mesh.num_triangles(); ++t) {
        const basis::RwgSpace::Support sup = s.space.support(t);
        bool active = false;
        for (int a = 0; a < sup.count; ++a)
            active = active || masked[static_cast<std::size_t>(sup.n[a])] == 0;
        if (!active)
            continue;
        for (Index c = 0; c < 3; ++c)
            use_vertex[static_cast<std::size_t>(s.mesh.triangles()(t, c))] = 1;
    }
    check_controlled_region(s.mesh, *s.excitation, &use_vertex);

    const Real ratio =
        measure_dark_band(s.mesh, s.pml->half[0], s.pml->half[2], *s.excitation, lambda1);
    s.pml->dark_band_ratio = ratio;
    if (ratio > pml.dark_band_tolerance) {
        std::ostringstream os;
        os << "Simulation (PML): no dark band: max |E_inc| on the PML-start square "
              "max(|x|, |y|) = "
           << s.pml->half[2] << " m is " << ratio
           << " of its maximum on the patch, above PmlTermination::dark_band_tolerance = "
           << pml.dark_band_tolerance
           << " (ADR 0009 decision 4: the right-hand side is zero in the PML); use a smaller "
              "beam waist or a wider gap";
        throw std::invalid_argument(os.str());
    }
    check_operator_config(s.config, 2 * s.mesh.num_edges());
    if (s.config.compression == "mlfmm" && s.config.mlfmm.leaf_radius_quantile < 1.0) {
        // The PML ring defines near / far by leaf adjacency (every function on the leaf level).
        throw std::invalid_argument(
            "Simulation (PML): compression \"mlfmm\" on the PML route needs "
            "mlfmm.leaf_radius_quantile = 1 (the local leaf rule is not supported by the exact "
            "PML ring, WP-PML5a)");
    }
    warn_pml_configuration(s.config, *s.excitation, lambda1, patch_mesh_size(s.mesh, s.pml->zone),
                           s.pml->half[2], ratio, pml.dark_band_tolerance);

    const formulation::Recommendation rec = formulation::recommend(s.config.object.eps_r);
    const formulation::Kind kind = s.config.formulation.value_or(rec.kind);
    s.setup(rec, kind, checked_formulation(kind));
    s.problem.stretch = &s.pml->stretch;
    s.window = post::CurrentWindow{s.pml->half[1], s.pml->half[2]};
    op::validate(s.problem);
    SBEM_INFO(
        "Simulation (PML): sigma_max {}, order {}, gap {:.4g} m, window [{:.4g}, {:.4g}] m, "
        "dark band ratio {:.3e}, {} of {} test functions masked, far field from the {}",
        pml.sigma_max, pml.order, gap, s.pml->half[1], s.pml->half[2], ratio,
        std::count(masked.begin(), masked.end(), std::uint8_t{1}), masked.size(),
        source_name(pml.far_field_source));
}

Simulation::~Simulation() = default;

bool Simulation::Impl::uses_diagonal() const {
    return config.solver == SolverKind::Gmres && jacobi;
}

void Simulation::Impl::ensure_diagonal() {
    if (!uses_diagonal() || diagonal.size() > 0)
        return;
    const ScopedTimer t("Simulation: Jacobi diagonal");
    // MLFMM (with or without the PML ring): the diagonal is stored in the near parts (WP-PML5c;
    // bitwise op::assemble_diagonal without a stretch and on the real rows, to rounding on the
    // stretched rows), so it costs nothing; otherwise assembled from the support pairs.
    if (const auto* fast = dynamic_cast<const mlfmm::MlfmmOperator*>(op.get()))
        diagonal = fast->diagonal();
    else
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

bool Simulation::assembled() const {
    return impl_->op != nullptr;
}

bool Simulation::solved() const {
    return impl_->solved;
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
          << ", leaf_radius_quantile " << m.leaf_radius_quantile << ", max_exact_far_bytes "
          << m.max_exact_far_bytes;
        if (s.pml) {
            const mlfmm::PmlRingParams& r = m.pml_ring;
            o << "; pml_ring: use_aca " << (r.use_aca ? "true" : "false") << ", aca_epsilon "
              << r.aca_epsilon << ", eta " << r.eta << ", max_rank " << r.max_rank
              << ", row_clusters " << (r.row_clusters ? "true" : "false") << ", max_far_bytes "
              << r.max_far_bytes;
        }
        o << ")";
    }
    o << "\n";
    if (s.pml) {
        const Impl::Pml& p = *s.pml;
        o << "  termination:    PML (ADR 0009; sigma_max " << p.termination.sigma_max << ", order "
          << p.termination.order << "), half widths patch " << p.half[0] << " / taper " << p.half[1]
          << " / PML start " << p.half[2] << " / outer " << p.half[3] << " m, far-field window ["
          << p.half[1] << ", " << p.half[2] << "] m, dark band ratio " << p.dark_band_ratio << "\n";
        o << "  far field:      " << source_name(p.termination.far_field_source);
        if (p.termination.far_field_source == FarFieldSource::Huygens) {
            const post::HuygensOptions& h = p.termination.huygens;
            const Real lambda1 = exterior_wavelength(s.config.exterior, s.config.wavelength);
            const post::CurrentWindow w = h.window.value_or(*s.window);
            o << " (ADR 0010): z_h " << h.z_h.value_or(lambda1)
              << " m below the lowest vertex, spacing " << h.spacing.value_or(0.25 * lambda1)
              << " m, plane window [" << w.inner_half << ", " << w.outer_half << "] m";
        } else {
            o << " (ADR 0009 decision 5), window [" << p.half[1] << ", " << p.half[2] << "] m";
        }
        o << "\n";
    }
    if (s.op == nullptr) {
        o << "  operator:       not assembled\n";
        return o.str();
    }
    o << "  operator:       " << s.op->describe() << ", "
      << static_cast<double>(s.op->memory_bytes()) / 1e6 << " MB\n";
    if (const auto* fast = dynamic_cast<const mlfmm::MlfmmOperator*>(s.op.get());
        fast != nullptr && fast->ring_operator() != nullptr) {
        const mlfmm::PmlRingOperator& r = *fast->ring_operator();
        const Index n = s.space.size();
        const mlfmm::PmlRingFarStats& f = r.far_stats();
        o << "  PML ring:       " << r.stretched_count() << " stretched / "
          << n - r.stretched_count() << " real functions (MLFMM on the real ones), ring near "
          << static_cast<double>(r.near_bytes()) / 1e6 << " MB (" << r.near_pairs()
          << " pairs), far " << static_cast<double>(r.far_bytes()) / 1e6 << " MB (" << r.far_pairs()
          << " pairs, " << (f.aca ? "ACA" : "exact") << "), dense matrix would be "
          << static_cast<double>(r.estimate().dense_bytes) / 1e6 << " MB\n";
        if (f.aca) {
            o << "  ring far ACA:   eps " << f.epsilon << ", eta " << f.eta << ": " << f.aca_blocks
              << " low-rank blocks " << static_cast<double>(f.aca_bytes) / 1e6
              << " MB (ranks min / median / max " << f.rank_min << " / " << f.rank_median << " / "
              << f.rank_max << ", mean JJ " << f.rank_mean[0] << " JM " << f.rank_mean[1] << " MJ "
              << f.rank_mean[2] << " MM " << f.rank_mean[3] << "), "
              << f.exact_blocks + f.fallback_blocks << " exact blocks (" << f.fallback_blocks
              << " at the rank cap, " << f.skipped_attempts << " of them without an ACA attempt) "
              << static_cast<double>(f.exact_bytes) / 1e6 << " MB, " << f.dropped_blocks
              << " dropped, region parts dropped R1 " << f.region_dropped[0] << " / R2 "
              << f.region_dropped[1] << " (bound <= " << f.drop_tolerance
              << "); exact far ring would be " << static_cast<double>(r.estimate().far_bytes) / 1e6
              << " MB\n";
            const auto beyond = [](Real d) {
                std::ostringstream t;
                if (d >= 1e299)
                    t << "never";
                else
                    t << "beyond " << d << " m";
                return t.str();
            };
            o << "  ring far drop:  triangle pairs reduced " << f.pair_reductions << " of "
              << f.pair_distinct + f.pair_skips << " (" << f.pair_skips
              << " skipped), region parts dropped R1 " << beyond(f.drop_distance[0]) << ", R2 "
              << beyond(f.drop_distance[1]) << " (bound <= " << f.drop_tolerance << ")\n";
        }
    }
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
        o << ", rcond " << s.direct_info.rcond << " (LU "
          << solver::to_string(s.direct_info.backend) << ")";
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

bool Simulation::has_pml() const {
    return impl_->pml.has_value();
}

const std::optional<post::CurrentWindow>& Simulation::far_field_window() const {
    return impl_->window;
}

post::FieldOptions Simulation::far_field_options(int quad_degree) const {
    return far_field_options(quad_degree, far_field_source());
}

post::FieldOptions Simulation::far_field_options(int quad_degree, FarFieldSource source) const {
    post::FieldOptions o;
    o.quad_degree = quad_degree;
    o.window = impl_->window;
    if (source == FarFieldSource::Huygens) {
        if (!has_pml()) {
            throw std::logic_error(
                "Simulation::far_field_options: FarFieldSource::Huygens needs the PML route (a "
                "closed mesh has no Huygens plane)");
        }
        o.huygens = impl_->pml->termination.huygens;
    }
    return o;
}

FarFieldSource Simulation::far_field_source() const {
    return has_pml() ? impl_->pml->termination.far_field_source : FarFieldSource::Currents;
}

namespace {

/// PML route: R1 is the half space below the surface, so far-field directions must point into
/// the reflection hemisphere (k_z <= 1e-12 |k_hat|).
void require_reflection_directions(bool pml, const Vertices& directions, const char* who) {
    if (!pml)
        return;
    for (Index i = 0; i < directions.rows(); ++i) {
        const Real len = directions.row(i).norm();
        if (directions(i, 2) > 1e-12 * len) {
            throw std::invalid_argument(
                std::string("Simulation::") + who + ": direction " + std::to_string(i) +
                " has k_z > 0; on the PML route R1 is the half space below the surface, so only "
                "the reflection hemisphere k_z <= 0 has a far field");
        }
    }
}

}  // namespace

Eigen::Matrix<Complex, Eigen::Dynamic, 3> Simulation::far_field(
    const Vertices& directions, const post::FieldOptions& opt) const {
    const post::SurfaceSolution& s = solution();
    require_reflection_directions(has_pml(), directions, "far_field");
    Eigen::Matrix<Complex, Eigen::Dynamic, 3> F;
    post::far_field(s, directions, F, opt);
    return F;
}

Eigen::Matrix<Complex, Eigen::Dynamic, 3> Simulation::far_field(const Vertices& directions,
                                                                int quad_degree) const {
    return far_field(directions, far_field_options(quad_degree));
}

Real Simulation::incident_power() const {
    const excitation::Excitation* e = impl_->excitation.get();
    if (const auto* asb = dynamic_cast<const excitation::AngularSpectrumBeam*>(e))
        return asb->power();
    if (const auto* gb = dynamic_cast<const excitation::GaussianBeam*>(e))
        return gb->power();
    if (dynamic_cast<const excitation::PlaneWave*>(e) != nullptr) {
        throw std::logic_error(
            "Simulation::incident_power: a plane wave carries infinite power; normalise by its "
            "intensity |E_0|^2 / (2 eta_1) instead (cross sections, bistatic_rcs)");
    }
    throw std::logic_error(
        "Simulation::incident_power: the excitation has no finite power known to Simulation "
        "(AngularSpectrumBeam and GaussianBeam have one)");
}

VectorXr Simulation::drc(const Vertices& directions, int quad_degree) const {
    const post::SurfaceSolution& s = solution();
    require_reflection_directions(has_pml(), directions, "drc");
    return post::drc(s, directions, incident_power(), far_field_options(quad_degree));
}

post::PolarizedIntensity Simulation::drc_polarized(const Vertices& directions,
                                                   const Vec3& co_pol_direction,
                                                   int quad_degree) const {
    const post::SurfaceSolution& s = solution();
    require_reflection_directions(has_pml(), directions, "drc_polarized");
    return post::drc_polarized(s, directions, incident_power(), co_pol_direction,
                               far_field_options(quad_degree));
}

Real Simulation::radiated_power(const Vertices& directions, const VectorXr& weights,
                                int quad_degree) const {
    const post::SurfaceSolution& s = solution();
    require_reflection_directions(has_pml(), directions, "radiated_power");
    return post::radiated_power(s, directions, weights, far_field_options(quad_degree));
}

Real Simulation::reflected_power(int n_theta, int n_phi, int quad_degree) const {
    const post::SurfaceSolution& s = solution();
    return post::radiated_power(s, post::hemisphere_grid(-1.0, n_theta, n_phi),
                                far_field_options(quad_degree));
}

Real Simulation::scattered_power(int n_theta, int n_phi, int quad_degree) const {
    const post::SurfaceSolution& s = solution();
    if (has_pml()) {
        throw std::logic_error(
            "Simulation::scattered_power: not defined on the PML route (the forward hemisphere "
            "lies in R2, which has no far field); use reflected_power()");
    }
    return post::radiated_power(s, post::sphere_grid(n_theta, n_phi),
                                far_field_options(quad_degree));
}

namespace {

void require_pml(bool has, const char* who) {
    if (!has) {
        throw std::logic_error(std::string("Simulation: ") + who +
                               "() is defined only for the PML route (open-surface constructor)");
    }
}

}  // namespace

const PmlTermination& Simulation::pml_termination() const {
    require_pml(has_pml(), "pml_termination");
    return impl_->pml->termination;
}

const std::vector<geometry::SurfaceZone>& Simulation::zones() const {
    require_pml(has_pml(), "zones");
    return impl_->pml->zone;
}

const geometry::CoordinateStretch& Simulation::stretch() const {
    require_pml(has_pml(), "stretch");
    return impl_->pml->stretch;
}

std::array<Real, 4> Simulation::zone_half_widths() const {
    require_pml(has_pml(), "zone_half_widths");
    return impl_->pml->half;
}

Real Simulation::dark_band_ratio() const {
    require_pml(has_pml(), "dark_band_ratio");
    return impl_->pml->dark_band_ratio;
}

}  // namespace specklebem
