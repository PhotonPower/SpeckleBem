#pragma once
/// @file pml_fresnel_support.hpp
/// Validation 7a of the open-surface PML route (ADR 0009, WP-PML3b; docs/05 "PML route" block):
/// a flat Ag or Si open surface (patch, taper, gap, PML; height map zero) terminated by the
/// complex-coordinate-stretching PML, under the rigorous angular-spectrum beam at incidence
/// theta (about y, from z < 0), s and p polarisation; the windowed reflected power (far field of
/// the real part, window across the gap, Simulation::far_field_options()) over the beam power
/// against the beam reflectance R_beam of the same beam on the infinite interface
/// (fresnel_flat::beam_reflectance). Shared by the study executable benchmarks/pml_fresnel.cpp and
/// the memory-guarded test tests/validation_large/test_pml_fresnel_large.cpp; no Catch2.
///
/// One run = one geometry and one operator, several polarisations: the dense operator does not
/// depend on the excitation, so it is assembled and LU-factorised once and every polarisation is
/// one more right-hand side (op::assemble_rhs of a second Simulation built with that beam, which
/// also runs the dark-band and controlled-region checks of the PML constructor). The LU uses
/// LAPACK zgetrf/zgetrs when CMake found BLAS/LAPACK (multithreaded, ~10-30x faster than Eigen's
/// PartialPivLU at 2N ~ 2e4..5e4) and Eigen's PartialPivLU otherwise; it factors a copy of the
/// matrix (2 x 16 (2N)^2 bytes peak) so that the residual |b - Z x| / |b| is checked against the
/// assembled matrix.
#include "specklebem/geometry/open_surface.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/operator/dense_operator.hpp"

#include <Eigen/LU>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "fresnel_flat_support.hpp"
#include "system_memory.hpp"

#ifdef SPECKLEBEM_HAVE_BLAS_LAPACK
extern "C" {
// LAPACK (Fortran interface, LP64 integers as in OpenBLAS / reference LAPACK / MKL LP64).
void zgetrf_(const int* m, const int* n, std::complex<double>* a, const int* lda, int* ipiv,
             int* info);
void zgetrs_(const char* trans, const int* n, const int* nrhs, const std::complex<double>* a,
             const int* lda, const int* ipiv, std::complex<double>* b, const int* ldb, int* info,
             std::size_t trans_len);
}
#endif

namespace pml_fresnel {

using namespace specklebem;
using fresnel_flat::kDeg;
using fresnel_flat::kLambda;

/// One run: geometry, PML, kernel options, the polarisations solved with one LU.
struct PmlCase {
    std::string material = "si";  ///< "si" (silicon_500nm) or "ag" (silver_500nm)
    Real theta_deg = 0.0;         ///< incidence angle about y [deg]
    std::vector<excitation::Polarization> pols = {excitation::Polarization::P,
                                                  excitation::Polarization::S};
    Real waist = 0.5e-6;      ///< w0 [m]
    Real patch = 1.5e-6;      ///< patch edge L [m] (flat: only the zone boundary a_0 = L / 2)
    Real taper = 0.5e-6;      ///< W_t [m]
    Real gap = 0.5e-6;        ///< W_g [m] (the default far-field window spans it)
    Real pml_width = 1.0e-6;  ///< W_pml [m] (2 lambda_1 at 500 nm)
    Real mesh_size = 125e-9;  ///< h [m]; must divide L / 2, W_t, W_g, W_pml for no rounding
    Real sigma_max = 4.0;     ///< PmlTermination::sigma_max
    Real order = 2.0;         ///< PmlTermination::order
    Real dark_band_tolerance = 1e-4;
    Real target_accuracy = 1e-5;         ///< kernels::OperatorOptions::target_accuracy
    int stretched_touching_points = 10;  ///< kernels::OperatorOptions::stretched_touching_points
    /// Far-field window override (both or neither); empty: Simulation's (taper_end, pml_start).
    std::optional<Real> window_inner;
    std::optional<Real> window_outer;
    Real ff_dtheta_deg = 1.0;  ///< hemisphere grid (Gauss-Legendre in theta, phi twice as coarse)
    int ff_degree = 6;         ///< Dunavant degree of the far-field source quadrature
    bool use_lapack = true;    ///< LAPACK LU when available (else Eigen PartialPivLU)
};

/// The validation 7a configuration at h = lambda / 4 (benchmarks/results/pml_validation.md) for
/// the material "si" or "ag" at theta_deg = 0 or 45: taper = gap = 0.5 um, W_pml = 2 lambda =
/// 1 um, sigma_max 4, order 2, both polarisations, default dark band (1e-4), and a waist / patch
/// pair on the measured grid that meets that dark band:
///  *  0 deg: w0 = lambda = 0.5 um, L = 1.5 um (pml_start 1.75 um = 3.5 w0; ratio 1.3e-5);
///  * 45 deg: w0 = 0.75 um, L = 7 um (pml_start 4.5 um; ratio 5.2e-5 / 4.0e-5 for p / s), chosen
///    with a ~2x margin; smaller passing points are w0 = 0.75 um, L = 6.5 um (8.9e-5) and
///    w0 = 0.7 um, L = 7 um (7.7e-5). The waist must satisfy w0 cos(theta) >~ lambda (the
///    spectrum cut at k_t = k puts grazing plane waves of relative amplitude
///    exp(-(k w0 cos theta)^2 / 4) into the surface), and the beam defocuses along the footprint
///    (Rayleigh range pi w0^2 / lambda), so pml_start grows faster than 3.5 w0 / cos(theta).
/// Ag (0 and 45 deg) and Si at 0 deg are accepted at this mesh; Si at 45 deg is not
/// mesh-converged there (h = lambda / 4 is 1.08 Si wavelengths; lambda / 5 gave +5.9 % for p), its
/// acceptance case is seven_a_si45_fine_case().
/// @throws std::invalid_argument for another angle.
[[nodiscard]] inline PmlCase seven_a_case(const std::string& material, Real theta_deg) {
    PmlCase c;
    c.material = material;
    c.theta_deg = theta_deg;
    c.taper = 0.5e-6;
    c.gap = 0.5e-6;
    c.pml_width = 2.0 * kLambda;
    c.mesh_size = kLambda / 4.0;
    c.sigma_max = 4.0;
    c.order = 2.0;
    if (theta_deg == 0.0) {
        c.waist = kLambda;
        c.patch = 1.5e-6;
    } else if (theta_deg == 45.0) {
        c.waist = 1.5 * kLambda;
        c.patch = 7.0e-6;
    } else {
        throw std::invalid_argument("seven_a_case: theta_deg must be 0 or 45");
    }
    return c;
}

/// Mesh-converged Si 45 deg case of 7a: h = lambda / 6 (0.72 Si wavelengths; the power balance
/// 1 - R - A_real closes to 0.2 %), which the dark-band geometry of seven_a_case() cannot afford
/// densely (2N = 104 016, 173 GB dense). Smaller geometry instead: w0 = lambda, L = 3 um
/// (pml_start 2.5 um), dark-band tolerance relaxed to 1e-2 (measured ratio 5.9e-3 / 5.6e-3; at
/// h = lambda / 4 this geometry and the 1e-4 one of seven_a_case() differ by <= 0.06 % in R, at
/// h = lambda / 6 it and the 2.6e-2 geometry L = 1.5 um by 0.05 % / 0.08 %), 2N = 42 000. The
/// dark band is an ADR 0009 decision 4 configuration check; the 7a tolerance is unchanged.
[[nodiscard]] inline PmlCase seven_a_si45_fine_case() {
    PmlCase c = seven_a_case("si", 45.0);
    c.waist = kLambda;
    c.patch = 3.0e-6;
    c.mesh_size = kLambda / 6.0;
    c.dark_band_tolerance = 1e-2;
    return c;
}

[[nodiscard]] inline const char* pol_name(excitation::Polarization p) {
    return p == excitation::Polarization::P ? "p" : "s";
}

/// Open flat mesh of a case.
[[nodiscard]] inline geometry::OpenSurfaceMesh make_open_mesh(const PmlCase& c) {
    geometry::OpenSurfaceParams op;
    op.patch_size = c.patch;
    op.taper_width = c.taper;
    op.gap_width = c.gap;
    op.pml_width = c.pml_width;
    return geometry::make_open_surface_mesh(
        fresnel_flat::flat_height_map(c.patch + 2.0 * c.taper, c.mesh_size), op);
}

/// Beam of a case: focus at the origin, region radius covering the unmasked part (the corners of
/// the gap square, 1.02 sqrt(2) pml_start).
[[nodiscard]] inline std::shared_ptr<excitation::AngularSpectrumBeam> make_beam(
    const PmlCase& c, excitation::Polarization pol, Real pml_start) {
    excitation::AngularSpectrumBeam::Params bp;
    bp.wavelength = kLambda;
    bp.waist_radius = c.waist;
    bp.focus = Vec3::Zero();
    bp.incidence_angle = c.theta_deg * kDeg;
    bp.polarization = pol;
    bp.region_radius = 1.02 * std::sqrt(2.0) * pml_start;
    return std::make_shared<excitation::AngularSpectrumBeam>(bp, material::vacuum());
}

[[nodiscard]] inline SimulationConfig make_config(const PmlCase& c) {
    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.exterior = material::vacuum();
    cfg.object = fresnel_flat::object_material(c.material);
    cfg.compression = "dense";
    cfg.solver = SolverKind::Direct;
    cfg.kernels.target_accuracy = c.target_accuracy;
    cfg.kernels.stretched_touching_points = c.stretched_touching_points;
    return cfg;
}

[[nodiscard]] inline PmlTermination make_termination(const PmlCase& c) {
    PmlTermination p;
    p.sigma_max = c.sigma_max;
    p.order = c.order;
    p.dark_band_tolerance = c.dark_band_tolerance;
    return p;
}

/// Mesh size of a case without assembly.
struct Size {
    Index triangles = 0;
    Index unknowns = 0;                                                  ///< 2N
    std::array<Real, 4> half{};                                          ///< a_0 .. a_3 [m]
    Real dense_bytes = 0;                                                ///< 16 (2N)^2
    [[nodiscard]] Real peak_bytes() const { return 2.0 * dense_bytes; }  ///< matrix + LU copy
};

[[nodiscard]] inline Size size_of(const PmlCase& c) {
    const geometry::OpenSurfaceMesh om = make_open_mesh(c);
    Size s;
    s.triangles = om.mesh.num_triangles();
    s.unknowns = 2 * om.mesh.num_edges();
    s.half = {om.patch_half, om.taper_end, om.pml_start, om.outer_half};
    s.dense_bytes = 16.0 * static_cast<Real>(s.unknowns) * static_cast<Real>(s.unknowns);
    return s;
}

/// Dense LU of a copy of Z, solving several right-hand sides.
class DenseLu {
public:
    DenseLu(const MatrixXc& Z, bool use_lapack) : n_(Z.rows()) {
        if (Z.rows() != Z.cols())
            throw std::invalid_argument("DenseLu: matrix must be square");
#ifdef SPECKLEBEM_HAVE_BLAS_LAPACK
        lapack_ = use_lapack;
#else
        (void)use_lapack;
#endif
        if (lapack_) {
#ifdef SPECKLEBEM_HAVE_BLAS_LAPACK
            if (n_ > static_cast<Index>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("DenseLu: n too large for LP64 LAPACK");
            lu_ = Z;
            const int n = static_cast<int>(n_);
            ipiv_.resize(static_cast<std::size_t>(n_));
            int info = 0;
            zgetrf_(&n, &n, lu_.data(), &n, ipiv_.data(), &info);
            if (info != 0)
                throw std::runtime_error("DenseLu: zgetrf failed, info = " + std::to_string(info));
#endif
        } else {
            eigen_ = std::make_unique<Eigen::PartialPivLU<MatrixXc>>(Z);
        }
    }
    [[nodiscard]] VectorXc solve(const VectorXc& b) const {
        if (b.size() != n_)
            throw std::invalid_argument("DenseLu: right-hand side size mismatch");
        if (!lapack_)
            return eigen_->solve(b);
        VectorXc x = b;
#ifdef SPECKLEBEM_HAVE_BLAS_LAPACK
        const int n = static_cast<int>(n_);
        const int nrhs = 1;
        int info = 0;
        const char trans = 'N';
        zgetrs_(&trans, &n, &nrhs, lu_.data(), &n, ipiv_.data(), x.data(), &n, &info, 1);
        if (info != 0)
            throw std::runtime_error("DenseLu: zgetrs failed, info = " + std::to_string(info));
#endif
        return x;
    }
    [[nodiscard]] bool lapack() const { return lapack_; }

private:
    Index n_;
    bool lapack_ = false;
    MatrixXc lu_;
    std::vector<int> ipiv_;
    std::unique_ptr<Eigen::PartialPivLU<MatrixXc>> eigen_;
};

/// Largest |J|, |M| at the centroids of the outermost ring of triangles (max(|x|, |y|) > a_3 -
/// h) relative to the largest over all triangles.
struct EdgeDecay {
    Real j = 0;
    Real m = 0;
};

[[nodiscard]] inline EdgeDecay edge_decay(const post::SurfaceSolution& s, Real outer_half, Real h) {
    const basis::RwgSpace& space = *s.problem->space;
    const geometry::TriangleMesh& mesh = space.mesh();
    const Index N = space.size();
    Real j_max = 0, m_max = 0, j_edge = 0, m_edge = 0;
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        const Vec3 c = mesh.centroid(t);
        const basis::RwgSpace::Support sup = space.support(t);
        Vec3c J = Vec3c::Zero(), M = Vec3c::Zero();
        for (int a = 0; a < sup.count; ++a) {
            const Vec3 f = space.value(sup.n[a], t, c);
            J += s.currents(sup.n[a]) * f.cast<Complex>();
            M += s.currents(N + sup.n[a]) * f.cast<Complex>();
        }
        j_max = std::max(j_max, J.norm());
        m_max = std::max(m_max, M.norm());
        if (std::max(std::abs(c.x()), std::abs(c.y())) > outer_half - h) {
            j_edge = std::max(j_edge, J.norm());
            m_edge = std::max(m_edge, M.norm());
        }
    }
    return {j_edge / j_max, m_edge / m_max};
}

/// 1/2 Re int (n x M) . J* dS over the triangles whose centroid lies inside max(|x|, |y|) <
/// half (the power entering R2 through the real, unstretched part; diagnostic of the power
/// split, exact for the quadratic integrand with Dunavant degree 2).
[[nodiscard]] inline Real absorbed_power_inside(const post::SurfaceSolution& s, Real half) {
    const basis::RwgSpace& space = *s.problem->space;
    const geometry::TriangleMesh& mesh = space.mesh();
    const Index N = space.size();
    const kernels::TriangleRule& rule = kernels::triangle_rule(2);
    Real p_abs = 0.0;
    for (Index t = 0; t < mesh.num_triangles(); ++t) {
        const Vec3 cen = mesh.centroid(t);
        if (!(std::max(std::abs(cen.x()), std::abs(cen.y())) < half))
            continue;
        const Vec3 v0 = mesh.vertices().row(mesh.triangles()(t, 0)).transpose();
        const Vec3 v1 = mesh.vertices().row(mesh.triangles()(t, 1)).transpose();
        const Vec3 v2 = mesh.vertices().row(mesh.triangles()(t, 2)).transpose();
        const Vec3 nt = mesh.normal(t);
        const basis::RwgSpace::Support sup = space.support(t);
        Real acc = 0.0;
        for (std::size_t q = 0; q < rule.weights.size(); ++q) {
            const Vec3& l = rule.barycentric[q];
            const Vec3 r = l(0) * v0 + l(1) * v1 + l(2) * v2;
            Vec3c J = Vec3c::Zero();
            Vec3c M = Vec3c::Zero();
            for (int a = 0; a < sup.count; ++a) {
                const Vec3 f = space.value(sup.n[a], t, r);
                J += s.currents(sup.n[a]) * f.cast<Complex>();
                M += s.currents(N + sup.n[a]) * f.cast<Complex>();
            }
            const Vec3c nxm(nt.y() * M.z() - nt.z() * M.y(), nt.z() * M.x() - nt.x() * M.z(),
                            nt.x() * M.y() - nt.y() * M.x());
            const Complex p =
                nxm(0) * std::conj(J(0)) + nxm(1) * std::conj(J(1)) + nxm(2) * std::conj(J(2));
            acc += rule.weights[q] * p.real();
        }
        p_abs += 0.5 * mesh.area(t) * acc;
    }
    return p_abs;
}

/// Result of one polarisation.
struct PolResult {
    excitation::Polarization pol = excitation::Polarization::P;
    Real dark_band = 0;  ///< Simulation::dark_band_ratio()
    Real power = 0;      ///< AngularSpectrumBeam::power() [W]
    fresnel_flat::BeamReflectance ref;
    fresnel_flat::HemispherePower refl;  ///< primary grid, ff_degree
    Real p_refl_grid = 0;                ///< half the nodes per direction [W]
    Real p_refl_deg = 0;                 ///< half the nodes, Dunavant degree 8 [W]
    Real p_abs_real = 0;                 ///< absorbed_power_inside(pml_start) [W]
    Real residual = 0;                   ///< |b - Z x| / |b|
    EdgeDecay decay;
    Index plane_waves = 0;
    [[nodiscard]] Real r_sim() const { return refl.total / power; }
    [[nodiscard]] Real error() const { return r_sim() / ref.reflectance - 1.0; }
    [[nodiscard]] Real a_real() const { return p_abs_real / power; }
};

/// Result of one run.
struct RunResult {
    Size size;
    Index masked = 0;  ///< masked test functions (of N)
    post::CurrentWindow window;
    std::string formulation;
    bool lapack = false;
    Real setup_s = 0;     ///< beams and Simulation construction (checks)
    Real assembly_s = 0;  ///< operator + first right-hand side
    Real lu_s = 0;
    Real solve_s = 0;  ///< right-hand sides (after the first) + back substitutions
    Real post_s = 0;
    Real peak_rss = 0;  ///< process peak [bytes]
    std::vector<PolResult> pols;
    std::string report;
};

/// Dark-band ratio per polarisation of c.pols, measured by the PML constructor of Simulation
/// with the tolerance relaxed to 1 (no assembly; the other checks still apply).
[[nodiscard]] inline std::vector<Real> dark_band_ratios(const PmlCase& c) {
    PmlCase relaxed = c;
    relaxed.dark_band_tolerance = 1.0;
    const Size s = size_of(c);
    std::vector<Real> out;
    for (const excitation::Polarization pol : c.pols) {
        const Simulation sim(make_open_mesh(relaxed), make_termination(relaxed),
                             make_beam(relaxed, pol, s.half[2]), make_config(relaxed));
        out.push_back(sim.dark_band_ratio());
    }
    return out;
}

/// Builds, assembles, factorises and solves a case for every polarisation of c.pols.
inline RunResult run(const PmlCase& c) {
    using Clock = std::chrono::steady_clock;
    const auto secs = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<Real>(b - a).count();
    };
    if (c.pols.empty())
        throw std::invalid_argument("pml_fresnel: no polarisation");
    if (c.window_inner.has_value() != c.window_outer.has_value())
        throw std::invalid_argument("pml_fresnel: set both window bounds or neither");
    RunResult r;
    r.size = size_of(c);
    const auto t0 = Clock::now();
    const SimulationConfig cfg = make_config(c);
    const PmlTermination term = make_termination(c);
    std::vector<std::shared_ptr<excitation::AngularSpectrumBeam>> beams;
    std::vector<std::unique_ptr<Simulation>> sims;
    for (const excitation::Polarization pol : c.pols) {
        beams.push_back(make_beam(c, pol, r.size.half[2]));
        sims.push_back(std::make_unique<Simulation>(make_open_mesh(c), term, beams.back(), cfg));
    }
    Simulation& sim = *sims.front();
    // The operator of the first Simulation is reused for the right-hand sides of the others:
    // their spaces must be the same (same size and bitwise the same vertices).
    const basis::RwgSpace& space0 = *sim.problem().space;
    for (std::size_t i = 1; i < sims.size(); ++i) {
        const basis::RwgSpace& space_i = *sims[i]->problem().space;
        const Vertices& v0 = space0.mesh().vertices();
        const Vertices& vi = space_i.mesh().vertices();
        if (space_i.size() != space0.size() || vi.rows() != v0.rows() ||
            !(vi.array() == v0.array()).all())
            throw std::logic_error("pml_fresnel: the Simulations of one run differ in mesh/space");
    }
    const std::vector<std::uint8_t> mask =
        op::pml_masked_functions(*sim.problem().space, sim.stretch());
    r.masked = static_cast<Index>(std::count(mask.begin(), mask.end(), std::uint8_t{1}));
    r.window = *sim.far_field_window();
    if (c.window_inner)
        r.window = post::CurrentWindow{*c.window_inner, *c.window_outer};
    r.formulation = formulation::make_formulation(sim.formulation_kind())->name();
    const auto t1 = Clock::now();
    r.setup_s = secs(t0, t1);
    sim.assemble();
    const auto t2 = Clock::now();
    r.assembly_s = secs(t1, t2);
    const auto dense = std::dynamic_pointer_cast<const op::DenseOperator>(sim.system_operator());
    if (!dense)
        throw std::logic_error("pml_fresnel: the PML route operator is not dense");
    const MatrixXc& Z = dense->matrix();
    std::vector<VectorXc> x(c.pols.size());
    {
        const DenseLu lu(Z, c.use_lapack);
        r.lapack = lu.lapack();
        const auto t3 = Clock::now();
        r.lu_s = secs(t2, t3);
        for (std::size_t i = 0; i < c.pols.size(); ++i) {
            const VectorXc b = i == 0 ? sim.rhs() : op::assemble_rhs(sims[i]->problem());
            x[i] = lu.solve(b);
            PolResult pr;
            pr.pol = c.pols[i];
            pr.residual = (b - Z * x[i]).norm() / b.norm();
            r.pols.push_back(pr);
        }
        r.solve_s = secs(t3, Clock::now());
    }
    const auto t4 = Clock::now();
    const Real eta1 = material::vacuum().wave_impedance(beams.front()->omega()).real();
    const Real th = c.theta_deg * kDeg;
    const Vec3 specular(std::sin(th), 0.0, -std::cos(th));
    const Real cone = 3.0 * kLambda / (constants::pi * c.waist);
    const fresnel_flat::HemisphereGrid g = fresnel_flat::hemisphere_grid(-1.0, c.ff_dtheta_deg);
    const fresnel_flat::HemisphereGrid g2 = fresnel_flat::hemisphere_grid(-1.0, c.ff_dtheta_deg, 2);
    for (std::size_t i = 0; i < c.pols.size(); ++i) {
        PolResult& pr = r.pols[i];
        const post::SurfaceSolution s{&sims[i]->problem(), x[i]};
        post::FieldOptions fo = sims[i]->far_field_options(c.ff_degree);
        fo.window = r.window;
        post::FieldOptions fo8 = fo;
        fo8.quad_degree = 8;
        pr.dark_band = sims[i]->dark_band_ratio();
        pr.power = beams[i]->power();
        pr.plane_waves = beams[i]->num_plane_waves();
        pr.ref = fresnel_flat::beam_reflectance(*beams[i], 1.0, cfg.object.eps_r);
        pr.refl = fresnel_flat::hemisphere_power(s, g, fo, eta1, specular, cone);
        pr.p_refl_grid = fresnel_flat::hemisphere_power(s, g2, fo, eta1, specular, cone).total;
        pr.p_refl_deg = fresnel_flat::hemisphere_power(s, g2, fo8, eta1, specular, cone).total;
        pr.p_abs_real = absorbed_power_inside(s, r.size.half[2]);
        pr.decay = edge_decay(s, r.size.half[3], c.mesh_size);
    }
    r.post_s = secs(t4, Clock::now());
    r.report = sim.report();
    r.peak_rss = system_memory::peak_rss_bytes();
    return r;
}

/// One machine-readable line per polarisation.
[[nodiscard]] inline std::string row_line(const PmlCase& c, const RunResult& r,
                                          const PolResult& p) {
    char buf[2000];
    const int len = std::snprintf(
        buf, sizeof(buf),
        "ROW | %s | theta %.4g | pol %s | w0 %.4g | patch %.4g | taper %.4g | gap %.4g | pml %.4g "
        "| "
        "h %.4g | sigma %.3g | m %.3g | acc %.1e | stp %d | half %.4g/%.4g/%.4g/%.4g | window "
        "%.4g/%.4g | 2N %lld | masked %lld of %lld | dark %.3e | R_sim %.7f | R_beam %.7f | F "
        "%.7f | err %+.4f %% | grid %+.2e | deg8 %+.2e | cone %.6f | grazing %.2e | A_real %.6f "
        "| 1-R-A %+.3e | edge J %.2e M %.2e | res %.1e | %s | assembly %.1f s | LU %.1f s | "
        "solve %.1f s | post %.1f s | peak %.2f GB",
        c.material.c_str(), c.theta_deg, pol_name(p.pol), c.waist, c.patch, c.taper, c.gap,
        c.pml_width, c.mesh_size, c.sigma_max, c.order, c.target_accuracy,
        c.stretched_touching_points, r.size.half[0], r.size.half[1], r.size.half[2], r.size.half[3],
        r.window.inner_half, r.window.outer_half, static_cast<long long>(r.size.unknowns),
        static_cast<long long>(r.masked), static_cast<long long>(r.size.unknowns / 2), p.dark_band,
        p.r_sim(), p.ref.reflectance, p.ref.fresnel_central, 100.0 * p.error(),
        p.p_refl_grid / p.refl.total - 1.0, p.p_refl_deg / p.refl.total - 1.0,
        p.refl.cone / p.refl.total, p.refl.grazing / p.refl.total, p.a_real(),
        1.0 - p.r_sim() - p.a_real(), p.decay.j, p.decay.m, p.residual,
        r.lapack ? "lapack" : "eigen", r.assembly_s, r.lu_s, r.solve_s, r.post_s, r.peak_rss / 1e9);
    if (len < 0 || static_cast<std::size_t>(len) >= sizeof(buf))
        throw std::logic_error("pml_fresnel::row_line: formatting failed or line truncated");
    return buf;
}

}  // namespace pml_fresnel
