#pragma once
/// @file mlfmm_scaling_support.hpp
/// One case of the MLFMM scaling study (WP22b1, Phase 4 definition of done in docs/01: "scaling
/// exponent gamma (time per solve vs N) <= 1.5 for Si rough surfaces, <= 2.0 for Ag, over N =
/// 5e4 ... 1e6"). Shared by the study executable benchmarks/mlfmm_scaling.cpp and the smoke test
/// tests/unit/test_mlfmm_scaling.cpp, so it must not depend on Catch2.
///
/// Setup (docs/06, ADR 0006 with its 2026-10-10 amendment, ADR 0008): Gaussian rough surface
/// (sigma = 50 nm by default, Case::sigma; Lc = 500 nm, mesh 50 nm, fixed seed) on an L x L patch,
/// closed by the graded box chosen by rough_surface_box_params(..., BoxGradingMode::Graded)
/// (simulation.hpp): depth default_box_depth (Si 5.65 um, Ag 2 um), lambda1 = 500 nm and, for
/// Si, the fine band 3 delta + 3 sigma. The coarse spacing is explicit, box_mesh_size = 100 nm =
/// lambda1 / 5 by default (the recorded series; box_mesh_size = mesh_size gives the uniform box,
/// the library default since the ADR 0006 amendment 2026-10-12b); unset, the generator's
/// automatic rule capped at lambda1 / 5 applies; `uncapped_box` reproduces the record's "auto"
/// series (the automatic rule without the cap). The waist rule check_beam_waist with the
/// material's fraction (beam_waist_fraction: w0 <= L / 6 for Si, L / 4 for Ag; ADR 0006
/// amendment 2026-10-12b) is enforced unless allow_wide_beam. Object Si (eps_r = 18.478 -
/// 0.606j) or Ag (-9.794 - 0.313j) in vacuum, lambda = 500 nm; Gaussian beam at normal incidence
/// (travelling +z, focus at the origin, E along x, waist w0 = L / waist_factor_of(case), by
/// default L / 6 for Si and L / 4 for Ag; the recorded WP22b/WP22d series used L / 4 for both):
/// the paraxial excitation::GaussianBeam of the recorded series, or the rigorous
/// excitation::AngularSpectrumBeam (Beam::angular_spectrum, controlled ball covering the
/// mesh). Simulation with compression
/// "mlfmm" (d0 = 3, automatic leaf rule and exact-part budget), formulation per
/// formulation::recommend (Si: ICTF, Ag: ICTF + left Jacobi), full GMRES with tolerance 1e-3.
///
/// Per case: 2N, octree levels and leaf edge, setup time (Simulation::assemble) split into the
/// octree, the far setup (order searches, block checks, translators, the Ag exact part) and the
/// near field, the time per matvec (median of several applies of the MLFMM operator, and of its
/// near and far parts separately), GMRES iterations and solve time, memory per component and the
/// peak working set after the setup and at the end.
#include "specklebem/basis/rwg.hpp"
#include "specklebem/compression/mlfmm/far_operator.hpp"
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/near_field.hpp"
#include "specklebem/compression/mlfmm/octree.hpp"
#include "specklebem/compression/mlfmm/patterns.hpp"
#include "specklebem/compression/mlfmm/plane_wave.hpp"
#include "specklebem/core/logging.hpp"
#include "specklebem/core/path.hpp"
#include "specklebem/core/types.hpp"
#include "specklebem/excitation/angular_spectrum_beam.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/kernels/quadrature.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/sparse_operator.hpp"
#include "specklebem/simulation.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fresnel_flat_support.hpp"
#include "system_memory.hpp"

namespace mlfmm_scaling {

using namespace specklebem;

inline constexpr Real kLambda = 500e-9;
inline constexpr Real kSigma = 50e-9;
inline constexpr Real kCorrelationLength = 500e-9;
inline constexpr Real kMeshSize = 50e-9;
/// Default coarse spacing of the closing box: lambda1 / 5 (WP-V1 recommendation 1).
inline constexpr Real kBoxMeshSize = 100e-9;

/// Incident beam of a case (both: normal incidence, focus at the origin, E along x, w0 = L /
/// waist_factor).
enum class Beam {
    paraxial,         ///< excitation::GaussianBeam (the recorded series)
    angular_spectrum  ///< excitation::AngularSpectrumBeam, controlled ball covering the mesh
};

/// One case of the study.
struct Case {
    std::string material = "si";  ///< "si" or "ag"
    Real L = 4e-6;                ///< patch edge [m]
    /// RMS height sigma of the Gaussian surface [m] (the recorded series: 50 nm; the Phase 4
    /// Si 30 x 30 um target of docs/01: 100 nm). Also enters the fine band 3 delta + 3 sigma.
    Real sigma = kSigma;
    Real mesh_size = kMeshSize;  ///< top-face spacing [m]
    std::uint64_t seed = 1;
    /// Coarse spacing of the closing box [m]; empty: the automatic rule of
    /// geometry::RoughSurfaceParams capped at lambda1 / 5 (rough_surface_box_params).
    std::optional<Real> box_mesh_size = kBoxMeshSize;
    /// With box_mesh_size empty: the automatic rule without the lambda1 / 5 cap (exterior
    /// wavelength unset; the record's "auto" series, which the generator logs as not validated
    /// under illumination).
    bool uncapped_box = false;
    /// ADR 0006 fine band 3 delta + 3 sigma; empty: as rough_surface_box_params (Si yes, Ag no).
    std::optional<bool> fine_band;
    /// Box depth [m]; empty: default_box_depth (Si 5.65 um, Ag 2 um), as in every recorded row.
    /// Set only by the fast smoke tests (fewer wall unknowns; no benchmark option); a set depth
    /// is named in the label and the ROW line (depth_um).
    std::optional<Real> box_depth;
    /// w0 = L / waist_factor; empty: 1 / beam_waist_fraction of the object (Si 6, Ag 4 at
    /// h = 50 nm; ADR 0006 amendment 2026-10-12b). The recorded WP22b/WP22d rows used 4 for both
    /// materials (Si: reproduce with waist_factor = 4 and allow_wide_beam).
    std::optional<Real> waist_factor;
    /// check_beam_waist(L, w0, {.allow_wide = ..., .max_fraction = beam_waist_fraction}).
    bool allow_wide_beam = false;
    Beam beam = Beam::paraxial;
    Real digits = 3.0;  ///< accuracy_digits d0
    Real tolerance = 1e-3;
    int max_iter = 3000;
    int restart = 0;                      ///< 0: full GMRES
    int matvecs = 5;                      ///< timed applies per operator (median)
    bool solve = true;                    ///< false: setup and matvec timings only (no GMRES)
    std::size_t max_exact_far_bytes = 0;  ///< 0: automatic budget
    /// MlfmmParams::leaf_radius_quantile (local leaf rule, ADR 0008 amendment 2026-10-11); 1 =
    /// the global WP21 rule of the recorded WP22b1 rows.
    Real leaf_radius_quantile = mlfmm::MlfmmParams{}.leaf_radius_quantile;
    std::optional<formulation::Kind> formulation;
    std::optional<bool> jacobi;
    /// > 0: after the solve, the reflectance R = P_refl / P_inc and the absorbed fraction A =
    /// P_abs / P_inc (reflected_power; needs Beam::angular_spectrum for P_inc) on a reflection
    /// hemisphere grid with this polar spacing [deg]; 0: off (the recorded scaling rows).
    Real far_field_dtheta_deg = 0.0;
    int far_field_degree = 6;  ///< Dunavant degree of the far-field source quadrature
};

inline material::Material object_material(const std::string& name) {
    if (name == "si")
        return material::silicon_500nm();
    if (name == "ag")
        return material::silver_500nm();
    throw std::invalid_argument("mlfmm_scaling: material must be \"si\" or \"ag\", got \"" + name +
                                "\"");
}

/// Beam-waist fraction of a case's object: beam_waist_fraction (1/6 Si, 1/4 Ag at 50 nm).
inline Real waist_fraction(const Case& c) {
    return beam_waist_fraction(object_material(c.material), kLambda, c.mesh_size);
}

/// w0 = L / waist_factor_of(c): the case's waist_factor, or 1 / waist_fraction(c) (6 or 4).
inline Real waist_factor_of(const Case& c) {
    return c.waist_factor ? *c.waist_factor : std::round(1.0 / waist_fraction(c));
}

/// Closing-box parameters of a case: rough_surface_box_params (vacuum background, graded
/// option: the study's coarse spacing is explicit or the generator's capped rule) with the
/// case's overrides (explicit coarse spacing, fine band on / off).
inline RoughBoxParams box_params(const Case& c) {
    const material::Material object = object_material(c.material);
    RoughBoxParams b = rough_surface_box_params(object, material::vacuum(), kLambda, c.sigma,
                                                c.mesh_size, BoxGradingMode::Graded);
    if (c.box_mesh_size)
        b.box_mesh_size = *c.box_mesh_size;
    if (c.box_depth)
        b.box_depth = *c.box_depth;
    if (c.fine_band) {
        if (*c.fine_band)
            b.box_fine_depth = default_box_fine_depth(object, kLambda, c.sigma);
        else
            b.box_fine_depth.reset();
    }
    return b;
}

inline bool uses_fine_band(const Case& c) {
    return box_params(c).box_fine_depth.has_value();
}

/// Rough box mesh of a case and its grading data.
struct Geometry {
    geometry::TriangleMesh mesh;
    Real depth = 0;
    std::optional<Real> fine_depth;
    geometry::detail::BoxGrading grading;
    Index top_triangles = 0;
};

/// Builds the rough box of a case. Validates the case and the waist rule
/// check_beam_waist(L, L / waist_factor_of(c), {.allow_wide = allow_wide_beam, .max_fraction =
/// waist_fraction(c)}) (ADR 0006 amendments: w0 <= L / 6 for Si, L / 4 for Ag at the normal
/// incidence of the study).
/// @throws std::invalid_argument for L, mesh_size, sigma, waist_factor or a set box_depth not
///         > 0, seed 0, an unknown material, uncapped_box with an explicit box_mesh_size, or a
///         waist above the material's limit without allow_wide_beam.
inline Geometry make_geometry(const Case& c) {
    if (!(c.L > 0.0) || !(c.mesh_size > 0.0) || !(c.sigma > 0.0) ||
        (c.waist_factor && !(*c.waist_factor > 0.0)) || c.seed == 0 ||
        (c.box_depth && !(*c.box_depth > 0.0)))
        throw std::invalid_argument(
            "mlfmm_scaling: need L > 0, mesh_size > 0, sigma > 0, waist_factor > 0, "
            "box_depth > 0 (if set) and a fixed seed > 0");
    if (c.uncapped_box && c.box_mesh_size)
        throw std::invalid_argument(
            "mlfmm_scaling: uncapped_box needs the automatic rule (box_mesh_size unset)");
    check_beam_waist(c.L, c.L / waist_factor_of(c),
                     {.allow_wide = c.allow_wide_beam, .max_fraction = waist_fraction(c)});
    const RoughBoxParams b = box_params(c);
    geometry::RoughSurfaceParams rp;
    rp.edge_length_L = c.L;
    rp.rms_roughness = c.sigma;
    rp.correlation_length = kCorrelationLength;
    rp.mesh_size = c.mesh_size;
    rp.seed = c.seed;
    b.apply_to(rp);
    if (c.uncapped_box)
        rp.exterior_wavelength.reset();
    const geometry::HeightMap hm = geometry::generate_gaussian_height_map(rp);
    const Real depth = rp.box_depth.value();
    geometry::detail::BoxGrading g = geometry::detail::box_grading(
        hm, depth, rp.box_mesh_size, rp.box_fine_depth, rp.exterior_wavelength);
    geometry::TriangleMesh mesh = geometry::make_mesh_from_height_map(
        hm, depth, rp.box_mesh_size, rp.box_fine_depth, rp.exterior_wavelength);
    const Index top = 2 * (hm.z.rows() - 1) * (hm.z.cols() - 1);
    return Geometry{std::move(mesh), depth, rp.box_fine_depth, std::move(g), top};
}

/// Incident beam of a case on its geometry. Beam::angular_spectrum sets the controlled ball to
/// 1.01 x the largest vertex distance from the focus (the Simulation constructor rejects meshes
/// outside it; the plane-wave count grows as (R / w0)^2).
inline std::shared_ptr<excitation::Excitation> make_beam(const Case& c, const Geometry& geo) {
    if (c.beam == Beam::angular_spectrum) {
        excitation::AngularSpectrumBeam::Params ap;
        ap.wavelength = kLambda;
        ap.waist_radius = c.L / waist_factor_of(c);
        ap.focus = Vec3::Zero();
        ap.incidence_angle = 0.0;
        ap.polarization = excitation::Polarization::P;  // E along x at normal incidence
        ap.region_radius = 1.01 * geo.mesh.vertices().rowwise().norm().maxCoeff();
        return std::make_shared<excitation::AngularSpectrumBeam>(ap);
    }
    excitation::GaussianBeam::Params bp;
    bp.wavelength = kLambda;
    bp.waist_radius = c.L / waist_factor_of(c);
    bp.focus = Vec3::Zero();
    bp.incidence_angle = 0.0;
    bp.polarization = excitation::Polarization::P;  // E along x at normal incidence
    return std::make_shared<excitation::GaussianBeam>(bp);
}

inline SimulationConfig make_config(const Case& c) {
    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.exterior = material::vacuum();
    cfg.object = object_material(c.material);
    cfg.compression = "mlfmm";
    cfg.mlfmm.accuracy_digits = c.digits;
    cfg.mlfmm.max_exact_far_bytes = c.max_exact_far_bytes;
    cfg.mlfmm.leaf_radius_quantile = c.leaf_radius_quantile;
    cfg.formulation = c.formulation;
    cfg.diagonal_preconditioner = c.jacobi;
    cfg.gmres.tolerance = c.tolerance;
    cfg.gmres.max_iter = c.max_iter;
    cfg.gmres.restart = c.restart;
    cfg.gmres.verbose = true;
    return cfg;
}

/// Estimated number of ordered basis pairs of the exact far part of region R2 (ADR 0008 §6
/// fallback of a lossy interior): pairs whose support bounding boxes are closer than x*(d0) /
/// alpha (alpha = -Im k2, the decay bound delta > 10^-(d0+1) of far_operator.hpp), minus the near
/// pairs (estimate_near_bytes / 96 bytes). Assumes, as measured for Ag (WP21, WP22a), that the
/// expansion fails on the levels that hold these pairs; 0 for a region without decay strong
/// enough for the fallback (x* / alpha > 2 um: Si). A geometric count without assembly, O(N
/// candidates per basis).
inline Index estimate_exact_pairs(const basis::RwgSpace& space, Complex k2, Real digits,
                                  Index near_pairs) {
    const Real alpha = -k2.imag();
    if (!(alpha > 0.0))
        return 0;
    const Real reach = mlfmm::truncation_decay_exponent(digits) / alpha;
    if (reach > 2e-6)
        return 0;
    const geometry::TriangleMesh& mesh = space.mesh();
    const Index n = space.size();
    std::vector<Vec3> lo(static_cast<std::size_t>(n)), hi(static_cast<std::size_t>(n));
    Real extent = 0.0;
    for (Index b = 0; b < n; ++b) {
        const std::array<Index, 4> v = {mesh.edges()(b, 0), mesh.edges()(b, 1),
                                        space.plus_free_vertex(b), space.minus_free_vertex(b)};
        Vec3 l = mesh.vertices().row(v[0]).transpose();
        Vec3 h = l;
        for (const Index vi : v) {
            const Vec3 p = mesh.vertices().row(vi).transpose();
            l = l.cwiseMin(p);
            h = h.cwiseMax(p);
        }
        lo[static_cast<std::size_t>(b)] = l;
        hi[static_cast<std::size_t>(b)] = h;
        extent = std::max(extent, (h - l).maxCoeff());
    }
    // Uniform grid over the lower bbox corners with cell >= reach + extent: partners lie in the
    // 27 neighbouring cells.
    const Real cell = reach + extent;
    Vec3 origin = lo[0];
    for (const Vec3& l : lo) origin = origin.cwiseMin(l);
    const auto key = [&](const Vec3& p, int dx, int dy, int dz) {
        const auto ix = static_cast<std::int64_t>(std::floor((p.x() - origin.x()) / cell)) + dx;
        const auto iy = static_cast<std::int64_t>(std::floor((p.y() - origin.y()) / cell)) + dy;
        const auto iz = static_cast<std::int64_t>(std::floor((p.z() - origin.z()) / cell)) + dz;
        return (ix * 1048576 + iy) * 1048576 + iz;
    };
    std::unordered_map<std::int64_t, std::vector<Index>> grid;
    for (Index b = 0; b < n; ++b) grid[key(lo[static_cast<std::size_t>(b)], 0, 0, 0)].push_back(b);
    Index within = 0;  // ordered pairs (including m = n)
    for (Index b = 0; b < n; ++b) {
        const Vec3& lb = lo[static_cast<std::size_t>(b)];
        const Vec3& hb = hi[static_cast<std::size_t>(b)];
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dz = -1; dz <= 1; ++dz) {
                    const auto it = grid.find(key(lb, dx, dy, dz));
                    if (it == grid.end())
                        continue;
                    for (const Index o : it->second) {
                        const Vec3& lo2 = lo[static_cast<std::size_t>(o)];
                        const Vec3& hi2 = hi[static_cast<std::size_t>(o)];
                        const Vec3 gap = (lo2 - hb).cwiseMax(lb - hi2).cwiseMax(Vec3::Zero());
                        if (gap.norm() < reach)
                            ++within;
                    }
                }
    }
    return std::max<Index>(0, within - near_pairs);
}

/// Mesh, octree and memory estimates of a case, without assembly.
struct Estimate {
    Index triangles = 0;
    Index top_triangles = 0;
    Index unknowns = 0;    ///< 2N
    Index box_levels = 0;  ///< M of the graded box
    Index fine_rows = 0;
    Real depth = 0;
    Real fine_depth = 0;          ///< 0: no fine band
    Real max_support_radius = 0;  ///< r_max [m]
    int levels = 0;               ///< octree levels incl. the root (after the leaf rule)
    Real leaf_edge = 0;           ///< [m]
    Index leaves = 0;
    /// Basis functions per home level (local leaf rule; all on the leaf without elevated ones).
    std::vector<Index> home_counts;
    Index elevated = 0;           ///< basis functions above the leaf level
    std::size_t near_bytes = 0;   ///< mlfmm::estimate_near_bytes
    Index exact_pairs = 0;        ///< estimate_exact_pairs (ordered basis pairs of R2)
    std::size_t exact_bytes = 0;  ///< exact_pairs x 40 bytes
    std::size_t exact_budget = 0;
    /// Leaf sampling order per region from mlfmm::leaf_sampling (0: region without a usable
    /// leaf expansion, e.g. the Ag interior).
    std::array<int, 2> leaf_order{};
    /// Expansion levels per region (the leaf and the coarser levels whose order search is
    /// achievable, as in MlfmmFarOperator; the block check is not modelled).
    std::array<int, 2> expansion_levels{};
    /// Radiation patterns of both regions: the leaf-level functions x 2 (L + 1)^2 directions x 2
    /// components x 16 bytes per expanded region (the dominant part of the far memory), plus the
    /// elevated functions at the sampling of their home level (expansion levels only).
    std::size_t pattern_bytes = 0;
    std::size_t elevated_pattern_bytes = 0;  ///< the elevated part of pattern_bytes
    /// Far tables of the expansion levels of both regions (MlfmmFarOperator::memory_bytes
    /// without the patterns and the exact parts): translators (one per distinct interaction
    /// offset, directions x 16 bytes), one apply workspace (2 x 4 fields x boxes x directions x
    /// 16 bytes) and the phase shifts; interpolation tables are neglected (< 1 %).
    std::size_t far_table_bytes = 0;
    /// Peak working set model of the setup (operator built, before the solve):
    /// kPeakNearFactor near + 1.1 (patterns + far tables) + 2 exact + 1 GB.
    Real peak_setup_bytes = 0;
};

/// Near-field factor of the peak model. After d00317a (the near-field matrix is taken without a
/// copy) the measured peak after the setup was 1.0-1.3 x near + far (rough boxes, record
/// "Memory model"); 1.5 keeps the model conservative.
inline constexpr Real kPeakNearFactor = 1.5;

/// Far tables per region (Estimate::far_table_bytes, leaf patterns, expansion levels) of one
/// region with wavenumber k: the level loop of MlfmmFarOperator (leaf: leaf_sampling, coarser:
/// search_truncation_order with the enlarged diagonal; stop at the first level whose search is
/// not achievable or underflows), counting the translators of the distinct interaction offsets.
struct FarTableEstimate {
    int leaf_order = 0;                      ///< leaf sampling order (0: no leaf expansion)
    int levels = 0;                          ///< expansion levels
    std::size_t pattern_bytes = 0;           ///< leaf and elevated patterns
    std::size_t elevated_pattern_bytes = 0;  ///< elevated patterns (local leaf rule)
    std::size_t table_bytes = 0;
};

/// Largest support radius of the functions with home level >= l, per level (the radius in the
/// enlarged diagonal of MlfmmFarOperator; r_max on every level without elevated functions).
inline std::vector<Real> level_support_radii(const basis::RwgSpace& space,
                                             const mlfmm::Octree& tree) {
    const std::vector<Real> r = mlfmm::support_radii(space);
    std::vector<Real> out(static_cast<std::size_t>(tree.levels()), 0.0);
    for (Index p = 0; p < space.size(); ++p) {
        auto& v = out[static_cast<std::size_t>(tree.home_level(p))];
        v = std::max(v,
                     r[static_cast<std::size_t>(tree.permutation()[static_cast<std::size_t>(p)])]);
    }
    for (int l = tree.leaf_level() - 1; l >= 0; --l)
        out[static_cast<std::size_t>(l)] =
            std::max(out[static_cast<std::size_t>(l)], out[static_cast<std::size_t>(l + 1)]);
    return out;
}

inline FarTableEstimate estimate_far_tables(const basis::RwgSpace& space, const mlfmm::Octree& tree,
                                            Complex k, Real digits) {
    FarTableEstimate f;
    const std::vector<Real> rl = level_support_radii(space, tree);
    const int leaf = tree.leaf_level();
    for (int l = leaf; l >= 2; --l) {
        int sampling_order = 0;
        const Real rmax = rl[static_cast<std::size_t>(l)];
        try {
            if (l == leaf) {
                const mlfmm::LeafSampling ls = mlfmm::leaf_sampling(tree, k, digits, rmax);
                if (!ls.search.achievable)
                    break;
                sampling_order = ls.sampling_order;
            } else {
                mlfmm::TruncationSearchOptions opt;
                const Real a = tree.box_size(l);
                opt.box_diagonal = std::sqrt(3.0) * a + 2.0 * rmax;
                const mlfmm::TruncationSearch s = mlfmm::search_truncation_order(k, a, digits, opt);
                if (!s.achievable)
                    break;
                sampling_order = s.order;
            }
        } catch (const std::underflow_error&) {
            break;  // as MlfmmFarOperator: the interaction underflows, no expansion
        }
        const auto dirs = static_cast<std::size_t>(mlfmm::SphereSampling(sampling_order).size());
        const std::vector<Index>& boxes = tree.boxes_at_level(l);
        std::vector<std::array<Index, 3>> offsets;
        for (const Index ia : boxes) {
            const mlfmm::Box& A = tree.boxes()[static_cast<std::size_t>(ia)];
            for (const Index ib : A.interaction_list) {
                const mlfmm::Box& B = tree.boxes()[static_cast<std::size_t>(ib)];
                const std::array<Index, 3> d = {A.ijk[0] - B.ijk[0], A.ijk[1] - B.ijk[1],
                                                A.ijk[2] - B.ijk[2]};
                if (std::find(offsets.begin(), offsets.end(), d) == offsets.end())
                    offsets.push_back(d);
            }
        }
        f.table_bytes += offsets.size() * dirs * sizeof(Complex);        // translators
        f.table_bytes += 2 * 4 * boxes.size() * dirs * sizeof(Complex);  // apply workspace
        if (l == leaf) {
            f.leaf_order = sampling_order;
            const std::vector<Index> counts = tree.home_level_counts();
            f.pattern_bytes = static_cast<std::size_t>(counts.back()) * dirs * 2 * sizeof(Complex);
        } else {
            f.table_bytes += 16 * dirs * sizeof(Complex);  // phase shifts at the parent
            const std::size_t elevated =
                tree.elevated_positions(l).size() * dirs * 2 * sizeof(Complex);
            f.pattern_bytes += elevated;
            f.elevated_pattern_bytes += elevated;
        }
        ++f.levels;
    }
    return f;
}

/// Result of one solved case.
struct Result {
    Estimate est;
    std::string formulation;
    /// to_string(Simulation::preconditioner_kind()) of the solve ("none", "diagonal" = left
    /// Jacobi, "block_diagonal").
    std::string preconditioner = "none";
    int iterations = 0;
    bool converged = false;
    Real final_residual = 0;  ///< monitored (preconditioned) residual
    Real true_residual = 0;   ///< |b - Z x| / |b|
    Real assembly_s = 0;      ///< Simulation::assemble (octree, far, near, rhs, Jacobi diagonal)
    Real tree_s = 0, far_s = 0, near_s = 0;  ///< parts of the operator construction
    Real exact_s = 0;                        ///< exact parts (included in far_s)
    Real matvec_s = 0;                       ///< median MlfmmOperator::apply
    Real near_matvec_s = 0, far_matvec_s = 0;
    Real solve_s = 0;  ///< GMRES wall time
    std::size_t near_bytes = 0;
    std::size_t near_nnz = 0;
    std::array<std::size_t, 2> exact_bytes{};  ///< per region (R1, R2)
    std::array<Index, 2> exact_pairs{};
    std::size_t far_bytes = 0;  ///< far operator without the exact parts (tables, workspaces)
    std::size_t operator_bytes = 0;
    Real krylov_bytes = 0;  ///< (iterations + 1) 2N 16 bytes (full GMRES)
    /// GMRES monitored residuals r_0 = 1, r_1 .. r_iterations (solver::GmresResult; empty
    /// without a solve); history_record writes them (WP22b2).
    std::vector<Real> residual_history;
    /// Power summary (Case::far_field_dtheta_deg > 0, else all 0): incident beam power
    /// AngularSpectrumBeam::power(), reflected power on the primary hemisphere grid and on the
    /// grid with half the nodes per direction (convergence check), absorbed power from the
    /// currents [W], and the wall time of this post-processing [s].
    Real p_inc = 0, p_refl = 0, p_refl_coarse = 0, p_abs = 0, post_s = 0;
    /// Non-empty if the power summary threw after the solve (run_power_summary): the message;
    /// the p_* fields are then 0 and the ROW line carries post_error=1 instead of R / A.
    std::string post_error;
    Real peak_setup_rss = 0;  ///< peak working set after Simulation::assemble [bytes]
    Real peak_rss = 0;        ///< at the end [bytes]
    std::string describe;     ///< MlfmmOperator::describe()
    std::string report;       ///< Simulation::report()
};

/// Estimate for a built geometry (no assembly and no excitation; seconds to a minute: the order
/// searches of the coarse Si levels dominate).
inline Estimate estimate(const Geometry& geo, const Case& c) {
    const SimulationConfig cfg = make_config(c);
    const basis::RwgSpace space(geo.mesh);
    Estimate e;
    e.triangles = geo.mesh.num_triangles();
    e.top_triangles = geo.top_triangles;
    e.unknowns = 2 * space.size();
    e.box_levels = geo.grading.levels;
    e.fine_rows = geo.grading.fine_rows;
    e.depth = geo.depth;
    e.fine_depth = geo.fine_depth.value_or(0.0);
    e.max_support_radius = mlfmm::max_support_radius(space);
    const mlfmm::Octree tree = mlfmm::make_octree(space, kLambda, cfg.mlfmm);
    e.levels = tree.levels();
    e.leaf_edge = tree.box_size(tree.leaf_level());
    e.leaves = static_cast<Index>(tree.boxes_at_level(tree.leaf_level()).size());
    e.home_counts = tree.home_level_counts();
    e.elevated = space.size() - e.home_counts.back();
    e.near_bytes = mlfmm::estimate_near_bytes(tree);
    const auto near_pairs = static_cast<Index>(e.near_bytes / 96);
    const Real omega = 2.0 * constants::pi * constants::c0 / kLambda;
    const Complex k2 = cfg.object.wavenumber(omega);
    e.exact_pairs = estimate_exact_pairs(space, k2, c.digits, near_pairs);
    e.exact_bytes = static_cast<std::size_t>(e.exact_pairs) * 40;
    e.exact_budget =
        c.max_exact_far_bytes > 0
            ? c.max_exact_far_bytes
            : std::min(std::max(static_cast<std::size_t>(mlfmm::kExactFarNearFactor *
                                                         static_cast<Real>(e.near_bytes)),
                                mlfmm::kExactFarMinBytes),
                       std::size_t{16} * static_cast<std::size_t>(e.unknowns) *
                           static_cast<std::size_t>(e.unknowns));
    const std::array<Complex, 2> k = {cfg.exterior.wavenumber(omega), k2};
    for (std::size_t i = 0; i < 2; ++i) {
        // A region whose leaf order search fails (the Ag interior: exact / truncated far part)
        // has no patterns or tables. (Si with lambda/0.7 leaves has alpha D = 1.5 and still
        // expands.)
        const FarTableEstimate f = estimate_far_tables(space, tree, k[i], c.digits);
        e.leaf_order[i] = f.leaf_order;
        e.expansion_levels[i] = f.levels;
        e.pattern_bytes += f.pattern_bytes;
        e.elevated_pattern_bytes += f.elevated_pattern_bytes;
        e.far_table_bytes += f.table_bytes;
    }
    e.peak_setup_bytes = kPeakNearFactor * static_cast<Real>(e.near_bytes) +
                         1.1 * static_cast<Real>(e.pattern_bytes + e.far_table_bytes) +
                         2.0 * static_cast<Real>(e.exact_bytes) + 1e9;
    return e;
}

/// Number following `marker` in `text` (the parts of MlfmmOperator::describe()).
/// @throws std::runtime_error if the marker is missing or not followed by a number (never a
///         silent 0 in the recorded rows).
inline Real number_after(const std::string& text, const std::string& marker) {
    const auto pos = text.find(marker);
    if (pos == std::string::npos)
        throw std::runtime_error("mlfmm_scaling::number_after: marker \"" + marker +
                                 "\" not found (MlfmmOperator::describe() format changed?)");
    std::istringstream is(text.substr(pos + marker.size()));
    Real v = 0.0;
    is >> v;
    if (!is)
        throw std::runtime_error("mlfmm_scaling::number_after: no number after \"" + marker + "\"");
    return v;
}

/// Wall time of one call of f [s].
inline Real seconds_of(const std::function<void()>& f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
}

/// Median of a non-empty sample.
inline Real median(std::vector<Real> t) {
    if (t.empty())
        throw std::invalid_argument("mlfmm_scaling::median: empty sample");
    std::sort(t.begin(), t.end());
    const std::size_t m = t.size() / 2;
    return t.size() % 2 == 1 ? t[m] : 0.5 * (t[m - 1] + t[m]);
}

/// Called once per GMRES iteration with (k, monitored residual, seconds since the solve
/// started).
using IterationLog = std::function<void(int iter, Real residual, Real seconds)>;

/// Power summary of a solved case (Result::p_*): P_inc = AngularSpectrumBeam::power(), P_refl
/// = (1 / 2 eta1) int |F|^2 dOmega of the scattered far field over the reflection hemisphere
/// k_z < 0 (Gauss-Legendre in theta x trapezoid in phi, fresnel_flat::hemisphere_grid, polar
/// spacing dtheta_deg; also on the grid with half the nodes per direction), P_abs = 1/2 Re
/// oint (n x M) . J* dS from the currents (fresnel_flat::absorbed_power). For a closed box under
/// a beam that the top face intercepts, 1 - R - A is the power scattered into the forward
/// hemisphere around the box plus the discretisation and solver error.
/// @throws std::invalid_argument for a non-positive spacing or degree, or a beam that is not an
///         AngularSpectrumBeam (no exact P_inc).
inline void reflected_power(const Simulation& sim, const excitation::Excitation& beam,
                            Real dtheta_deg, int degree, Result& r) {
    if (!(dtheta_deg > 0.0) || degree < 1)
        throw std::invalid_argument("mlfmm_scaling::reflected_power: need dtheta > 0, degree >= 1");
    const auto* as = dynamic_cast<const excitation::AngularSpectrumBeam*>(&beam);
    if (as == nullptr)
        throw std::invalid_argument(
            "mlfmm_scaling::reflected_power: needs the angular-spectrum beam (exact P_inc)");
    const auto t0 = std::chrono::steady_clock::now();
    const Real omega = 2.0 * constants::pi * constants::c0 / kLambda;
    const Real eta1 = material::vacuum().wave_impedance(omega).real();
    const Vec3 specular(0.0, 0.0, -1.0);  // normal incidence
    const Real cone = 3.0 * kLambda / (constants::pi * as->params().waist_radius);
    r.p_inc = as->power();
    r.p_refl = fresnel_flat::hemisphere_power(sim.solution(),
                                              fresnel_flat::hemisphere_grid(-1.0, dtheta_deg),
                                              degree, eta1, specular, cone)
                   .total;
    r.p_refl_coarse = fresnel_flat::hemisphere_power(
                          sim.solution(), fresnel_flat::hemisphere_grid(-1.0, dtheta_deg, 2),
                          degree, eta1, specular, cone)
                          .total;
    r.p_abs = fresnel_flat::absorbed_power(sim.solution());
    r.post_s = std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
}

/// Smallest polar spacing of the power summary [deg] (n_phi = 180 000 on the primary grid).
inline constexpr Real kMinFarFieldSpacingDeg = 1e-3;

/// Checks the power-summary options of a case (Case::far_field_dtheta_deg,
/// Case::far_field_degree, Case::beam) before any assembly, so that a bad option fails in
/// seconds instead of after a multi-hour solve. No-op for far_field_dtheta_deg == 0 (off). Both
/// hemisphere grids of reflected_power (fresnel_flat::hemisphere_grid: n_theta =
/// round(90 / (c dtheta)) >= 1, n_phi = round(180 / (c dtheta)) >= 4 for c = 1 and the half-node
/// grid c = 2) must be valid, i.e. dtheta <= 25.7 deg, and the source quadrature must be a
/// positive-interior Dunavant rule (post::FieldOptions::quad_degree: 1, 2, 4, 5, 6, 8, 9, 10,
/// ...).
/// @throws std::invalid_argument for a negative or non-finite spacing, a spacing below
///         kMinFarFieldSpacingDeg or too coarse for either grid, a degree outside 1..20 or not
///         positive-interior, or a power summary without the angular-spectrum beam.
inline void validate_power_summary(const Case& c) {
    const Real d = c.far_field_dtheta_deg;
    if (!std::isfinite(d) || d < 0.0)
        throw std::invalid_argument(
            "mlfmm_scaling: the far-field spacing must be finite and >= 0 (0 = off)");
    if (d == 0.0)
        return;
    if (d < kMinFarFieldSpacingDeg)
        throw std::invalid_argument("mlfmm_scaling: far-field spacing below 1e-3 deg");
    for (const int coarsening : {1, 2}) {
        const long long nt = std::llround(90.0 / (d * coarsening));
        const long long np = std::llround(180.0 / (d * coarsening));
        if (nt < 1 || np < 4)
            throw std::invalid_argument(
                "mlfmm_scaling: far-field spacing " + std::to_string(d) + " deg too coarse: the " +
                (coarsening == 1 ? "primary" : "half-node") +
                " hemisphere grid needs n_theta >= 1 and n_phi >= 4 (spacing <= 25.7 deg)");
    }
    if (c.far_field_degree < 1 || c.far_field_degree > 20 ||
        !kernels::triangle_rule_is_positive_interior(c.far_field_degree))
        throw std::invalid_argument(
            "mlfmm_scaling: far-field degree " + std::to_string(c.far_field_degree) +
            " is not a positive-interior Dunavant rule (1, 2, 4, 5, 6, 8, 9, 10, ...)");
    if (c.beam != Beam::angular_spectrum)
        throw std::invalid_argument(
            "mlfmm_scaling: the power summary needs the angular-spectrum beam (exact P_inc)");
}

/// Runs the power summary `post` (reflected_power) after a finished solve. An exception there
/// must not lose the solve: it is logged with SBEM_ERROR, the power fields are reset to 0 (the
/// ROW line then has no R / A) and Result::post_error keeps the message (ROW annotation
/// post_error=1).
inline void run_power_summary(const std::function<void()>& post, Result& r) {
    const auto fail = [&r](const std::string& what) {
        r.p_inc = r.p_refl = r.p_refl_coarse = r.p_abs = r.post_s = 0.0;
        r.post_error = what.empty() ? std::string("unknown error") : what;
        SBEM_ERROR(
            "mlfmm_scaling: power summary failed after the solve ({}); the solve result is "
            "kept, R and A are omitted",
            r.post_error);
    };
    try {
        post();
    } catch (const std::exception& e) {
        fail(e.what());
    } catch (...) {
        fail("non-standard exception");
    }
}

/// Assembles and solves the case through Simulation (compression "mlfmm"), timing the matvec.
/// `est` is the case's estimate(geo, c) (computed once by the caller); `on_iteration`
/// (optional) is called once per GMRES iteration (the residual / timestamp history of the
/// server run kit, printed live by the benchmark).
inline Result run(const Geometry& geo, const Case& c, const Estimate& est,
                  const IterationLog& on_iteration = {}) {
    using Clock = std::chrono::steady_clock;
    Result r;
    r.est = est;
    validate_power_summary(c);  // before the assembly: a bad option must not cost a solve
    const std::shared_ptr<excitation::Excitation> beam = make_beam(c, geo);
    Simulation sim(geo.mesh, beam, make_config(c));
    r.formulation = formulation::make_formulation(sim.formulation_kind())->name();
    r.preconditioner = to_string(sim.preconditioner_kind());
    {
        const auto t0 = Clock::now();
        sim.assemble();
        r.assembly_s = std::chrono::duration<Real>(Clock::now() - t0).count();
    }
    r.peak_setup_rss = system_memory::peak_rss_bytes();
    const auto op = std::dynamic_pointer_cast<const mlfmm::MlfmmOperator>(sim.system_operator());
    if (op == nullptr)
        throw std::logic_error("mlfmm_scaling::run: the system operator is not an MLFMM");
    r.describe = op->describe();
    r.tree_s = number_after(r.describe, "built in ");
    r.near_s = number_after(r.describe, "assembled in ");
    r.far_s = number_after(r.describe, "\n  far: ");
    r.near_bytes = op->near_operator().memory_bytes();
    r.near_nnz = static_cast<std::size_t>(op->near_operator().nonzeros());
    std::size_t exact_total = 0;
    for (int region = 0; region < 2; ++region) {
        const mlfmm::ExactPartInfo& xi = op->far_operator().exact_info(region);
        const auto i = static_cast<std::size_t>(region);
        r.exact_bytes[i] = xi.bytes;
        r.exact_pairs[i] = xi.pairs;
        r.exact_s += xi.seconds;
        exact_total += xi.bytes;
    }
    {
        // Timed applies on a fixed random vector.
        std::mt19937_64 rng(12345);
        std::normal_distribution<Real> nd;
        VectorXc x(sim.num_unknowns());
        for (Index i = 0; i < x.size(); ++i) x(i) = Complex(nd(rng), nd(rng));
        VectorXc y(x.size());
        // Interleaved (full, near, far per round, after one untimed round): transients of the
        // machine (other processes, memory released after the setup) hit all three alike.
        const std::array<std::function<void()>, 3> calls = {
            [&] { op->apply(x, y); }, [&] { op->near_operator().apply(x, y); },
            [&] { op->far_operator().apply(x, y); }};
        std::array<std::vector<Real>, 3> t;
        for (int round = 0; round <= std::max(1, c.matvecs); ++round)
            for (std::size_t k = 0; k < 3; ++k) {
                const Real s = seconds_of(calls[k]);
                if (round > 0)
                    t[k].push_back(s);
            }
        r.matvec_s = median(t[0]);
        r.near_matvec_s = median(t[1]);
        r.far_matvec_s = median(t[2]);
    }
    if (c.solve) {
        const auto solve_start = Clock::now();
        solver::IterationCallback cb;
        if (on_iteration)
            cb = [&](int k, Real res) {
                on_iteration(k, res,
                             std::chrono::duration<Real>(Clock::now() - solve_start).count());
            };
        const solver::GmresResult g = sim.solve(cb);
        r.iterations = g.iterations;
        r.converged = g.converged;
        r.final_residual = g.residual_history.empty() ? 0.0 : g.residual_history.back();
        r.residual_history = g.residual_history;
        r.true_residual = g.true_relative_residual;
        r.solve_s = g.wall_seconds;
        const int basis = c.restart > 0 ? std::min(g.iterations, c.restart) : g.iterations;
        r.krylov_bytes = static_cast<Real>(basis + 1) * static_cast<Real>(sim.num_unknowns()) *
                         static_cast<Real>(sizeof(Complex));
        r.report = sim.report();
        if (c.far_field_dtheta_deg > 0.0)
            run_power_summary(
                [&] { reflected_power(sim, *beam, c.far_field_dtheta_deg, c.far_field_degree, r); },
                r);
    }
    r.far_bytes = op->far_operator().memory_bytes() - exact_total;
    r.operator_bytes = op->memory_bytes();
    r.peak_rss = system_memory::peak_rss_bytes();
    return r;
}

/// Box option of a case: the coarse spacing in nm, "auto" (automatic rule capped at
/// lambda1 / 5) or "uncapped" (automatic rule without the cap).
inline std::string box_option(const Case& c) {
    if (c.box_mesh_size)
        return std::to_string(*c.box_mesh_size * 1e9);
    return c.uncapped_box ? "uncapped" : "auto";
}

inline std::string beam_name(const Case& c) {
    return c.beam == Beam::angular_spectrum ? "angular_spectrum" : "paraxial";
}

/// Case label, e.g. "si L = 4 um, h = 50 nm, box 100 nm, fine band, paraxial beam w0 = L/4, ...".
inline std::string label(const Case& c) {
    std::ostringstream os;
    os << c.material << " L = " << c.L * 1e6 << " um, sigma = " << c.sigma * 1e9
       << " nm, h = " << c.mesh_size * 1e9 << " nm, box "
       << (c.box_mesh_size
               ? std::to_string(static_cast<int>(std::lround(*c.box_mesh_size * 1e9))) + " nm"
               : box_option(c))
       << (uses_fine_band(c) ? ", fine band" : ", no fine band")
       << (c.box_depth ? ", depth " + std::to_string(*c.box_depth * 1e6) + " um" : std::string())
       << ", " << beam_name(c) << " beam w0 = L/" << waist_factor_of(c) << ", seed " << c.seed
       << ", d0 = " << c.digits << ", leaf radius quantile " << c.leaf_radius_quantile
       << ", tol = " << c.tolerance;
    return os.str();
}

inline Real gb(Real bytes) {
    return bytes / 1e9;
}
inline Real gb(std::size_t bytes) {
    return static_cast<Real>(bytes) / 1e9;
}

/// Multi-line summary of an estimate.
inline std::string summary(const Estimate& e) {
    std::ostringstream os;
    os << "triangles " << e.triangles << " (top " << e.top_triangles << ", box "
       << e.triangles - e.top_triangles << " = "
       << 100.0 * static_cast<Real>(e.triangles - e.top_triangles) /
              static_cast<Real>(e.top_triangles)
       << " %), box M = " << e.box_levels << ", fine rows " << e.fine_rows << ", depth "
       << e.depth * 1e6 << " um, fine band " << e.fine_depth * 1e6 << " um; 2N = " << e.unknowns
       << "; r_max " << e.max_support_radius * 1e9 << " nm; octree " << e.levels
       << " levels, leaf edge " << e.leaf_edge * 1e9 << " nm (lambda / " << kLambda / e.leaf_edge
       << "), " << e.leaves << " leaves; " << e.elevated << " elevated functions (home levels";
    for (std::size_t l = 0; l < e.home_counts.size(); ++l) {
        if (e.home_counts[l] > 0)
            os << " " << l << ": " << e.home_counts[l];
    }
    os << "); near estimate " << gb(e.near_bytes) << " GB, exact R2 estimate " << gb(e.exact_bytes)
       << " GB (" << e.exact_pairs << " pairs), budget " << gb(e.exact_budget)
       << " GB; leaf orders " << e.leaf_order[0] << " / " << e.leaf_order[1]
       << " (expansion levels " << e.expansion_levels[0] << " / " << e.expansion_levels[1]
       << "), patterns " << gb(e.pattern_bytes) << " GB (elevated " << gb(e.elevated_pattern_bytes)
       << " GB), far tables " << gb(e.far_table_bytes) << " GB; setup peak estimate "
       << gb(e.peak_setup_bytes) << " GB";
    return os.str();
}

/// Multi-line summary of a result.
inline std::string summary(const Result& r) {
    std::ostringstream os;
    os << summary(r.est) << "\n"
       << "formulation " << r.formulation << fresnel_flat::preconditioner_suffix(r.preconditioner)
       << "; GMRES " << r.iterations << " iterations, converged " << (r.converged ? "yes" : "no")
       << ", monitored residual " << r.final_residual << ", true residual " << r.true_residual
       << "\nsetup " << r.assembly_s << " s (octree " << r.tree_s << " s, far " << r.far_s
       << " s incl. exact parts " << r.exact_s << " s, near " << r.near_s << " s); matvec "
       << r.matvec_s << " s (near " << r.near_matvec_s << " s, far " << r.far_matvec_s
       << " s); solve " << r.solve_s << " s (" << r.solve_s / std::max(1, r.iterations)
       << " s/it)\nmemory: near " << gb(r.near_bytes) << " GB (nnz " << r.near_nnz << ", "
       << static_cast<Real>(r.near_nnz) / static_cast<Real>(r.est.unknowns)
       << " per row), exact R1 " << gb(r.exact_bytes[0]) << " GB, exact R2 " << gb(r.exact_bytes[1])
       << " GB (" << r.exact_pairs[1] << " pairs), far " << gb(r.far_bytes) << " GB, operator "
       << gb(r.operator_bytes) << " GB, Krylov " << gb(r.krylov_bytes) << " GB, peak after setup "
       << gb(r.peak_setup_rss) << " GB, peak " << gb(r.peak_rss) << " GB";
    if (r.p_inc > 0.0)
        os << "\npower: P_inc " << r.p_inc << " W, R = P_refl / P_inc " << r.p_refl / r.p_inc
           << " (half the nodes: " << r.p_refl_coarse / r.p_inc << "), A = P_abs / P_inc "
           << r.p_abs / r.p_inc << ", 1 - R - A " << 1.0 - (r.p_refl + r.p_abs) / r.p_inc
           << "; post-processing " << r.post_s << " s";
    if (!r.post_error.empty())
        os << "\npower: post-processing failed after the solve: " << r.post_error;
    return os.str();
}

/// The ROW / HISTORY key jacobi= (WP22b2): 1 for the left Jacobi preconditioner, else 0; the
/// solve's preconditioner itself is the key precond= (Result::preconditioner).
inline int jacobi_key(const Result& r) {
    return r.preconditioner == "diagonal" ? 1 : 0;
}

/// Lower-case formulation name of a result ("pmchwt", "ictf", "mctf", "jmcfie"; the ROW and
/// HISTORY key form=); "unknown" before run() set it.
inline std::string formulation_key(const Result& r) {
    std::string f = r.formulation.empty() ? std::string("unknown") : r.formulation;
    std::transform(f.begin(), f.end(), f.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return f;
}

/// One machine-readable line (key=value, SI seconds, GB), parsed by
/// benchmarks/mlfmm_scaling_fit.py. Ends with form= (formulation_key), jacobi=0|1 (WP22b2; 1 for
/// the left Jacobi preconditioner, kept for the readers and the older rows) and precond=
/// (Result::preconditioner, WP-PC1 review) before the optional power-summary keys.
inline std::string row(const Case& c, const Result& r) {
    std::ostringstream os;
    os.precision(6);
    os << "ROW material=" << c.material << " L_um=" << c.L * 1e6 << " box=" << box_option(c)
       << " fine=" << (uses_fine_band(c) ? 1 : 0);
    if (c.box_depth)
        os << " depth_um=" << *c.box_depth * 1e6;
    os << " restart=" << c.restart << " h_nm=" << c.mesh_size * 1e9 << " sigma_nm=" << c.sigma * 1e9
       << " waist=" << waist_factor_of(c) << " beam=" << beam_name(c) << " d0=" << c.digits
       << " tol=" << c.tolerance << " lq=" << c.leaf_radius_quantile << " seed=" << c.seed
       << " solved=" << (c.solve ? 1 : 0) << " N2=" << r.est.unknowns << " levels=" << r.est.levels
       << " leaf_nm=" << r.est.leaf_edge * 1e9 << " rmax_nm=" << r.est.max_support_radius * 1e9
       << " elevated=" << r.est.elevated << " setup_s=" << r.assembly_s << " tree_s=" << r.tree_s
       << " far_s=" << r.far_s << " exact_s=" << r.exact_s << " near_s=" << r.near_s
       << " matvec_s=" << r.matvec_s << " near_mv_s=" << r.near_matvec_s
       << " far_mv_s=" << r.far_matvec_s << " it=" << r.iterations
       << " conv=" << (r.converged ? 1 : 0) << " res=" << r.final_residual
       << " true_res=" << r.true_residual << " solve_s=" << r.solve_s
       << " near_gb=" << gb(r.near_bytes) << " exact_gb=" << gb(r.exact_bytes[0] + r.exact_bytes[1])
       << " exact_est_gb=" << gb(r.est.exact_bytes) << " pattern_est_gb=" << gb(r.est.pattern_bytes)
       << " far_est_gb=" << gb(r.est.pattern_bytes + r.est.far_table_bytes)
       << " peak_est_gb=" << gb(r.est.peak_setup_bytes) << " far_gb=" << gb(r.far_bytes)
       << " krylov_gb=" << gb(r.krylov_bytes) << " peak_setup_gb=" << gb(r.peak_setup_rss)
       << " peak_gb=" << gb(r.peak_rss) << " form=" << formulation_key(r)
       << " jacobi=" << jacobi_key(r) << " precond=" << r.preconditioner;
    if (r.p_inc > 0.0)
        os << " R=" << r.p_refl / r.p_inc << " R_coarse=" << r.p_refl_coarse / r.p_inc
           << " A=" << r.p_abs / r.p_inc << " post_s=" << r.post_s;
    if (!r.post_error.empty())
        os << " post_error=1";
    return os.str();
}

/// Iterations k at which history_record samples the residual history of `iterations` GMRES
/// iterations: 0, stride, 2 stride, ... and always the last iteration (WP22b2: compact
/// histories every 10th iteration). Empty for iterations < 0.
/// @throws std::invalid_argument for stride < 1.
inline std::vector<int> history_indices(int iterations, int stride) {
    if (stride < 1)
        throw std::invalid_argument("mlfmm_scaling::history_indices: stride must be >= 1");
    std::vector<int> k;
    if (iterations < 0)
        return k;
    for (int i = 0; i <= iterations; i += stride) k.push_back(i);
    if (k.back() != iterations)
        k.push_back(iterations);
    return k;
}

/// One line "HISTORY material=.. L_um=.. box=.. fine=.. waist=.. beam=.. d0=.. tol=.. max_iter=..
/// seed=.. N2=.. form=.. jacobi=0|1 precond=.. it=.. conv=0|1 stride=.. res=r_0,r_s,...,r_it" of a
/// solved case: the monitored residuals at history_indices(it, stride) with `digits` significant
/// digits (WP22b2, benchmarks/results/formulation_convergence_mlfmm_histories.txt; stride 1 and
/// 17 digits give the full-precision history of the Jacobi identity check). waist=
/// (waist_factor_of) and max_iter= were added in the WP22b2 review round; the 40 original records
/// lack them (all at waist 4, see their ROW lines).
/// @throws std::invalid_argument for stride < 1, digits outside 1..17, or a result whose history
///         does not hold iterations + 1 values.
inline std::string history_record(const Case& c, const Result& r, int stride, int digits) {
    if (digits < 1 || digits > 17)
        throw std::invalid_argument("mlfmm_scaling::history_record: digits must be in 1..17");
    if (r.iterations < 0 || r.residual_history.size() != static_cast<std::size_t>(r.iterations) + 1)
        throw std::invalid_argument(
            "mlfmm_scaling::history_record: the residual history must hold iterations + 1 values");
    const std::vector<int> ks = history_indices(r.iterations, stride);
    std::ostringstream os;
    os.precision(6);
    os << "HISTORY material=" << c.material << " L_um=" << c.L * 1e6 << " box=" << box_option(c)
       << " fine=" << (uses_fine_band(c) ? 1 : 0) << " waist=" << waist_factor_of(c)
       << " beam=" << beam_name(c) << " d0=" << c.digits << " tol=" << c.tolerance
       << " max_iter=" << c.max_iter << " seed=" << c.seed << " N2=" << r.est.unknowns
       << " form=" << formulation_key(r) << " jacobi=" << jacobi_key(r)
       << " precond=" << r.preconditioner << " it=" << r.iterations
       << " conv=" << (r.converged ? 1 : 0) << " stride=" << stride << " res=";
    std::array<char, 40> buf{};
    for (std::size_t i = 0; i < ks.size(); ++i) {
        std::snprintf(buf.data(), buf.size(), "%.*e", digits - 1,
                      r.residual_history[static_cast<std::size_t>(ks[i])]);
        os << (i == 0 ? "" : ",") << buf.data();
    }
    return os.str();
}

/// Appends `line` and a newline to the text file `path` (UTF-8 narrow path, ADR 0007).
/// @throws std::runtime_error if the file cannot be opened or written.
inline void append_line(const std::string& path, const std::string& line) {
    std::ofstream f(core::path_from_utf8(path), std::ios::app | std::ios::binary);
    if (!f)
        throw std::runtime_error("mlfmm_scaling: cannot open " + path + " for appending");
    f << line << '\n';
    f.flush();
    if (!f)
        throw std::runtime_error("mlfmm_scaling: cannot write " + path);
}

namespace detail {

/// Converts the whole of `v` (the value of option `opt`) with `conv` (std::stod-like, taking
/// the position out-parameter): a value with trailing characters ("5x"), no number or out of
/// range is refused, never truncated to its numeric prefix.
/// @throws std::invalid_argument naming the option and the value.
template <typename F>
auto parse_number(const std::string& opt, const std::string& v, F conv)
    -> decltype(conv(v, nullptr)) {
    if (!v.empty() && std::isspace(static_cast<unsigned char>(v.front())))
        throw std::invalid_argument(opt + ": leading whitespace in '" + v + "'");
    std::size_t pos = 0;
    decltype(conv(v, nullptr)) x{};
    try {
        x = conv(v, &pos);
    } catch (const std::out_of_range&) {
        throw std::invalid_argument(opt + ": value out of range: '" + v + "'");
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument(opt + ": not a number: '" + v + "'");
    }
    if (pos != v.size())
        throw std::invalid_argument(opt + ": trailing characters in '" + v + "'");
    return x;
}

/// Also refuses non-finite values ("inf", "nan").
inline double to_double(const std::string& opt, const std::string& v) {
    const double x =
        parse_number(opt, v, [](const std::string& s, std::size_t* p) { return std::stod(s, p); });
    if (!std::isfinite(x))
        throw std::invalid_argument(opt + ": not finite: '" + v + "'");
    return x;
}

inline int to_int(const std::string& opt, const std::string& v) {
    return parse_number(opt, v,
                        [](const std::string& s, std::size_t* p) { return std::stoi(s, p); });
}

/// Unsigned conversion that also refuses a sign (std::stoull wraps "-1" around).
inline std::uint64_t to_uint64(const std::string& opt, const std::string& v) {
    if (v.empty() || !std::isdigit(static_cast<unsigned char>(v.front())))
        throw std::invalid_argument(opt + ": not an unsigned integer: '" + v + "'");
    return parse_number(opt, v, [](const std::string& s, std::size_t* p) {
        return static_cast<std::uint64_t>(std::stoull(s, p));
    });
}

}  // namespace detail

/// Command-line options of benchmarks/mlfmm_scaling.cpp (parse_options).
struct Options {
    Case c;
    bool estimate_only = false;
    bool force = false;
    bool history = false;  ///< --history: one HIST line per GMRES iteration on stdout
    /// --history-file: append history_record(c, r, history_stride, history_digits) after the
    /// solve (WP22b2); empty: off.
    std::string history_file;
    int history_stride = 10;  ///< --history-stride (>= 1)
    int history_digits = 4;   ///< --history-digits (1..17)
    /// --history-full-file: append the full-precision record (stride 1, 17 digits); empty: off.
    std::string history_full_file;
};

/// Parses the benchmark's arguments (argv[1..], without the program name). Defaults: those of
/// Case and Options (formulation and Jacobi per formulation::recommend unless --formulation /
/// --jacobi are given).
/// Numeric values must be consumed whole ("5x", "1e999" and "-1" for --seed are refused, never
/// truncated or wrapped). Writability of the history files is checked separately by
/// probe_history_files (parsing has no side effects on the file system).
/// @throws std::invalid_argument for an unknown option, a missing or invalid value (including
///         a numeric value with trailing characters or out of range), or inconsistent options
///         (validate_power_summary; --history-file / --history-full-file with --no-solve or
///         --estimate-only).
inline Options parse_options(const std::vector<std::string>& args) {
    Options o;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= args.size())
                throw std::invalid_argument("missing value after " + a);
            return args[++i];
        };
        if (a == "--material") {
            o.c.material = value();
        } else if (a == "--L") {
            o.c.L = detail::to_double(a, value());
        } else if (a == "--box-mesh-size") {
            const std::string v = value();
            o.c.uncapped_box = false;
            if (v == "auto") {
                o.c.box_mesh_size.reset();
            } else if (v == "uncapped") {
                o.c.box_mesh_size.reset();
                o.c.uncapped_box = true;
            } else {
                o.c.box_mesh_size = detail::to_double(a, v);
            }
        } else if (a == "--fine-band") {
            o.c.fine_band = detail::to_int(a, value()) != 0;
        } else if (a == "--sigma") {
            o.c.sigma = detail::to_double(a, value());
        } else if (a == "--history") {
            o.history = true;
        } else if (a == "--history-file") {
            o.history_file = value();
        } else if (a == "--history-full-file") {
            o.history_full_file = value();
        } else if (a == "--history-stride") {
            o.history_stride = detail::to_int(a, value());
        } else if (a == "--history-digits") {
            o.history_digits = detail::to_int(a, value());
        } else if (a == "--far-field") {
            o.c.far_field_dtheta_deg = detail::to_double(a, value());  // 0 = off; checked below
        } else if (a == "--far-field-degree") {
            o.c.far_field_degree = detail::to_int(a, value());
        } else if (a == "--mesh-size") {
            o.c.mesh_size = detail::to_double(a, value());
        } else if (a == "--seed") {
            o.c.seed = detail::to_uint64(a, value());
        } else if (a == "--waist-factor") {
            o.c.waist_factor = detail::to_double(a, value());
        } else if (a == "--allow-wide-beam") {
            o.c.allow_wide_beam = true;
        } else if (a == "--beam") {
            const std::string v = value();
            if (v == "paraxial")
                o.c.beam = Beam::paraxial;
            else if (v == "angular-spectrum")
                o.c.beam = Beam::angular_spectrum;
            else
                throw std::invalid_argument("--beam must be paraxial or angular-spectrum");
        } else if (a == "--digits") {
            o.c.digits = detail::to_double(a, value());
        } else if (a == "--leaf-radius-quantile") {
            o.c.leaf_radius_quantile = detail::to_double(a, value());
        } else if (a == "--tol") {
            o.c.tolerance = detail::to_double(a, value());
        } else if (a == "--max-iter") {
            o.c.max_iter = detail::to_int(a, value());
        } else if (a == "--restart") {
            o.c.restart = detail::to_int(a, value());
        } else if (a == "--matvecs") {
            o.c.matvecs = detail::to_int(a, value());
        } else if (a == "--exact-budget-gb") {
            const double g = detail::to_double(a, value());
            if (!(g > 0.0))
                throw std::invalid_argument("--exact-budget-gb must be positive");
            o.c.max_exact_far_bytes = static_cast<std::size_t>(g * 1e9);
        } else if (a == "--formulation") {
            o.c.formulation = formulation::parse_kind(value());  // throws for an unknown name
        } else if (a == "--jacobi") {
            const std::string v = value();
            if (v != "0" && v != "1")
                throw std::invalid_argument("--jacobi must be 0 or 1");
            o.c.jacobi = v == "1";
        } else if (a == "--estimate-only") {
            o.estimate_only = true;
        } else if (a == "--no-solve") {
            o.c.solve = false;
        } else if (a == "--force") {
            o.force = true;
        } else {
            throw std::invalid_argument("unknown option " + a);
        }
    }
    if (o.c.matvecs < 1)
        throw std::invalid_argument("--matvecs must be >= 1");
    if (o.history_stride < 1)
        throw std::invalid_argument("--history-stride must be >= 1");
    if (o.history_digits < 1 || o.history_digits > 17)
        throw std::invalid_argument("--history-digits must be in 1..17");
    if ((!o.c.solve || o.estimate_only) &&
        (!o.history_file.empty() || !o.history_full_file.empty()))
        throw std::invalid_argument(
            "--history-file / --history-full-file need a solve (not with --no-solve or "
            "--estimate-only)");
    // --far-field / --far-field-degree / --beam: both hemisphere grids, the quadrature degree and
    // the beam are checked here, before the estimate and the assembly, so that a bad option
    // cannot surface only after a multi-hour solve.
    validate_power_summary(o.c);
    return o;
}

/// Opens each history file of `o` for appending (creating it if absent) and closes it again, so
/// that an unwritable path fails before the assembly and the solve rather than after them.
/// @throws std::invalid_argument naming the file if it cannot be opened for appending.
inline void probe_history_files(const Options& o) {
    for (const std::string* p : {&o.history_file, &o.history_full_file}) {
        if (p->empty())
            continue;
        const std::ofstream f(core::path_from_utf8(*p), std::ios::app | std::ios::binary);
        if (!f)
            throw std::invalid_argument("cannot open history file " + *p + " for appending");
    }
}

}  // namespace mlfmm_scaling
