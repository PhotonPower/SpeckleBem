/// @file test_pml_ring.cpp
/// WP-PML5a (ADR 0008 amendment 2026-10-18): the fast path of the PML route, MlfmmOperator with a
/// PML stretch = Z_rr (MLFMM on the real functions) + Z_ring (mlfmm::PmlRingOperator, exact).
/// Classification = right-hand-side mask, pattern completeness (every pair in exactly one part),
/// ring entries against independently assembled exact columns and the dense PML matrix, Z_rr
/// against the unstretched MLFMM on the same mesh, budget refusal, describe / memory_bytes, the
/// Simulation PML route with "mlfmm". Fixed seeds; tiny flat open surfaces with cheap kernels.
/// The docs/05 matvec criteria at d0 = 3 and 5 against the dense PML matrix (minutes in release)
/// are tests/validation/test_pml_ring.cpp.
#include "specklebem/basis/rwg.hpp"
#include "specklebem/compression/mlfmm/far_operator.hpp"
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/compression/mlfmm/near_field.hpp"
#include "specklebem/compression/mlfmm/pml_ring.hpp"
#include "specklebem/core/system_memory.hpp"
#include "specklebem/excitation/angular_spectrum_beam.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/geometry/coordinate_stretch.hpp"
#include "specklebem/geometry/open_surface.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/operator/region_sparse_operator.hpp"
#include "specklebem/operator/sparse_operator.hpp"
#include "specklebem/simulation.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <spdlog/sinks/ringbuffer_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef SPECKLEBEM_HAVE_OPENMP
#include <omp.h>
#endif

using namespace specklebem;
using Catch::Matchers::ContainsSubstring;
using geometry::CoordinateStretch;
using geometry::OpenSurfaceMesh;
using mlfmm::MlfmmOperator;

namespace {

constexpr Real kLambda = 500e-9;

std::size_t sz(Index i) {
    return static_cast<std::size_t>(i);
}

/// Flat open surface with spacing h and the given zone widths (multiples of h).
OpenSurfaceMesh flat_open(Real h, Real patch, Real taper, Real gap, Real pml) {
    geometry::OpenSurfaceParams op;
    op.patch_size = patch;
    op.taper_width = taper;
    op.gap_width = gap;
    op.pml_width = pml;
    const Real edge = patch + 2.0 * taper;
    const auto cells = static_cast<Index>(std::llround(edge / h));
    geometry::HeightMap m;
    m.dx = edge / static_cast<Real>(cells);
    m.dy = m.dx;
    m.z = MatrixXr::Zero(cells + 1, cells + 1);
    return geometry::make_open_surface_mesh(m, op);
}

/// The tiny surface of the structural cases: patch 0.2, taper 0.1, gap 0.1, PML 0.1 um at
/// h = 100 nm (8 x 8 cells, 128 triangles, 2N = 352).
OpenSurfaceMesh tiny_surface() {
    return flat_open(100e-9, 0.2e-6, 0.1e-6, 0.1e-6, 0.1e-6);
}

/// The tiny rough case (W1 of the WP-PML5a review): a Gaussian patch (fixed seed, sigma 20 nm,
/// Lc = 200 nm = the taper width) at h = 100 nm, patch 0.2, taper 0.2, gap 0.1, PML 0.1 um
/// (10 x 10 cells, 200 triangles). The rough triangles make K non-zero, in particular between
/// the real rows of the patch and the stretched (flat) columns (on a flat mesh K vanishes).
OpenSurfaceMesh tiny_rough_surface() {
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

/// Cheap kernels for structural checks (exact for any options; as test_pml_route's tiny solve).
kernels::OperatorOptions cheap_options() {
    kernels::OperatorOptions opt;
    opt.outer_grading_levels = 0;
    opt.target_accuracy = 0.0;
    opt.quad_degree_far = 2;
    opt.quad_degree_near = 4;
    opt.quad_degree_sing = 2;
    opt.stretched_touching_points = 4;
    return opt;
}

/// Open-surface problem with the PML stretch set.
struct Case {
    OpenSurfaceMesh om;
    basis::RwgSpace space;
    std::unique_ptr<formulation::Formulation> form;
    CoordinateStretch pml;
    op::Problem problem;
    Case(OpenSurfaceMesh m, material::Material object, Real sigma_max, formulation::Kind kind)
        : Case(m, object, kind, geometry::make_pml_stretch(m, sigma_max)) {}
    /// With an explicit stretch (other profile orders or starts).
    Case(OpenSurfaceMesh m, material::Material object, formulation::Kind kind,
         CoordinateStretch stretch)
        : om(std::move(m)),
          space(om.mesh),
          form(formulation::make_formulation(kind)),
          pml(std::move(stretch)) {
        problem.space = &space;
        problem.exterior = material::vacuum();
        problem.object = object;
        problem.formulation = form.get();
        problem.omega = 2.0 * constants::pi * constants::c0 / kLambda;
        problem.kernel_options = cheap_options();
        problem.stretch = &pml;
    }
    Case(const Case&) = delete;
    Case& operator=(const Case&) = delete;
};

/// A low-index object and sigma_max 1 keep the stretched touching rule at its base order.
std::unique_ptr<Case> tiny_case(formulation::Kind kind = formulation::Kind::PMCHWT) {
    return std::make_unique<Case>(tiny_surface(), material::Material{Complex(2.25, 0.0), 1.0}, 1.0,
                                  kind);
}

/// Exactly `levels` octree levels (no leaf rule), d0 digits.
mlfmm::MlfmmParams tree_params(int levels, Real digits) {
    mlfmm::MlfmmParams p;
    p.octree = mlfmm::OctreeParams{1, levels, 0.0};
    p.accuracy_digits = digits;
    p.automatic_leaf_size = false;
    return p;
}

VectorXc random_vector(Index n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    const auto u = [&]() { return static_cast<Real>(rng() >> 11) * 0x1.0p-53 - 0.5; };
    VectorXc x(n);
    for (Index i = 0; i < n; ++i) x(i) = Complex(u(), u());
    return x;
}

/// Leaf box (index into boxes()) of every basis function.
std::vector<Index> leaf_of(const mlfmm::Octree& tree) {
    std::vector<Index> leaf(tree.permutation().size(), -1);
    for (const Index b : tree.boxes_at_level(tree.leaf_level())) {
        const mlfmm::Box& box = tree.boxes()[sz(b)];
        for (Index p = box.first_element; p < box.first_element + box.num_elements; ++p)
            leaf[sz(tree.permutation()[sz(p)])] = b;
    }
    return leaf;
}

bool near_leaves(const mlfmm::Octree& tree, Index a, Index b) {
    const auto& u = tree.boxes()[sz(a)].ijk;
    const auto& v = tree.boxes()[sz(b)].ijk;
    return std::abs(u[0] - v[0]) <= 1 && std::abs(u[1] - v[1]) <= 1 && std::abs(u[2] - v[2]) <= 1;
}

/// True if the row-major sparse matrix stores (r, c).
bool stored(const op::SparseOperator::Matrix& Z, Index r, Index c) {
    const Index* outer = Z.outerIndexPtr();
    const Index* inner = Z.innerIndexPtr();
    const Index* b = inner + outer[r];
    const Index* e = inner + outer[r + 1];
    const Index* it = std::lower_bound(b, e, c);
    return it != e && *it == c;
}

/// Exact columns `cols` (J and M, all rows) of the stretched system, via
/// op::assemble_sparse_stretched on a one-group pattern (independent of the ring's far path).
std::shared_ptr<op::SparseOperator> exact_columns(const op::Problem& p,
                                                  const std::vector<Index>& cols) {
    op::BasisPattern pat;
    pat.group_of_row.assign(sz(p.space->size()), 0);
    pat.cols = cols;
    std::sort(pat.cols.begin(), pat.cols.end());
    pat.col_ptr = {0, static_cast<Index>(pat.cols.size())};
    return op::assemble_sparse_stretched(p, pat);
}

Real rel(const VectorXc& a, const VectorXc& b) {
    return (a - b).norm() / b.norm();
}

}  // namespace

TEST_CASE("pml_ring: classification equals the right-hand-side mask", "[pml_ring]") {
    const auto c = tiny_case();
    const std::vector<std::uint8_t> s = mlfmm::classify_stretched(c->space, c->pml);
    const std::vector<std::uint8_t> mask = op::pml_masked_functions(c->space, c->pml);
    REQUIRE(s.size() == mask.size());
    CHECK(s == mask);
    const auto ns = std::count(s.begin(), s.end(), std::uint8_t{1});
    CHECK(ns > 0);
    CHECK(ns < c->space.size());
    // Identity stretch: nothing stretched; a stretch with sigma_max = 0 still classifies by the
    // geometric zone (as the mask).
    const auto none = mlfmm::classify_stretched(c->space, CoordinateStretch::identity());
    CHECK(std::all_of(none.begin(), none.end(), [](std::uint8_t v) { return v == 0; }));
    const CoordinateStretch zero = geometry::make_pml_stretch(c->om, 0.0);
    CHECK(mlfmm::classify_stretched(c->space, zero) == op::pml_masked_functions(c->space, zero));
}

TEST_CASE("pml_ring: split near pattern partitions the near pattern", "[pml_ring]") {
    const auto c = tiny_case();
    const mlfmm::Octree tree(c->space, kLambda, mlfmm::OctreeParams{1, 3, 0.0});
    const std::vector<std::uint8_t> s = mlfmm::classify_stretched(c->space, c->pml);
    const op::BasisPattern base = mlfmm::near_pattern(tree, c->space);
    const mlfmm::SplitNearPattern split = mlfmm::split_near_pattern(tree, c->space, s);
    const Index N = c->space.size();
    const auto cols = [](const op::BasisPattern& p, Index m) {
        const auto g = sz(p.group_of_row[sz(m)]);
        return std::vector<Index>(p.cols.begin() + p.col_ptr[g], p.cols.begin() + p.col_ptr[g + 1]);
    };
    Index real_pairs = 0, ring_pairs = 0;
    bool ok = true;
    for (Index m = 0; m < N; ++m) {
        const std::vector<Index> b = cols(base, m), r = cols(split.real, m),
                                 g = cols(split.ring, m);
        real_pairs += static_cast<Index>(r.size());
        ring_pairs += static_cast<Index>(g.size());
        std::vector<Index> u;
        std::merge(r.begin(), r.end(), g.begin(), g.end(), std::back_inserter(u));
        ok = ok && u == b;  // disjoint union (duplicates would break the equality)
        for (const Index n : r) ok = ok && s[sz(m)] == 0 && s[sz(n)] == 0;
        for (const Index n : g) ok = ok && (s[sz(m)] != 0 || s[sz(n)] != 0);
    }
    CHECK(ok);
    CHECK(real_pairs == split.real_pairs);
    CHECK(ring_pairs == split.ring_pairs);
    CHECK(real_pairs > 0);
    CHECK(ring_pairs > 0);
    CHECK_THROWS_AS(mlfmm::split_near_pattern(tree, c->space, std::vector<std::uint8_t>(3, 0)),
                    std::invalid_argument);
}

TEST_CASE("pml_ring: every pair lies in exactly one of Z_rr near / far and the ring near / far",
          "[pml_ring]") {
    const auto c = tiny_case();
    const MlfmmOperator Z(c->problem, tree_params(3, 3.0));
    REQUIRE(Z.ring_operator() != nullptr);
    const mlfmm::PmlRingOperator& ring = *Z.ring_operator();
    const mlfmm::Octree& tree = Z.octree();
    REQUIRE(tree.levels() == 3);
    const Index N = c->space.size();
    const std::vector<std::uint8_t>& s = ring.stretched();
    CHECK(s == op::pml_masked_functions(c->space, c->pml));
    const std::vector<Index> leaf = leaf_of(tree);
    const auto& rr_near = Z.near_operator().matrix();
    const auto& ring_near = ring.near_operator().matrix();
    Index counts[4] = {0, 0, 0, 0};
    Index violations = 0, block_mismatch = 0;
    for (Index m = 0; m < N; ++m) {
        for (Index n = 0; n < N; ++n) {
            const bool real_pair = s[sz(m)] == 0 && s[sz(n)] == 0;
            const bool near = near_leaves(tree, leaf[sz(m)], leaf[sz(n)]);
            const bool in[4] = {stored(rr_near, m, n), real_pair && !near, stored(ring_near, m, n),
                                ring.far_contains(m, n)};
            int k = 0;
            for (int i = 0; i < 4; ++i) {
                k += in[i] ? 1 : 0;
                counts[i] += in[i] ? 1 : 0;
            }
            if (k != 1)
                ++violations;
            // Real pairs only in Z_rr, ring pairs only in the ring; near pairs in a near part.
            if ((real_pair && (in[2] || in[3])) || (!real_pair && (in[0] || in[1])) ||
                (near && (in[1] || in[3])) || (!near && (in[0] || in[2])))
                ++violations;
            // All four blocks stored alike.
            for (const auto* M : {&rr_near, &ring_near}) {
                const bool jj = stored(*M, m, n);
                if (stored(*M, m, N + n) != jj || stored(*M, N + m, n) != jj ||
                    stored(*M, N + m, N + n) != jj)
                    ++block_mismatch;
            }
        }
    }
    CHECK(violations == 0);
    CHECK(block_mismatch == 0);
    CHECK(counts[0] > 0);
    CHECK(counts[1] > 0);  // the tiny tree has real-real far pairs
    CHECK(counts[2] == ring.near_pairs());
    CHECK(counts[3] == ring.far_pairs());
    CHECK(counts[3] > 0);
    CHECK(ring.estimate().near_bytes == ring.near_bytes());
    // ACA path (default): the stored blocks never exceed the exact estimate; the exact path's
    // allocation equals it.
    const mlfmm::PmlRingFarStats& st = ring.far_stats();
    CHECK(st.aca);
    CHECK(st.exact_bytes + st.aca_bytes <= ring.estimate().far_bytes);
    CHECK(st.exact_pairs + st.aca_pairs + st.dropped_pairs == ring.far_stored_pairs());
    mlfmm::MlfmmParams exact = tree_params(3, 3.0);
    exact.pml_ring.use_aca = false;
    const MlfmmOperator Ze(c->problem, exact);
    CHECK_FALSE(Ze.ring_operator()->far_stats().aca);
    CHECK(Ze.ring_operator()->estimate().far_bytes == Ze.ring_operator()->far_bytes());
    CHECK(Ze.ring_operator()->far_blocks().empty());
}

TEST_CASE("pml_ring: ring columns equal the exact stretched entries", "[pml_ring]") {
    const auto c = tiny_case();
    const MlfmmOperator Z(c->problem, tree_params(3, 3.0));
    const mlfmm::PmlRingOperator& ring = *Z.ring_operator();
    const Index N = c->space.size();
    const std::vector<std::uint8_t>& s = ring.stretched();
    // Two stretched and two real columns, spread over the mesh.
    std::vector<Index> picks;
    for (const std::uint8_t want : {std::uint8_t{1}, std::uint8_t{0}}) {
        std::vector<Index> of;
        for (Index n = 0; n < N; ++n)
            if (s[sz(n)] == want)
                of.push_back(n);
        REQUIRE(of.size() >= 2);
        picks.push_back(of[of.size() / 5]);
        picks.push_back(of[(4 * of.size()) / 5]);
    }
    const auto exact = exact_columns(c->problem, picks);
    for (const Index n : picks) {
        for (const Index col : {n, N + n}) {
            const VectorXc e = VectorXc::Unit(2 * N, col);
            const VectorXc yr = ring * e;
            const VectorXc ex = exact->matrix().col(col);
            // Ring rows of this column: the pairs (m, n) with a stretched member.
            Real err = 0, scale = 0, outside = 0;
            for (Index r = 0; r < 2 * N; ++r) {
                const Index m = r % N;
                const bool ring_pair = s[sz(m)] != 0 || s[sz(n)] != 0;
                if (ring_pair) {
                    err = std::max(err, std::abs(yr(r) - ex(r)));
                    scale = std::max(scale, std::abs(ex(r)));
                } else {
                    outside = std::max(outside, std::abs(yr(r)));
                }
            }
            INFO("column " << col << (s[sz(n)] != 0 ? " (stretched)" : " (real)"));
            CHECK(scale > 0.0);
            CHECK(err <= 1e-12 * scale);
            CHECK(outside == 0.0);
        }
    }
}

TEST_CASE("pml_ring: rough patch, transposed K entries and ring x = (D - D_rr) x", "[pml_ring]") {
    // On the flat meshes K vanishes, so the JM / MJ sub-blocks of the transposed products (real
    // rows x stretched far columns) would be untested; the rough patch makes them non-zero.
    Case c(tiny_rough_surface(), material::Material{Complex(2.25, 0.0), 1.0}, 1.0,
           formulation::Kind::PMCHWT);
    const Vertices& V = c.om.mesh.vertices();
    REQUIRE(V.col(2).cwiseAbs().maxCoeff() > 1e-9);
    const MlfmmOperator Z(c.problem, tree_params(3, 3.0));
    const mlfmm::PmlRingOperator& ring = *Z.ring_operator();
    const Index N = c.space.size();
    const std::vector<std::uint8_t>& s = ring.stretched();
    // Four stretched columns, spread over the mesh, each with far real rows.
    std::vector<Index> picks;
    {
        std::vector<Index> of;
        for (Index n = 0; n < N; ++n) {
            bool far_real = false;
            for (Index m = 0; m < N && !far_real; ++m)
                far_real = s[sz(m)] == 0 && ring.far_contains(m, n);
            if (s[sz(n)] != 0 && far_real)
                of.push_back(n);
        }
        REQUIRE(of.size() >= 4);
        for (const std::size_t k : {std::size_t{1}, std::size_t{3}, std::size_t{5}, std::size_t{7}})
            picks.push_back(of[(k * of.size()) / 8]);
    }
    const auto exact = exact_columns(c.problem, picks);
    Real k_transposed = 0, k_scale = 0;
    for (const Index n : picks) {
        for (const Index col : {n, N + n}) {
            const VectorXc yr = ring * VectorXc::Unit(2 * N, col);
            const VectorXc ex = exact->matrix().col(col);
            Real err = 0, scale = 0;
            for (Index r = 0; r < 2 * N; ++r) {
                err = std::max(err, std::abs(yr(r) - ex(r)));
                scale = std::max(scale, std::abs(ex(r)));
            }
            // The K sub-blocks of the transposed products: MJ(m, n) in column n (rows N + m),
            // JM(m, n) in column N + n (rows m), m real and (m, n) far.
            for (Index m = 0; m < N; ++m) {
                if (s[sz(m)] == 0 && ring.far_contains(m, n))
                    k_transposed = std::max(k_transposed, std::abs(yr(col == n ? N + m : m)));
            }
            k_scale = std::max(k_scale, scale);
            INFO("column " << col);
            CHECK(scale > 0.0);
            CHECK(err <= 1e-12 * scale);
        }
    }
    INFO("max |JM|, |MJ| over the transposed far rows " << k_transposed << " (column scale "
                                                        << k_scale << ")");
    // Measured 7e-6 of the column scale (the self terms of L dominate it); exactly 0 when flat.
    // The column check above bounds their error by 1e-12 of the scale, far below their size.
    CHECK(k_transposed > 1e-6 * k_scale);
    // ring x = (D - D_rr) x, D_rr = D with the stretched rows and columns zeroed.
    const auto D = op::DenseStrategy().build(c.problem);
    const VectorXc x = random_vector(2 * N, 19);
    VectorXc xr = x;
    for (Index n = 0; n < N; ++n)
        if (s[sz(n)] != 0)
            xr(n) = xr(N + n) = Complex(0.0, 0.0);
    VectorXc yrr = *D * xr;
    for (Index n = 0; n < N; ++n)
        if (s[sz(n)] != 0)
            yrr(n) = yrr(N + n) = Complex(0.0, 0.0);
    const VectorXc expected = *D * x - yrr;
    const VectorXc y = ring * x;
    INFO("relative difference " << rel(y, expected));
    CHECK(rel(y, expected) < 1e-12);
}

TEST_CASE("pml_ring: Z_rr equals the unstretched MLFMM restricted to the real functions",
          "[pml_ring]") {
    const auto c = tiny_case(formulation::Kind::ICTF);
    const mlfmm::MlfmmParams params = tree_params(3, 3.0);
    const MlfmmOperator Z(c->problem, params);
    op::Problem plain = c->problem;
    plain.stretch = nullptr;
    const MlfmmOperator Z0(plain, params);
    CHECK(Z0.ring_operator() == nullptr);
    const Index N = c->space.size();
    const std::vector<std::uint8_t>& s = Z.ring_operator()->stretched();
    const VectorXc x = random_vector(2 * N, 7);
    VectorXc xr = x;
    for (Index n = 0; n < N; ++n)
        if (s[sz(n)] != 0)
            xr(n) = xr(N + n) = Complex(0.0, 0.0);
    VectorXc y_rr;
    Z.apply_real(x, y_rr);
    VectorXc y0 = Z0 * xr;
    for (Index n = 0; n < N; ++n)
        if (s[sz(n)] != 0)
            y0(n) = y0(N + n) = Complex(0.0, 0.0);
    CHECK(rel(y_rr, y0) < 1e-13);
    // Rows of stretched functions are exactly zero in Z_rr.
    Real stretched_rows = 0;
    for (Index n = 0; n < N; ++n)
        if (s[sz(n)] != 0)
            stretched_rows = std::max({stretched_rows, std::abs(y_rr(n)), std::abs(y_rr(N + n))});
    CHECK(stretched_rows == 0.0);
    // The far part with the mask: zero rows, no exact pairs with a stretched member.
    CHECK(Z.far_operator().excluded_count() ==
          static_cast<Index>(std::count(s.begin(), s.end(), std::uint8_t{1})));
    CHECK(Z0.far_operator().excluded_count() == 0);
    // apply() = apply_real() + ring.
    const VectorXc y = Z * x;
    const VectorXc yring = *Z.ring_operator() * x;
    CHECK(rel(y, y_rr + yring) < 1e-14);
}

TEST_CASE("pml_ring: exact-part bookkeeping of the masked far operator skips excluded functions",
          "[pml_ring]") {
    // Both regions forced exact (no per-basis truncation): the pre-assembly bound equals the
    // assembled pairs only if it counts the functions that are not excluded per box; counting the
    // excluded ones (they hold no exact pair) would make the budget throw spuriously.
    const auto c = tiny_case();
    op::Problem plain = c->problem;
    plain.stretch = nullptr;
    const mlfmm::Octree tree(c->space, kLambda, mlfmm::OctreeParams{1, 3, 0.0});
    REQUIRE(tree.levels() == 3);
    mlfmm::MlfmmParams p = tree_params(3, 3.0);
    p.exact_far_regions = {true, true};
    p.max_exact_far_bytes = std::numeric_limits<std::size_t>::max();
    // Mask: the PML classification plus every function of the fullest leaf (an all-excluded box
    // with far partners).
    std::vector<std::uint8_t> s = mlfmm::classify_stretched(c->space, c->pml);
    const Index N = c->space.size();
    const std::vector<Index>& leaves = tree.boxes_at_level(tree.leaf_level());
    const auto fullest = *std::max_element(leaves.begin(), leaves.end(), [&](Index a, Index b) {
        return tree.boxes()[sz(a)].num_elements < tree.boxes()[sz(b)].num_elements;
    });
    const mlfmm::Box& box = tree.boxes()[sz(fullest)];
    REQUIRE(box.num_elements > 0);
    REQUIRE(!box.interaction_list.empty());
    for (Index q = box.first_element; q < box.first_element + box.num_elements; ++q)
        s[sz(tree.permutation()[sz(q)])] = 1;
    const mlfmm::MlfmmFarOperator full(plain, tree, p);
    const mlfmm::MlfmmFarOperator masked(plain, tree, p, s);
    std::size_t bytes = 0;
    for (const int r : {0, 1}) {
        INFO("region " << r);
        const mlfmm::ExactPartInfo& x = masked.exact_info(r);
        REQUIRE(masked.exact_part(r) != nullptr);
        CHECK(x.pairs > 0);
        CHECK(x.pairs == masked.exact_part(r)->pairs());
        CHECK(x.pair_bound == x.pairs);
        CHECK(full.exact_info(r).pair_bound == full.exact_info(r).pairs);
        CHECK(full.exact_info(r).pairs > x.pairs);
        bytes += x.bytes;
    }
    // Exactly the assembled bytes as the budget: accepted without the per-basis count.
    p.max_exact_far_bytes = bytes;
    const mlfmm::MlfmmFarOperator fits(plain, tree, p, s);
    for (const int r : {0, 1}) {
        CHECK(fits.exact_info(r).counted_pairs == -1);
        CHECK(fits.exact_info(r).pairs == masked.exact_info(r).pairs);
    }
    // Everything excluded: every count is zero.
    const mlfmm::MlfmmFarOperator none(plain, tree, p, std::vector<std::uint8_t>(sz(N), 1));
    for (const int r : {0, 1}) {
        CHECK(none.exact_info(r).pair_bound == 0);
        CHECK(none.exact_info(r).pairs == 0);
        CHECK(none.exact_info(r).truncated_pairs == 0);
    }
    CHECK(none.excluded_count() == N);
}

TEST_CASE("pml_ring: matvec of the tiny case against the dense PML matrix", "[pml_ring]") {
    // Structure and assembly paths end to end on the tiny case (cheap kernels: the MLFMM far
    // part is only as accurate as its patterns, so the bound is loose here; the release case
    // below checks the docs/05 criteria).
    const auto c = tiny_case();
    const MlfmmOperator Z(c->problem, tree_params(3, 3.0));
    const auto D = op::DenseStrategy().build(c->problem);
    const Index N = c->space.size();
    const VectorXc x = random_vector(2 * N, 11);
    const VectorXc yd = *D * x;
    const VectorXc yf = Z * x;
    INFO("relative matvec error " << rel(yf, yd));
    CHECK(rel(yf, yd) < 5e-2);
    // The ring rows (stretched test functions) are exact: they agree to rounding.
    const std::vector<std::uint8_t>& s = Z.ring_operator()->stretched();
    Real err = 0, scale = 0;
    for (Index n = 0; n < N; ++n) {
        if (s[sz(n)] == 0)
            continue;
        for (const Index r : {n, N + n}) {
            err = std::max(err, std::abs(yf(r) - yd(r)));
            scale = std::max(scale, std::abs(yd(r)));
        }
    }
    CHECK(err <= 1e-12 * scale);
}

TEST_CASE("pml_ring: budget refusal, errors, describe and memory", "[pml_ring]") {
    const auto c = tiny_case();
    mlfmm::MlfmmParams p = tree_params(3, 3.0);
    p.pml_ring.max_far_bytes = 1;
    CHECK_THROWS_AS(MlfmmOperator(c->problem, p), std::runtime_error);
    CHECK_THROWS_WITH(MlfmmOperator(c->problem, p), ContainsSubstring("max_far_bytes"));
    const mlfmm::Octree tree(c->space, kLambda, mlfmm::OctreeParams{1, 3, 0.0});
    CHECK_THROWS_WITH(mlfmm::PmlRingOperator(c->problem, tree, p.pml_ring),
                      ContainsSubstring("would need"));
    const std::vector<std::uint8_t> s = mlfmm::classify_stretched(c->space, c->pml);
    const mlfmm::PmlRingEstimate est = mlfmm::estimate_pml_ring(tree, c->space, s);
    CHECK(est.budget == est.dense_bytes);
    CHECK(est.far_bytes < est.dense_bytes);
    CHECK(est.near_pairs + est.far_pairs + est.rr_near_pairs <= c->space.size() * c->space.size());
    CHECK_NOTHROW(mlfmm::check_pml_ring_budget(est, {}));
    // Memory warning (review W3): above the available memory only; 0 = unknown. The peak counts
    // the near ring and the larger of its assembly cache and the far blocks known to be exact
    // before assembly (WP-PML5c review S1, WP-PML5b; estimate_pml_ring alone has no stretch,
    // hence no cache estimate).
    CHECK(est.near_assembly_bytes == 0);
    // WP-PML5d: the far-ring pair memo (pair_memo_bytes per assembling thread) is counted with
    // the far blocks known to be exact.
    CHECK(est.far_memo_bytes > 0);
    CHECK(est.far_memo_bytes % mlfmm::PmlRingParams{}.pair_memo_bytes == 0);
    CHECK(mlfmm::pml_ring_peak_bytes(est) ==
          est.near_bytes + est.far_exact_bytes + est.far_memo_bytes);
    CHECK(mlfmm::warn_pml_ring_memory(est, 1));
    CHECK(mlfmm::warn_pml_ring_memory(est, est.far_exact_bytes));
    CHECK_FALSE(mlfmm::warn_pml_ring_memory(est, mlfmm::pml_ring_peak_bytes(est)));
    CHECK_FALSE(mlfmm::warn_pml_ring_memory(est, 0));
    // An unlimited memo budget saturates the peak instead of wrapping it (review W2).
    mlfmm::PmlRingEstimate unlimited = est;
    unlimited.far_memo_bytes = std::numeric_limits<std::size_t>::max();
    CHECK(mlfmm::pml_ring_peak_bytes(unlimited) == std::numeric_limits<std::size_t>::max());
    CHECK(mlfmm::warn_pml_ring_memory(unlimited, unlimited.near_bytes + unlimited.far_exact_bytes));
    // Without ACA every far block is exact: the peak of WP-PML5c, near + far_bytes.
    mlfmm::PmlRingParams no_aca;
    no_aca.use_aca = false;
    const mlfmm::PmlRingEstimate est_exact = mlfmm::estimate_pml_ring(tree, c->space, s, no_aca);
    CHECK(est_exact.far_exact_bytes == est_exact.far_bytes);
    CHECK(est_exact.far_memo_bytes == 0);
    CHECK(mlfmm::pml_ring_peak_bytes(est_exact) == est_exact.near_bytes + est_exact.far_bytes);
    CHECK(mlfmm::warn_pml_ring_memory(est_exact, est_exact.far_bytes));
    CHECK_FALSE(mlfmm::warn_pml_ring_memory(est_exact, mlfmm::pml_ring_peak_bytes(est_exact)));
    // The plan estimates the near-ring assembly cache: an upper bound of the measured one.
    const mlfmm::PmlRingPlan plan = mlfmm::plan_pml_ring(tree, c->space, c->pml);
    const op::SparseStretchedCacheEstimate ce =
        op::estimate_sparse_stretched_cache(c->space, c->pml, plan.split.ring);
    CHECK(plan.estimate.near_assembly_bytes == ce.bytes);
    CHECK(ce.bytes >= static_cast<std::size_t>(ce.canonical_pairs) * 576);
    const mlfmm::PmlRingOperator planned(c->problem, tree);
    const op::SparseStretchedStats& ns = planned.near_stats();
    CHECK(ns.canonical_pairs == ce.canonical_pairs);
    CHECK(ns.touching_pairs == ce.touching_pairs);
    CHECK(ns.cache_bytes > 0);
    CHECK(ns.cache_bytes <= ce.bytes);
    CHECK(planned.estimate().near_assembly_bytes == ce.bytes);
    const mlfmm::PmlRingEstimate pe = planned.estimate();
    CHECK(mlfmm::pml_ring_peak_bytes(pe) ==
          pe.near_bytes + std::max(pe.near_assembly_bytes, pe.far_exact_bytes + pe.far_memo_bytes));
    // The measured peak: near ring + max(measured cache, allocated far blocks incl. ACA factors
    // + the pair memos alive at once).
    const mlfmm::PmlRingFarStats& fs = planned.far_stats();
    CHECK(fs.memo_bytes_max > 0);
    CHECK(fs.memo_bytes_max <= fs.memo_budget);
    CHECK(fs.memo_peak_bytes >= fs.memo_bytes_max);
    CHECK(fs.memo_peak_bytes <= pe.far_memo_bytes);
    CHECK(planned.peak_bytes() ==
          planned.near_bytes() +
              std::max(ns.cache_bytes, planned.far_bytes() + fs.memo_peak_bytes));
    CHECK(planned.far_stats().exact_bytes + planned.far_stats().aca_bytes <= pe.far_bytes);
    CHECK(op::estimate_sparse_stretched_cache(c->space, c->pml, plan.split.ring,
                                              op::StretchedPairEvaluation::per_ordering)
              .bytes == 0);
#if defined(_WIN32) || defined(__linux__)
    CHECK(core::available_memory_bytes() > 0);
#endif
    // Without a stretch the ring refuses; the local leaf rule is refused with a stretch.
    op::Problem plain = c->problem;
    plain.stretch = nullptr;
    CHECK_THROWS_WITH(mlfmm::PmlRingOperator(plain, tree), ContainsSubstring("no (non-identity)"));
    const CoordinateStretch id = CoordinateStretch::identity();
    plain.stretch = &id;
    CHECK(MlfmmOperator(plain, tree_params(3, 3.0)).ring_operator() == nullptr);
    mlfmm::MlfmmParams q;
    q.leaf_radius_quantile = 0.9;
    CHECK_THROWS_WITH(MlfmmOperator(c->problem, q), ContainsSubstring("leaf_radius_quantile = 1"));
    // describe / memory_bytes.
    const MlfmmOperator Z(c->problem, tree_params(3, 3.0));
    const mlfmm::PmlRingOperator& ring = *Z.ring_operator();
    CHECK(ring.memory_bytes() >= ring.near_bytes() + ring.far_bytes());
    CHECK(ring.near_bytes() == ring.near_operator().memory_bytes());
    CHECK(Z.memory_bytes() >=
          Z.near_operator().memory_bytes() + Z.far_operator().memory_bytes() + ring.memory_bytes());
    CHECK(ring.stretched_count() == est.stretched);
    CHECK(ring.far_pairs() == est.far_pairs);
    CHECK(ring.far_stored_pairs() == est.far_stored);
    const std::string d = Z.describe();
    CHECK_THAT(d, ContainsSubstring("Z = Z_rr + Z_ring"));
    CHECK_THAT(d, ContainsSubstring("ring: PmlRingOperator"));
    CHECK_THAT(d, ContainsSubstring("near (real-real)"));
    CHECK_THAT(d, ContainsSubstring("excluded (PML ring"));
    CHECK_THAT(d, ContainsSubstring("near exact, far ACA"));
    CHECK_THAT(d, ContainsSubstring("ACA far ring, WP-PML5b"));
    CHECK_THAT(ring.describe(),
               ContainsSubstring(std::to_string(ring.stretched_count()) + " stretched"));
    CHECK_THROWS_AS(ring.far_contains(-1, 0), std::out_of_range);
    VectorXc bad(3);
    VectorXc y;
    CHECK_THROWS_AS(ring.apply(bad, y), std::invalid_argument);
    CHECK_THROWS_AS(Z.apply_real(bad, y), std::invalid_argument);
}

TEST_CASE("pml_ring: apply is deterministic, aliasing-safe and safe for concurrent calls",
          "[pml_ring]") {
    const auto c = tiny_case();
    const MlfmmOperator Z(c->problem, tree_params(3, 3.0));
    const mlfmm::PmlRingOperator& ring = *Z.ring_operator();
    const Index n2 = 2 * c->space.size();
    const VectorXc x = random_vector(n2, 3);
    const VectorXc y1 = ring * x;
    const VectorXc y2 = ring * x;
    CHECK((y1.array() == y2.array()).all());
    VectorXc a = x;
    ring.apply(a, a);
    CHECK((a.array() == y1.array()).all());
    std::vector<VectorXc> out(4);
    std::vector<std::thread> th;
    for (std::size_t i = 0; i < out.size(); ++i) th.emplace_back([&, i] { ring.apply(x, out[i]); });
    for (auto& t : th) t.join();
    for (const VectorXc& o : out) CHECK((o.array() == y1.array()).all());
#ifdef SPECKLEBEM_HAVE_OPENMP
    // Thread-count independent (pml_ring.hpp): 1 thread and the maximum thread count give
    // bitwise equal results, for the ring and for the whole fast operator.
    const int saved = omp_get_max_threads();
    const int most = std::max(saved, omp_get_num_procs());
    std::vector<VectorXc> ys, zs;
    for (const int t : {1, 2, 3, most}) {
        omp_set_num_threads(t);
        ys.push_back(ring * x);
        zs.push_back(Z * x);
    }
    omp_set_num_threads(saved);
    for (std::size_t i = 1; i < ys.size(); ++i) {
        INFO("thread count index " << i);
        CHECK((ys[i].array() == ys[0].array()).all());
        CHECK((zs[i].array() == zs[0].array()).all());
    }
    CHECK((ys[0].array() == y1.array()).all());
#endif
}

TEST_CASE("pml_ring: the PML Simulation solves with compression mlfmm", "[pml_ring]") {
    // Tiny flat case: GMRES on the fast operator against GMRES on the dense one (same
    // preconditioner and tolerance); the report shows the split.
    const auto make = [] { return tiny_surface(); };
    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.object = material::Material{Complex(2.25, 0.0), Complex(1.0, 0.0)};
    cfg.kernels = cheap_options();
    cfg.gmres.tolerance = 1e-8;
    cfg.gmres.max_iter = 2000;
    PmlTermination term;
    term.sigma_max = 1.0;
    term.dark_band_tolerance = 0.9;
    excitation::AngularSpectrumBeam::Params bp;
    bp.wavelength = kLambda;
    bp.waist_radius = 0.4e-6;
    bp.region_radius = 0.6e-6;
    const auto beam = std::make_shared<excitation::AngularSpectrumBeam>(bp);
    SimulationConfig fast = cfg;
    fast.compression = "mlfmm";
    fast.mlfmm.octree.max_elements_per_leaf = 1;  // down to the leaf floor: 3 levels, far pairs
    Simulation sf(make(), term, beam, fast);
    sf.assemble();
    const auto* zf = dynamic_cast<const MlfmmOperator*>(sf.system_operator().get());
    REQUIRE(zf != nullptr);
    REQUIRE(zf->ring_operator() != nullptr);
    CHECK(zf->octree().levels() == 3);
    CHECK(zf->ring_operator()->far_pairs() > 0);
    const solver::GmresResult rf = sf.solve();
    CHECK(rf.converged);
    const std::string rep = sf.report();
    CHECK_THAT(rep, ContainsSubstring("PML ring:"));
    CHECK_THAT(rep, ContainsSubstring("stretched"));
    // Review W2: the per-triangle-pair drop is part of the report (lossless: nothing dropped).
    CHECK_THAT(rep, ContainsSubstring("ring far drop:  triangle pairs reduced 0 of "));
    CHECK_THAT(rep, ContainsSubstring("R1 never, R2 never"));
    Simulation sd(make(), term, beam, cfg);
    const solver::GmresResult rd = sd.solve();
    CHECK(rd.converged);
    CHECK_THAT(sd.report(), !ContainsSubstring("PML ring:"));
    const VectorXc& xf = sf.solution().currents;
    const VectorXc& xd = sd.solution().currents;
    INFO("relative current difference " << rel(xf, xd));
    // The automatic leaf rule puts the tiny mesh on 3 levels (leaf 0.2 um): the fast operator
    // differs from dense by its d0 = 3 far part (plus the cheap kernels' far entries).
    CHECK(rel(xf, xd) < 5e-2);
}

// ---------------------------------------------------------------------------------------------
// WP-PML5b: the ACA far ring.
// ---------------------------------------------------------------------------------------------

namespace {

/// A 2 um flat surface for the partition checks (no assembly): patch 0.75, taper 0.125, gap
/// 0.125, PML 0.375 um at h = 125 nm (16 x 16 cells); on 4 levels (leaves 0.25 um) it has row
/// clusters and column clusters of several levels.
OpenSurfaceMesh aca_surface() {
    return flat_open(125e-9, 0.75e-6, 0.125e-6, 0.125e-6, 0.375e-6);
}

/// Exactly `levels` octree levels on the case's space (no MLFMM leaf rule: the ring alone).
mlfmm::Octree ring_tree(const basis::RwgSpace& space, int levels) {
    return mlfmm::Octree(space, kLambda, mlfmm::OctreeParams{1, levels, 0.0});
}

/// ACA forced on the tiny surfaces: every separated block with a positive distance admissible
/// (eta 20) and no break-even cap (max_rank 1000; the ranks of these poorly separated small blocks
/// exceed the break-even, so the default would store them exactly). The admissibility only sets
/// the ranks; the error is controlled by the ACA tolerance, which the tests check against the
/// exact ring.
mlfmm::PmlRingParams forced_aca() {
    mlfmm::PmlRingParams p;
    p.eta = 20.0;
    p.max_rank = 1000;
    return p;
}

/// The same without the near ring (PmlRingParams::assemble_near = false: the stretched touching
/// pairs dominate the cost under the sanitizers; the near ring is identical on both paths).
mlfmm::PmlRingParams far_only(mlfmm::PmlRingParams p) {
    p.assemble_near = false;
    return p;
}

/// ||(near + far_a - ring_b) x|| / ||ring_b x||: a far-only ring a against the full ring b
/// (near taken from b), maximum over three fixed vectors.
Real ring_difference_far(const mlfmm::PmlRingOperator& far_a, const mlfmm::PmlRingOperator& b,
                         Index n2) {
    Real e = 0;
    for (const std::uint64_t seed : {41u, 42u, 43u}) {
        const VectorXc x = random_vector(n2, seed);
        const VectorXc yb = b * x;
        e = std::max(e, rel(b.near_operator() * x + far_a * x, yb));
    }
    return e;
}

/// ||(ring_a - ring_b) x|| / ||ring_b x||, maximum over three fixed vectors.
Real ring_difference(const mlfmm::PmlRingOperator& a, const mlfmm::PmlRingOperator& b, Index n2) {
    Real e = 0;
    for (const std::uint64_t seed : {41u, 42u, 43u}) {
        const VectorXc x = random_vector(n2, seed);
        e = std::max(e, rel(a * x, b * x));
    }
    return e;
}

}  // namespace

TEST_CASE("pml_ring: ACA far ring vs the exact ring on the tiny flat and rough cases",
          "[pml_ring]") {
    for (const bool rough : {false, true}) {
        Case c(rough ? tiny_rough_surface() : tiny_surface(),
               material::Material{Complex(2.25, 0.0), 1.0}, 1.0, formulation::Kind::PMCHWT);
        const Index n2 = 2 * c.space.size();
        const mlfmm::Octree tree = ring_tree(c.space, 3);
        mlfmm::PmlRingParams exact;
        exact.use_aca = false;
        const mlfmm::PmlRingOperator re(c.problem, tree, exact);
        for (const Real d0 : {3.0, 5.0}) {
            const Real tol = std::pow(10.0, -(d0 + 1.0));
            const mlfmm::PmlRingOperator ra(c.problem, tree, far_only(forced_aca()), d0);
            const mlfmm::PmlRingFarStats& st = ra.far_stats();
            const Real err = ring_difference_far(ra, re, n2);
            INFO((rough ? "rough" : "flat")
                 << ", d0 " << d0 << ": ring difference " << err << "; " << ra.describe());
            REQUIRE(st.aca);
            CHECK(st.aca_blocks > 0);
            CHECK(st.epsilon == tol);
            CHECK(st.drop_tolerance == tol);
            // The criterion: the ACA ring reproduces the exact ring to 10^-(d0+1).
            CHECK(err <= tol);
            // Rank statistics.
            CHECK(st.rank_min <= st.rank_median);
            CHECK(st.rank_median <= st.rank_max);
            CHECK(st.rank_max > 0);
            Index hist = 0;
            for (const Index h : st.rank_histogram) hist += h;
            CHECK(hist == 4 * st.aca_blocks);
            CHECK(st.rank_mean[0] > 0.0);
            CHECK(st.rank_mean[3] > 0.0);
            CHECK(st.generated_rows > 0);
            CHECK(st.generated_columns > 0);
            CHECK(st.aca_pairs + st.exact_pairs + st.dropped_pairs == re.far_stored_pairs());
            if (!rough) {  // K vanishes between flat far pairs: JM, MJ rank 0
                CHECK(st.rank_mean[1] == 0.0);
                CHECK(st.rank_mean[2] == 0.0);
            } else {
                CHECK(st.rank_mean[1] > 0.0);
                CHECK(st.rank_mean[2] > 0.0);
            }
            CHECK_THAT(ra.describe(), ContainsSubstring("ACA far ring, WP-PML5b"));
            CHECK_THAT(ra.describe(), ContainsSubstring("ranks min / median / max"));
            // The default parameters (eta 2, break-even cap): the same accuracy, and the stored
            // blocks never exceed the exact far ring.
            const mlfmm::PmlRingOperator rd(c.problem, tree, far_only({}), d0);
            CHECK(ring_difference_far(rd, re, n2) <= tol);
            CHECK(rd.far_stats().aca_bytes + rd.far_stats().exact_bytes <= re.far_bytes());
        }
        CHECK(re.far_bytes() == re.estimate().far_bytes);
    }
}

TEST_CASE("pml_ring: ACA admissibility partition covers every far ring pair once", "[pml_ring]") {
    Case c(aca_surface(), material::Material{Complex(2.25, 0.0), 1.0}, 1.0,
           formulation::Kind::PMCHWT);
    const mlfmm::Octree tree = ring_tree(c.space, 4);
    const int L = tree.leaf_level();
    const Index N = c.space.size();
    const std::vector<Index> leaf = leaf_of(tree);
    const std::vector<Index>& perm = tree.permutation();
    const std::vector<std::uint8_t> s = mlfmm::classify_stretched(c.space, c.pml);
    for (const bool rows : {true, false}) {
        for (const bool coarse : {true, false}) {
            if (rows && !coarse)
                continue;  // coarse_columns only matters for row leaves
            for (const Real eta : {0.5, 1.0, 2.0}) {
                mlfmm::PmlRingParams p;
                p.eta = eta;
                p.coarse_columns = coarse;
                p.row_clusters = rows;
                const std::vector<mlfmm::PmlRingFarBlock> blocks =
                    mlfmm::plan_pml_ring_far(tree, c.space, s, p);
                const mlfmm::PmlRingEstimate e = mlfmm::estimate_pml_ring(tree, c.space, s, p);
                REQUIRE(!blocks.empty());
                INFO("row clusters " << rows << ", coarse columns " << coarse << ", eta " << eta);
                // cover(m, q): how often the pair (stretched row m, column position q) is stored.
                std::vector<Index> cover(sz(N) * sz(N), 0);
                Index admissible = 0, coarse_rows = 0, coarse_cols = 0, pairs = 0;
                for (const mlfmm::PmlRingFarBlock& b : blocks) {
                    const mlfmm::Box& rb = tree.boxes()[sz(b.row_box)];
                    const mlfmm::Box& cb = tree.boxes()[sz(b.box)];
                    CHECK(b.first == cb.first_element);
                    CHECK(b.count == cb.num_elements);
                    CHECK(b.level == cb.level);
                    CHECK(b.row_level == rb.level);
                    if (rows)
                        CHECK(b.row_level == b.level);  // same-level pairs
                    else
                        CHECK(b.row_level == L);
                    Index nrows = 0;
                    for (Index q = rb.first_element; q < rb.first_element + rb.num_elements; ++q) {
                        const Index m = perm[sz(q)];
                        if (s[sz(m)] == 0)
                            continue;
                        ++nrows;
                        for (Index t = b.first; t < b.first + b.count; ++t)
                            ++cover[sz(m) * sz(N) + sz(t)];
                    }
                    CHECK(nrows == b.rows);
                    pairs += b.rows * b.count;
                    if (b.admissible) {
                        ++admissible;
                        CHECK(b.distance > 0.0);
                        CHECK(std::max(b.row_diameter, b.col_diameter) <= eta * b.distance);
                    } else {
                        CHECK(b.level == L);  // non-admissible blocks are leaf pairs (exact)
                        CHECK(b.row_level == L);
                    }
                    coarse_rows += b.row_level < L ? 1 : 0;
                    coarse_cols += b.level < L ? 1 : 0;
                }
                Index violations = 0;
                for (Index m = 0; m < N; ++m) {
                    for (Index t = 0; t < N; ++t) {
                        const Index want =
                            s[sz(m)] != 0 && !near_leaves(tree, leaf[sz(m)], leaf[sz(perm[sz(t)])])
                                ? 1
                                : 0;
                        violations += cover[sz(m) * sz(N) + sz(t)] != want ? 1 : 0;
                    }
                }
                CHECK(violations == 0);
                CHECK(admissible > 0);
                CHECK(pairs == e.far_stored);
                CHECK(e.far_aca_blocks == admissible);
                CHECK(e.far_exact_blocks == static_cast<Index>(blocks.size()) - admissible);
                CHECK((rows ? coarse_rows > 0 : coarse_rows == 0));
                CHECK((coarse ? coarse_cols > 0 : coarse_cols == 0));
            }
        }
    }
    mlfmm::PmlRingParams off;
    off.use_aca = false;
    CHECK(mlfmm::plan_pml_ring_far(tree, c.space, s, off).empty());
}

TEST_CASE("pml_ring: ACA magnitude-bound dropping is logged and stays below the tolerance",
          "[pml_ring]") {
    // A strongly lossy object (eps_r = -30 - 20j, alpha ~ 72 / um): its region R2 parts of the
    // distant blocks fall below the bound and are dropped; with the same lossy exterior (a
    // diagnostic, not a physical PML case) whole blocks are dropped. The ACA ring still matches
    // the exact one to 10^-(d0+1).
    const material::Material lossy{Complex(-30.0, -20.0), 1.0};
    for (const bool lossy_exterior : {false, true}) {
        Case c(tiny_surface(), lossy, 1.0, formulation::Kind::PMCHWT);
        if (lossy_exterior)
            c.problem.exterior = lossy;
        mlfmm::PmlRingParams exact;
        exact.use_aca = false;
        const mlfmm::Octree tree = ring_tree(c.space, 3);
        // Far rings only (the lossy near ring is the expensive part, and identical on both).
        const mlfmm::PmlRingOperator ra(c.problem, tree, far_only(forced_aca()), 3.0);
        const mlfmm::PmlRingOperator re(c.problem, tree, far_only(exact), 3.0);
        const mlfmm::PmlRingFarStats& st = ra.far_stats();
        INFO("lossy exterior " << lossy_exterior << ": " << ra.describe());
        CHECK(st.region_dropped[1] > 0);
        CHECK(st.max_dropped_bound > 0.0);
        CHECK(st.max_dropped_bound <= st.drop_tolerance);
        if (lossy_exterior) {
            CHECK(st.region_dropped[0] > 0);
            CHECK(st.dropped_blocks > 0);
            CHECK(st.dropped_pairs > 0);
        } else {
            CHECK(st.region_dropped[0] == 0);  // vacuum: never dropped
            CHECK(st.dropped_blocks == 0);
        }
        Index dropped = 0, kept_r2 = 0;
        for (const mlfmm::PmlRingFarBlock& b : ra.far_blocks()) {
            if (!b.region[1])
                CHECK(b.bound[1] <= st.drop_tolerance);
            else
                ++kept_r2;
            if (b.region[1])
                CHECK(b.bound[1] > st.drop_tolerance);
            dropped += b.kind == mlfmm::PmlRingFarBlock::Kind::dropped ? 1 : 0;
        }
        CHECK(dropped == st.dropped_blocks);
        CHECK(kept_r2 > 0);  // the closest far blocks keep R2
        CHECK_THAT(ra.describe(), ContainsSubstring("dropped (magnitude bound"));
        // The bound is relative to the undamped entries: measure the difference against the far
        // ring of the same geometry with lossless (vacuum) regions.
        op::Problem undamped = c.problem;
        undamped.exterior = material::vacuum();
        undamped.object = material::vacuum();
        const mlfmm::PmlRingOperator ru(undamped, tree, far_only(exact), 3.0);
        Real err = 0;
        for (const std::uint64_t seed : {41u, 42u, 43u}) {
            const VectorXc x = random_vector(2 * c.space.size(), seed);
            err = std::max(err, (ra * x - re * x).norm() / (ru * x).norm());
        }
        INFO("far ring difference / undamped far ring " << err);
        CHECK(err <= 1e-4);
        CHECK(st.pair_reductions > 0);
        CHECK(st.drop_distance[1] < 1e300);
        CHECK((lossy_exterior ? st.drop_distance[0] < 1e300 : st.drop_distance[0] == 1e300));
    }
    // The bound: no decay without loss, decreasing in the distance and in the stretch.
    CHECK(mlfmm::pml_ring_decay_bound(0.0, 1e-6, 4.0) >= 1.0);
    CHECK(mlfmm::pml_ring_decay_bound(4e7, 0.0, 4.0) >= 1.0);
    const Real a = 4e7;
    CHECK(mlfmm::pml_ring_decay_bound(a, 2e-6, 4.0) < mlfmm::pml_ring_decay_bound(a, 1e-6, 4.0));
    CHECK(mlfmm::pml_ring_decay_bound(a, 1e-6, 1.0) < mlfmm::pml_ring_decay_bound(a, 1e-6, 4.0));
    // The premise of the bound: the complex distance of the stretch has Re R~ >= kappa R with
    // kappa = 2 / sqrt(4 + sigma_max^2), sampled over random pairs in and across the PML.
    Case s(aca_surface(), material::vacuum(), 4.0, formulation::Kind::PMCHWT);
    std::mt19937_64 rng(77);
    std::uniform_real_distribution<Real> u(-1.0e-6, 1.0e-6), w(-0.05e-6, 0.05e-6);
    const Real kappa = 2.0 / std::sqrt(4.0 + 16.0);
    Real worst = 1e300;
    for (int i = 0; i < 20000; ++i) {
        const Vec3 p(u(rng), u(rng), w(rng)), q(u(rng), u(rng), w(rng));
        const Vec3c d = s.pml.map(p) - s.pml.map(q);
        const Complex rt = std::sqrt(d(0) * d(0) + d(1) * d(1) + d(2) * d(2));
        worst = std::min(worst, rt.real() / (p - q).norm());
    }
    INFO("min Re R~ / R " << worst << ", kappa " << kappa);
    CHECK(worst >= kappa * (1.0 - 1e-12));
    CHECK(worst < 0.95);  // the stretch does shorten Re R~: the bound needs kappa
}

TEST_CASE("pml_ring: ACA path is thread-count independent in assembly and apply", "[pml_ring]") {
    Case c(tiny_rough_surface(), material::Material{Complex(2.25, 0.0), 1.0}, 1.0,
           formulation::Kind::PMCHWT);
    const Index n2 = 2 * c.space.size();
    const VectorXc x = random_vector(n2, 5);
    const mlfmm::Octree tree = ring_tree(c.space, 3);
#ifdef SPECKLEBEM_HAVE_OPENMP
    const int saved = omp_get_max_threads();
    const int most = std::max(saved, omp_get_num_procs());
    std::vector<VectorXc> ys;
    std::unique_ptr<mlfmm::PmlRingOperator> built_one;
    for (const int t : {1, most}) {  // assembly with 1 and all threads: identical factors
        omp_set_num_threads(t);
        auto r = std::make_unique<mlfmm::PmlRingOperator>(c.problem, tree, far_only(forced_aca()));
        ys.push_back(*r * x);
        if (t == 1)
            built_one = std::move(r);
    }
    for (const int t : {2, 3, most}) {  // apply with 2 / 3 / max threads
        omp_set_num_threads(t);
        ys.push_back(*built_one * x);
    }
    omp_set_num_threads(saved);
    REQUIRE(built_one->far_stats().aca_blocks > 0);
    for (std::size_t i = 1; i < ys.size(); ++i) {
        INFO("variant " << i);
        CHECK((ys[i].array() == ys[0].array()).all());
    }
#else
    const mlfmm::PmlRingOperator r1(c.problem, tree, far_only(forced_aca()));
    CHECK(r1.far_stats().aca_blocks > 0);
#endif
    // Aliasing and concurrent calls on the ACA path.
    const mlfmm::PmlRingOperator r(c.problem, tree, far_only(forced_aca()));
    const VectorXc y1 = r * x;
    VectorXc a = x;
    r.apply(a, a);
    CHECK((a.array() == y1.array()).all());
    std::vector<VectorXc> out(3);
    std::vector<std::thread> th;
    for (std::size_t i = 0; i < out.size(); ++i) th.emplace_back([&, i] { r.apply(x, out[i]); });
    for (auto& t : th) t.join();
    for (const VectorXc& o : out) CHECK((o.array() == y1.array()).all());
}

TEST_CASE("pml_ring: ACA budget bookkeeping and parameter errors", "[pml_ring]") {
    Case c(tiny_surface(), material::Material{Complex(2.25, 0.0), 1.0}, 1.0,
           formulation::Kind::PMCHWT);
    const mlfmm::Octree tree = ring_tree(c.space, 3);
    const std::vector<std::uint8_t> s = mlfmm::classify_stretched(c.space, c.pml);
    mlfmm::PmlRingParams aca = far_only(forced_aca());
    aca.eta = 4.5;  // corner leaf pairs admissible, the leaf pairs at distance 0 exact
    mlfmm::PmlRingParams exact;
    exact.use_aca = false;
    const mlfmm::PmlRingEstimate ea = mlfmm::estimate_pml_ring(tree, c.space, s, aca);
    const mlfmm::PmlRingEstimate ee = mlfmm::estimate_pml_ring(tree, c.space, s, exact);
    // The exact far bytes are the same upper bound on both paths; only the exact blocks are
    // known (and checked) before the ACA assembly.
    CHECK(ea.far_bytes == ee.far_bytes);
    CHECK(ee.far_exact_bytes == ee.far_bytes);
    CHECK(ea.far_exact_bytes < ea.far_bytes);
    CHECK(ea.far_aca_blocks > 0);
    CHECK(ea.far_exact_blocks > 0);
    CHECK(ee.far_aca_blocks == 0);
    // A budget between the two: the ACA path passes, the exact path is refused.
    mlfmm::PmlRingParams b_aca = aca, b_exact = exact;
    b_aca.max_far_bytes = b_exact.max_far_bytes = (ea.far_exact_bytes + ea.far_bytes) / 2;
    const mlfmm::PmlRingEstimate ba = mlfmm::estimate_pml_ring(tree, c.space, s, b_aca);
    const mlfmm::PmlRingEstimate be = mlfmm::estimate_pml_ring(tree, c.space, s, b_exact);
    CHECK_NOTHROW(mlfmm::check_pml_ring_budget(ba, b_aca));
    CHECK_THROWS_WITH(mlfmm::check_pml_ring_budget(be, b_exact), ContainsSubstring("would need"));
    b_aca.max_far_bytes = 1;
    CHECK_THROWS_WITH(
        mlfmm::check_pml_ring_budget(mlfmm::estimate_pml_ring(tree, c.space, s, b_aca), b_aca),
        ContainsSubstring("non-admissible"));
    CHECK_THROWS_WITH(mlfmm::PmlRingOperator(c.problem, tree, b_aca),
                      ContainsSubstring("max_far_bytes"));
    CHECK(mlfmm::warn_pml_ring_memory(ea, 1));
    CHECK(mlfmm::warn_pml_ring_memory(ea, ea.far_exact_bytes));  // + the near ring
    CHECK_FALSE(mlfmm::warn_pml_ring_memory(ea, mlfmm::pml_ring_peak_bytes(ea)));
    // After assembly: the blocks are the plan; the non-fallback exact blocks are the
    // pre-assembly exact part.
    const mlfmm::PmlRingOperator r(c.problem, tree, aca);
    const std::vector<mlfmm::PmlRingFarBlock> plan =
        mlfmm::plan_pml_ring_far(tree, c.space, s, aca);
    REQUIRE(plan.size() == r.far_blocks().size());
    for (std::size_t i = 0; i < plan.size(); ++i) {
        CHECK(plan[i].row_box == r.far_blocks()[i].row_box);
        CHECK(plan[i].box == r.far_blocks()[i].box);
        CHECK(plan[i].admissible == r.far_blocks()[i].admissible);
    }
    std::size_t exact_bytes = 0;
    for (const mlfmm::PmlRingFarBlock& b : r.far_blocks())
        if (b.kind == mlfmm::PmlRingFarBlock::Kind::exact)
            exact_bytes += 64 * sz(b.rows) * sz(b.count);
    const mlfmm::PmlRingFarStats& st = r.far_stats();
    REQUIRE(st.aca_blocks > 0);
    CHECK(exact_bytes <= ea.far_exact_bytes);
    CHECK(st.exact_bytes >= exact_bytes);
    CHECK(r.far_bytes() >= st.exact_bytes + st.aca_bytes);
    CHECK(r.memory_bytes() >= r.near_bytes() + r.far_bytes());
    CHECK(r.near_pairs() > 0);  // the estimate keeps the near pairs; assemble_near only skips them
    CHECK(r.near_operator().nonzeros() == 0);
    // A rank cap of 1 turns (almost) every admissible block into an exact fallback.
    mlfmm::PmlRingParams capped = aca;
    capped.max_rank = 1;
    const mlfmm::PmlRingOperator rc(c.problem, tree, capped);
    CHECK(rc.far_stats().fallback_blocks > 0);
    CHECK(rc.far_stats().fallback_blocks + rc.far_stats().aca_blocks ==
          st.aca_blocks + st.fallback_blocks);
    CHECK(ring_difference(rc, r, 2 * c.space.size()) <= 1e-4);
    // Parameter errors.
    mlfmm::PmlRingParams bad = aca;
    bad.eta = 0.0;
    CHECK_THROWS_AS(mlfmm::estimate_pml_ring(tree, c.space, s, bad), std::invalid_argument);
    bad = aca;
    bad.aca_epsilon = 1.0;
    CHECK_THROWS_AS(mlfmm::estimate_pml_ring(tree, c.space, s, bad), std::invalid_argument);
    bad = aca;
    bad.max_rank = -1;
    CHECK_THROWS_AS(mlfmm::PmlRingOperator(c.problem, tree, bad), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::PmlRingOperator(c.problem, tree, aca, 0.0), std::invalid_argument);
    // An explicit ACA tolerance overrides 10^-(d0+1); the drop threshold stays.
    mlfmm::PmlRingParams loose = aca;
    loose.aca_epsilon = 1e-2;
    const mlfmm::PmlRingOperator rl(c.problem, tree, loose, 3.0);
    CHECK(rl.far_stats().epsilon == 1e-2);
    CHECK(rl.far_stats().drop_tolerance == 1e-4);
    CHECK(rl.far_stats().rank_mean[0] <= st.rank_mean[0]);
}

// ---- WP-PML5b review round: near-ring flag, drop logging, exact blocks, leaf rows ----

namespace {

/// Captures log lines of at least the given level while alive (as test_open_surface.cpp).
class LogCapture {
public:
    explicit LogCapture(spdlog::level::level_enum level)
        : sink_(std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(256)),
          saved_level_(spdlog::default_logger()->level()) {
        sink_->set_level(level);
        spdlog::default_logger()->set_level(std::min(saved_level_, level));
        spdlog::default_logger()->sinks().push_back(sink_);
    }
    ~LogCapture() {
        auto& sinks = spdlog::default_logger()->sinks();
        sinks.erase(std::remove(sinks.begin(), sinks.end(), sink_), sinks.end());
        spdlog::default_logger()->set_level(saved_level_);
    }
    LogCapture(const LogCapture&) = delete;
    LogCapture& operator=(const LogCapture&) = delete;
    LogCapture(LogCapture&&) = delete;
    LogCapture& operator=(LogCapture&&) = delete;
    /// The captured lines containing `text`.
    [[nodiscard]] std::vector<std::string> lines(const std::string& text) const {
        std::vector<std::string> out;
        for (const std::string& l : sink_->last_formatted())
            if (l.find(text) != std::string::npos)
                out.push_back(l);
        return out;
    }

private:
    std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt> sink_;
    spdlog::level::level_enum saved_level_;
};

}  // namespace

TEST_CASE("pml_ring: assemble_near = false is refused outside a standalone ring", "[pml_ring]") {
    // Review W1: an MlfmmOperator (and a Simulation) without the near ring would solve another
    // system; the flag stays a diagnostic of a standalone PmlRingOperator.
    const auto c = tiny_case();
    mlfmm::MlfmmParams p = tree_params(3, 3.0);
    p.pml_ring.assemble_near = false;
    CHECK_THROWS_AS(MlfmmOperator(c->problem, p), std::invalid_argument);
    CHECK_THROWS_WITH(MlfmmOperator(c->problem, p), ContainsSubstring("assemble_near"));
    op::Problem plain = c->problem;  // refused without a stretch too (the flag is never valid)
    plain.stretch = nullptr;
    CHECK_THROWS_WITH(MlfmmOperator(plain, p), ContainsSubstring("assemble_near"));
    // The standalone ring accepts it (far ring only).
    const mlfmm::Octree tree = ring_tree(c->space, 3);
    const mlfmm::PmlRingOperator far(c->problem, tree, p.pml_ring, 3.0);
    CHECK(far.near_operator().nonzeros() == 0);
    CHECK(far.far_stored_pairs() > 0);
    // Simulation: refused at construction (check of the MLFMM parameters).
    SimulationConfig cfg;
    cfg.wavelength = kLambda;
    cfg.object = material::Material{Complex(2.25, 0.0), Complex(1.0, 0.0)};
    cfg.kernels = cheap_options();
    cfg.compression = "mlfmm";
    cfg.mlfmm.pml_ring.assemble_near = false;
    PmlTermination term;
    term.sigma_max = 1.0;
    term.dark_band_tolerance = 0.9;
    excitation::AngularSpectrumBeam::Params bp;
    bp.wavelength = kLambda;
    bp.waist_radius = 0.4e-6;
    bp.region_radius = 0.6e-6;
    const auto beam = std::make_shared<excitation::AngularSpectrumBeam>(bp);
    CHECK_THROWS_AS(Simulation(tiny_surface(), term, beam, cfg), std::invalid_argument);
    CHECK_THROWS_WITH(Simulation(tiny_surface(), term, beam, cfg),
                      ContainsSubstring("assemble_near"));
}

TEST_CASE("pml_ring: the per-triangle-pair drop is logged with its bound", "[pml_ring]") {
    // Review W2: the INFO summary also reports the per-pair reductions (pair_reductions) and the
    // drop distances; a lossy object (vacuum exterior: no block dropped whole) has reductions
    // only in R2. A lossless case logs no drop summary.
    const material::Material lossy{Complex(-30.0, -20.0), 1.0};
    Case c(tiny_surface(), lossy, 1.0, formulation::Kind::PMCHWT);
    const mlfmm::Octree tree = ring_tree(c.space, 3);
    std::vector<std::string> got;
    mlfmm::PmlRingFarStats st;
    {
        LogCapture log(spdlog::level::info);
        const mlfmm::PmlRingOperator ra(c.problem, tree, far_only(forced_aca()), 3.0);
        st = ra.far_stats();
        got = log.lines("triangle pairs reduced");
    }
    REQUIRE(st.pair_reductions > 0);
    CHECK(st.dropped_blocks == 0);
    CHECK(st.pair_skips <= st.pair_reductions);
    REQUIRE(got.size() == 1);
    const std::string& line = got.front();
    INFO(line);
    CHECK_THAT(line, ContainsSubstring("magnitude bound"));
    CHECK_THAT(line,
               ContainsSubstring("triangle pairs reduced " + std::to_string(st.pair_reductions) +
                                 " of " + std::to_string(st.pair_distinct + st.pair_skips)));
    CHECK_THAT(line, ContainsSubstring("R1 never"));
    CHECK_THAT(line, ContainsSubstring("bound <= 1.0e-04"));
    CHECK(st.drop_distance[1] < 1e300);
    // Lossless: nothing dropped, no summary line.
    Case u(tiny_surface(), material::Material{Complex(2.25, 0.0), 1.0}, 1.0,
           formulation::Kind::PMCHWT);
    LogCapture log(spdlog::level::info);
    const mlfmm::PmlRingOperator ru(u.problem, tree, far_only(forced_aca()), 3.0);
    CHECK(ru.far_stats().pair_reductions == 0);
    CHECK(log.lines("triangle pairs reduced").empty());
    CHECK_THAT(ru.describe(), ContainsSubstring("0 of "));
}

TEST_CASE("pml_ring: exact blocks of the ACA path equal the exact ring bitwise", "[pml_ring]") {
    // Review W4: with no admissible block (eta -> 0), leaf rows (row_clusters = false) and a
    // lossless object (no per-pair drop) the ACA path stores every far block exactly through
    // detail::BlockSource; its entries, the chunk partition and every summation order of
    // apply() are those of the exact ring of WP-PML5a, so the products are bitwise equal.
    for (const bool rough : {false, true}) {
        Case c(rough ? tiny_rough_surface() : tiny_surface(),
               material::Material{Complex(2.25, 0.0), 1.0}, 1.0, formulation::Kind::PMCHWT);
        const mlfmm::Octree tree = ring_tree(c.space, 3);
        mlfmm::PmlRingParams exact;
        exact.use_aca = false;
        mlfmm::PmlRingParams blocks;
        blocks.eta = 1e-12;  // nothing admissible
        blocks.row_clusters = false;
        const mlfmm::PmlRingOperator re(c.problem, tree, exact);
        const mlfmm::PmlRingOperator rb(c.problem, tree, blocks);
        const mlfmm::PmlRingFarStats& st = rb.far_stats();
        INFO((rough ? "rough" : "flat") << ": " << rb.describe());
        REQUIRE(st.aca);
        CHECK(st.aca_blocks == 0);
        CHECK(st.fallback_blocks == 0);
        CHECK(st.exact_blocks > 0);
        CHECK(st.pair_reductions == 0);
        CHECK(st.exact_pairs == re.far_stored_pairs());
        Real worst = 0;
        bool bitwise = true;
        for (const std::uint64_t seed : {41u, 42u, 43u}) {
            const VectorXc x = random_vector(2 * c.space.size(), seed);
            const VectorXc ye = re * x;
            const VectorXc yb = rb * x;
            worst = std::max(worst, rel(yb, ye));
            bitwise = bitwise && (yb.array() == ye.array()).all();
        }
        INFO("relative difference " << worst);
        CHECK(worst <= 1e-14);
        CHECK(bitwise);  // same entries and summation orders: bitwise (pml_ring.hpp)
    }
}

TEST_CASE("pml_ring: stored far blocks above the budget are warned about after assembly",
          "[pml_ring]") {
    // The budget bounds the blocks known to be exact before assembly; with an explicit max_rank
    // the ACA factors can end above it: one SBEM_WARN after the assembly, none within budget.
    Case c(tiny_surface(), material::Material{Complex(2.25, 0.0), 1.0}, 1.0,
           formulation::Kind::PMCHWT);
    const mlfmm::Octree tree = ring_tree(c.space, 3);
    mlfmm::PmlRingParams p = far_only(forced_aca());
    std::size_t stored = 0, exact_known = 0;
    {
        LogCapture log(spdlog::level::warn);
        const mlfmm::PmlRingOperator r(c.problem, tree, p, 3.0);
        stored = r.far_stats().exact_bytes + r.far_stats().aca_bytes;
        exact_known = r.estimate().far_exact_bytes;
        CHECK(log.lines("more than the far budget").empty());  // automatic budget: dense
    }
    REQUIRE(exact_known > 0);
    REQUIRE(stored > exact_known);
    p.max_far_bytes = exact_known;  // passes the pre-assembly check, below the stored blocks
    LogCapture log(spdlog::level::warn);
    const mlfmm::PmlRingOperator r(c.problem, tree, p, 3.0);
    const std::vector<std::string> got = log.lines("more than the far budget");
    REQUIRE(got.size() == 1);
    CHECK_THAT(got.front(), ContainsSubstring("max_far_bytes"));
}

TEST_CASE("pml_ring: no ACA attempt below the minimum break-even rank", "[pml_ring]") {
    // Review W3: under the automatic cap, admissible blocks with a break-even rank below
    // kPmlRingMinAcaRank are stored exactly without generating rows / columns first; with an
    // explicit max_rank every admissible block is attempted. aca_surface on 5 levels (leaves of
    // h = 125 nm) has admissible blocks with break-even ranks below 4 and above.
    Case c(aca_surface(), material::Material{Complex(2.25, 0.0), 1.0}, 1.0,
           formulation::Kind::PMCHWT);
    const mlfmm::Octree tree = ring_tree(c.space, 5);
    mlfmm::PmlRingParams p = far_only({});
    const mlfmm::PmlRingOperator r(c.problem, tree, p, 3.0);
    const mlfmm::PmlRingFarStats& st = r.far_stats();
    Index skipped = 0, attempted = 0;
    for (const mlfmm::PmlRingFarBlock& b : r.far_blocks()) {
        const Index be = b.rows * b.count / (b.rows + b.count);
        if (!b.admissible) {
            CHECK_FALSE(b.aca_skipped);
            continue;
        }
        if (be < mlfmm::kPmlRingMinAcaRank) {
            ++skipped;
            CHECK(b.aca_skipped);
            CHECK(b.kind == mlfmm::PmlRingFarBlock::Kind::fallback);
            CHECK(b.generated_rows == 0);
            CHECK(b.generated_columns == 0);
        } else {
            ++attempted;
            CHECK_FALSE(b.aca_skipped);
            CHECK(b.generated_rows > 0);
        }
    }
    INFO(r.describe());
    CHECK(skipped > 0);
    CHECK(attempted > 0);
    CHECK(st.skipped_attempts == skipped);
    CHECK(st.skipped_attempts <= st.fallback_blocks);
    CHECK_THAT(r.describe(),
               ContainsSubstring(std::to_string(skipped) + " without an ACA attempt"));
    // An explicit rank cap attempts every admissible block (the tiny surface: small blocks).
    Case t(tiny_surface(), material::Material{Complex(2.25, 0.0), 1.0}, 1.0,
           formulation::Kind::PMCHWT);
    const mlfmm::Octree tt = ring_tree(t.space, 3);
    mlfmm::PmlRingParams automatic_cap = far_only(forced_aca());
    automatic_cap.max_rank = 0;  // eta 20 with the break-even cap
    const mlfmm::PmlRingOperator rd(t.problem, tt, automatic_cap, 3.0);
    const mlfmm::PmlRingOperator rf(t.problem, tt, far_only(forced_aca()), 3.0);
    CHECK(rd.far_stats().skipped_attempts > 0);
    CHECK(rf.far_stats().skipped_attempts == 0);
    for (const mlfmm::PmlRingFarBlock& b : rf.far_blocks())
        if (b.admissible)
            CHECK(b.generated_rows > 0);
}

TEST_CASE("pml_ring: leaf-row ACA ring (row_clusters = false) meets the ring tolerance",
          "[pml_ring]") {
    // The brief's base design stays selectable: row leaves against coarse column clusters (and
    // leaf columns), each within 10^-(d0+1) of the exact far ring on the tiny rough case.
    Case c(tiny_rough_surface(), material::Material{Complex(2.25, 0.0), 1.0}, 1.0,
           formulation::Kind::PMCHWT);
    const Index n2 = 2 * c.space.size();
    const mlfmm::Octree tree = ring_tree(c.space, 3);
    mlfmm::PmlRingParams exact;
    exact.use_aca = false;
    const mlfmm::PmlRingOperator re(c.problem, tree, exact);
    for (const bool coarse : {true, false}) {
        mlfmm::PmlRingParams p = far_only(forced_aca());
        p.row_clusters = false;
        p.coarse_columns = coarse;
        for (const Real d0 : {3.0, 5.0}) {
            const mlfmm::PmlRingOperator ra(c.problem, tree, p, d0);
            const Real err = ring_difference_far(ra, re, n2);
            INFO("coarse columns " << coarse << ", d0 " << d0 << ": " << err << "; "
                                   << ra.describe());
            CHECK(ra.far_stats().aca_blocks > 0);
            for (const mlfmm::PmlRingFarBlock& b : ra.far_blocks())
                CHECK(b.row_level == tree.leaf_level());
            CHECK(err <= std::pow(10.0, -(d0 + 1.0)));
        }
    }
}

// ---- WP-PML5c: ring assembly cost (canonical-pair cache, determinism, diagonal, shared plan) ----

namespace {

/// True if two sparse matrices have the same structure and bitwise equal values.
bool bitwise_equal(const op::SparseOperator::Matrix& a, const op::SparseOperator::Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols() || a.nonZeros() != b.nonZeros())
        return false;
    const auto nnz = static_cast<std::size_t>(a.nonZeros());
    return std::equal(a.outerIndexPtr(), a.outerIndexPtr() + a.rows() + 1, b.outerIndexPtr()) &&
           std::equal(a.innerIndexPtr(), a.innerIndexPtr() + nnz, b.innerIndexPtr()) &&
           std::equal(a.valuePtr(), a.valuePtr() + nnz, b.valuePtr(),
                      [](const Complex& x, const Complex& y) {
                          return x.real() == y.real() && x.imag() == y.imag();
                      });
}

/// Stretched triangles that carry a basis function (each contributes one identical pair).
Index stretched_triangles_with_support(const Case& c) {
    Index n = 0;
    for (Index t = 0; t < c.om.mesh.num_triangles(); ++t)
        n += kernels::is_stretched_triangle(c.pml, c.om.mesh, t) && c.space.support(t).count > 0
                 ? 1
                 : 0;
    return n;
}

}  // namespace

TEST_CASE("pml_ring: canonical-pair cache and translation classes give the 5a near ring bitwise",
          "[pml_ring]") {
    // Flat (K = 0 on every stretched pair) and rough (K != 0 between tilted patch triangles and the
    // flat zones) tiny cases; the ring pattern of the octree and an asymmetric pattern (all rows x
    // a few columns, whose pairs are needed in one ordering only).
    for (const bool rough : {false, true}) {
        INFO((rough ? "rough" : "flat") << " case");
        Case c(rough ? tiny_rough_surface() : tiny_surface(),
               material::Material{Complex(2.25, -0.1), 1.0}, 1.0, formulation::Kind::ICTF);
        const mlfmm::Octree tree(c.space, kLambda, mlfmm::OctreeParams{1, 3, 0.0});
        const mlfmm::PmlRingPlan plan = mlfmm::plan_pml_ring(tree, c.space, c.pml);
        op::SparseStretchedStats s_new, s_old, s_tr;
        op::SparseStretchedOptions o_new, o_old, o_tr;
        o_new.stats = &s_new;
        o_new.evaluation = op::StretchedPairEvaluation::canonical_cache;
        o_old.stats = &s_old;
        o_old.evaluation = op::StretchedPairEvaluation::per_ordering;
        o_tr.stats = &s_tr;  // the default: translation classes
        const auto Zn = op::assemble_sparse_stretched(c.problem, plan.split.ring, o_new);
        const auto Zo = op::assemble_sparse_stretched(c.problem, plan.split.ring, o_old);
        const auto Zt = op::assemble_sparse_stretched(c.problem, plan.split.ring, o_tr);
        CHECK(bitwise_equal(Zn->matrix(), Zo->matrix()));
        // Translation classes (the default): the PML strips repeat their touching pairs along the
        // strip; one evaluation per class, the copies bitwise.
        CHECK(s_tr.translation_classes > 0);
        CHECK(s_tr.translated_pairs > 0);
        // Identical and corner pairs are evaluated themselves.
        CHECK(s_tr.translated_pairs + stretched_triangles_with_support(c) < s_tr.touching_pairs);
        CHECK(s_tr.canonical_pairs == s_new.canonical_pairs);
        CHECK(s_new.translated_pairs == 0);
        CHECK(bitwise_equal(Zt->matrix(), Zo->matrix()));
        // Same scatter; the cache evaluates each unordered stretched pair once: the ring pattern
        // is symmetric, so every pair but the identical ones is needed in both orderings.
        CHECK(s_new.ordered_pairs == s_old.ordered_pairs);
        CHECK(s_new.stretched_pairs == s_old.stretched_pairs);
        CHECK(s_old.canonical_pairs == 0);
        CHECK(s_old.cache_bytes == 0);
        const Index identical = stretched_triangles_with_support(c);
        CHECK(identical > 0);
        CHECK(s_new.stretched_pairs == 2 * s_new.canonical_pairs - identical);
        CHECK(s_new.touching_pairs > identical);
        CHECK(s_new.touching_pairs < s_new.canonical_pairs);
        CHECK(s_new.cache_bytes >= static_cast<std::size_t>(s_new.canonical_pairs) * 576);
        // Asymmetric pattern: every row x four columns (two stretched, two real).
        std::vector<Index> cols;
        for (const std::uint8_t want : {std::uint8_t{1}, std::uint8_t{0}}) {
            std::vector<Index> of;
            for (Index n = 0; n < c.space.size(); ++n)
                if (plan.stretched[sz(n)] == want)
                    of.push_back(n);
            REQUIRE(of.size() >= 2);
            cols.push_back(of[of.size() / 4]);
            cols.push_back(of[(3 * of.size()) / 4]);
        }
        std::sort(cols.begin(), cols.end());
        op::BasisPattern pat;
        pat.group_of_row.assign(sz(c.space.size()), 0);
        pat.cols = cols;
        pat.col_ptr = {0, static_cast<Index>(cols.size())};
        const auto An = op::assemble_sparse_stretched(c.problem, pat, o_new);
        const auto Ao = op::assemble_sparse_stretched(c.problem, pat, o_old);
        CHECK(bitwise_equal(An->matrix(), Ao->matrix()));
        CHECK(bitwise_equal(op::assemble_sparse_stretched(c.problem, pat)->matrix(), Ao->matrix()));
        // The ring operator uses the default evaluation: its near part is Zt.
        const mlfmm::PmlRingOperator ring(c.problem, tree);
        CHECK(bitwise_equal(ring.near_operator().matrix(), Zt->matrix()));
        CHECK(ring.near_stats().canonical_pairs == s_tr.canonical_pairs);
        CHECK(ring.near_stats().translated_pairs == s_tr.translated_pairs);
    }
}

TEST_CASE("pml_ring: the near-ring assembly is thread-count independent", "[pml_ring]") {
#ifdef SPECKLEBEM_HAVE_OPENMP
    Case c(tiny_rough_surface(), material::Material{Complex(2.25, -0.1), 1.0}, 1.0,
           formulation::Kind::PMCHWT);
    const mlfmm::Octree tree(c.space, kLambda, mlfmm::OctreeParams{1, 3, 0.0});
    const mlfmm::PmlRingPlan plan = mlfmm::plan_pml_ring(tree, c.space, c.pml);
    const int saved = omp_get_max_threads();
    const int most = std::max(saved, omp_get_num_procs());
    std::vector<std::shared_ptr<op::SparseOperator>> Z;
    for (const int t : {1, 3, most}) {
        omp_set_num_threads(t);
        Z.push_back(op::assemble_sparse_stretched(c.problem, plan.split.ring));
    }
    omp_set_num_threads(saved);
    for (std::size_t i = 1; i < Z.size(); ++i) {
        INFO("thread count index " << i);
        CHECK(bitwise_equal(Z[i]->matrix(), Z[0]->matrix()));
    }
#else
    SUCCEED("built without OpenMP");
#endif
}

TEST_CASE("pml_ring: the diagonal from the near parts equals op::assemble_diagonal", "[pml_ring]") {
    for (const bool rough : {false, true}) {
        INFO((rough ? "rough" : "flat") << " case");
        Case c(rough ? tiny_rough_surface() : tiny_surface(),
               material::Material{Complex(2.25, -0.1), 1.0}, 1.0, formulation::Kind::ICTF);
        const Index N = c.space.size();
        // With the stretch: real rows bitwise, stretched rows to rounding.
        const MlfmmOperator Z(c.problem, tree_params(3, 3.0));
        const VectorXc d = Z.diagonal();
        const VectorXc ref = op::assemble_diagonal(c.problem);
        REQUIRE(d.size() == 2 * N);
        const std::vector<std::uint8_t>& s = Z.ring_operator()->stretched();
        Real worst = 0;
        bool real_bitwise = true;
        for (Index r = 0; r < 2 * N; ++r) {
            if (s[sz(r % N)] != 0)
                worst = std::max(worst, std::abs(d(r) - ref(r)) / std::abs(ref(r)));
            else
                real_bitwise = real_bitwise && d(r) == ref(r);
        }
        INFO("stretched rows: max relative difference " << worst);  // measured 1.7e-16
        CHECK(real_bitwise);
        CHECK(worst <= 1e-14);
        // Without a stretch (same mesh): bitwise.
        op::Problem plain = c.problem;
        plain.stretch = nullptr;
        const MlfmmOperator Z0(plain, tree_params(3, 3.0));
        CHECK((Z0.diagonal().array() == op::assemble_diagonal(plain).array()).all());
    }
}

TEST_CASE("pml_ring: the shared plan gives the same ring as the self-planned constructor",
          "[pml_ring]") {
    const auto c = tiny_case();
    const mlfmm::Octree tree(c->space, kLambda, mlfmm::OctreeParams{1, 3, 0.0});
    const mlfmm::PmlRingPlan plan = mlfmm::plan_pml_ring(tree, c->space, c->pml);
    // The plan is exactly the three separate computations.
    const std::vector<std::uint8_t> s = mlfmm::classify_stretched(c->space, c->pml);
    const mlfmm::SplitNearPattern split = mlfmm::split_near_pattern(tree, c->space, s);
    const mlfmm::PmlRingEstimate est = mlfmm::estimate_pml_ring(tree, c->space, s);
    CHECK(plan.stretched == s);
    CHECK(plan.split.ring.cols == split.ring.cols);
    CHECK(plan.split.ring.col_ptr == split.ring.col_ptr);
    CHECK(plan.split.ring.group_of_row == split.ring.group_of_row);
    CHECK(plan.split.real.cols == split.real.cols);
    CHECK(plan.split.ring_pairs == est.near_pairs);
    CHECK(plan.split.real_pairs == est.rr_near_pairs);
    CHECK(plan.estimate.far_bytes == est.far_bytes);
    CHECK(plan.estimate.far_stored == est.far_stored);
    // ... and the ACA far partition of WP-PML5b from the same pass as the estimate.
    CHECK(plan.estimate.far_exact_bytes == est.far_exact_bytes);
    CHECK(plan.estimate.far_aca_blocks == est.far_aca_blocks);
    CHECK(plan.estimate.far_exact_blocks == est.far_exact_blocks);
    CHECK(plan.estimate.far_aca_pairs == est.far_aca_pairs);
    const std::vector<mlfmm::PmlRingFarBlock> far = mlfmm::plan_pml_ring_far(tree, c->space, s);
    REQUIRE(plan.far_blocks.size() == far.size());
    REQUIRE(!far.empty());
    for (std::size_t i = 0; i < far.size(); ++i) {
        CHECK(plan.far_blocks[i].row_box == far[i].row_box);
        CHECK(plan.far_blocks[i].box == far[i].box);
        CHECK(plan.far_blocks[i].rows == far[i].rows);
        CHECK(plan.far_blocks[i].admissible == far[i].admissible);
        CHECK(plan.far_blocks[i].distance == far[i].distance);
    }
    // A ring from the plan (as MlfmmOperator builds it, real pattern released) equals the
    // self-planned one bitwise.
    const mlfmm::PmlRingOperator self(c->problem, tree);
    mlfmm::PmlRingPlan moved = plan;
    moved.split.real = op::BasisPattern{};
    const mlfmm::PmlRingOperator shared(c->problem, tree, std::move(moved));
    CHECK(bitwise_equal(shared.near_operator().matrix(), self.near_operator().matrix()));
    CHECK(shared.near_pairs() == self.near_pairs());
    CHECK(shared.far_pairs() == self.far_pairs());
    CHECK(shared.far_bytes() == self.far_bytes());
    CHECK(shared.estimate().budget == self.estimate().budget);
    CHECK(shared.far_blocks().size() == self.far_blocks().size());
    CHECK(shared.far_stats().aca_blocks == self.far_stats().aca_blocks);
    CHECK(shared.far_stats().rank_max == self.far_stats().rank_max);
    const VectorXc x = random_vector(2 * c->space.size(), 23);
    CHECK(((shared * x).array() == (self * x).array()).all());
    // The same with the exact far ring (no partition in the plan) and with d0 = 5.
    mlfmm::PmlRingParams no_aca;
    no_aca.use_aca = false;
    const mlfmm::PmlRingPlan plan_exact = mlfmm::plan_pml_ring(tree, c->space, c->pml, no_aca);
    CHECK(plan_exact.far_blocks.empty());
    CHECK(plan_exact.estimate.far_exact_bytes == plan_exact.estimate.far_bytes);
    const mlfmm::PmlRingOperator self_exact(c->problem, tree, no_aca, 5.0);
    const mlfmm::PmlRingOperator shared_exact(c->problem, tree, plan_exact, no_aca, 5.0);
    CHECK_FALSE(shared_exact.far_stats().aca);
    CHECK(((shared_exact * x).array() == (self_exact * x).array()).all());
    const mlfmm::PmlRingOperator self5(c->problem, tree, {}, 5.0);
    const mlfmm::PmlRingOperator shared5(c->problem, tree, plan, {}, 5.0);
    CHECK(shared5.far_stats().epsilon == 1e-6);
    CHECK(((shared5 * x).array() == (self5 * x).array()).all());
    // A plan made with other far-partition parameters is refused (its partition and estimate
    // would not match); the budget and the ACA tolerance may differ.
    CHECK_THROWS_WITH(mlfmm::PmlRingOperator(c->problem, tree, plan, no_aca),
                      ContainsSubstring("far-ring partition parameters"));
    CHECK_THROWS_WITH(mlfmm::PmlRingOperator(c->problem, tree, plan_exact),
                      ContainsSubstring("far-ring partition parameters"));
    mlfmm::PmlRingParams other_eta;
    other_eta.eta = 1.0;
    CHECK_THROWS_AS(mlfmm::PmlRingOperator(c->problem, tree, plan, other_eta),
                    std::invalid_argument);
    mlfmm::PmlRingParams loose;
    loose.aca_epsilon = 1e-2;
    CHECK_NOTHROW(mlfmm::PmlRingOperator(c->problem, tree, plan, loose));
    CHECK_THROWS_AS(mlfmm::PmlRingOperator(c->problem, tree, plan, {}, 0.0), std::invalid_argument);
    // A far partition that does not match the tree's boxes is refused.
    mlfmm::PmlRingPlan bad_far = plan;
    bad_far.far_blocks.front().rows += 1;
    CHECK_THROWS_WITH(mlfmm::PmlRingOperator(c->problem, tree, std::move(bad_far)),
                      ContainsSubstring("row count"));
    // MlfmmOperator shares its plan: the ring's counts are the plan's.
    const MlfmmOperator Z(c->problem, tree_params(3, 3.0));
    const mlfmm::PmlRingPlan zp = mlfmm::plan_pml_ring(Z.octree(), c->space, c->pml);
    CHECK(Z.ring_operator()->near_pairs() == zp.split.ring_pairs);
    CHECK(Z.near_operator().nonzeros() == 4 * zp.split.real_pairs);
    // A plan of another stretch is refused; the budget is resolved from the given params.
    mlfmm::PmlRingPlan wrong = plan;
    wrong.stretched.assign(wrong.stretched.size(), 0);
    CHECK_THROWS_WITH(mlfmm::PmlRingOperator(c->problem, tree, std::move(wrong)),
                      ContainsSubstring("PmlRingPlan"));
    mlfmm::PmlRingParams tight;
    tight.max_far_bytes = 1;
    CHECK_THROWS_WITH(mlfmm::PmlRingOperator(c->problem, tree, plan, tight),
                      ContainsSubstring("max_far_bytes"));
}

TEST_CASE("pml_ring: a plan made on another octree is refused", "[pml_ring]") {
    // Review W3: same space and stretch (the classification and N agree), other tree.
    const auto c = tiny_case();
    const mlfmm::Octree tree3(c->space, kLambda, mlfmm::OctreeParams{1, 3, 0.0});
    const mlfmm::Octree tree3b(c->space, kLambda, mlfmm::OctreeParams{1, 3, 0.0});
    const mlfmm::Octree tree2(c->space, kLambda, mlfmm::OctreeParams{1, 2, 0.0});
    const mlfmm::PmlRingPlan plan = mlfmm::plan_pml_ring(tree3, c->space, c->pml);
    CHECK(plan.tree == mlfmm::pml_ring_tree_fingerprint(tree3));
    CHECK(mlfmm::pml_ring_tree_fingerprint(tree3b) == mlfmm::pml_ring_tree_fingerprint(tree3));
    const mlfmm::PmlRingTreeFingerprint f2 = mlfmm::pml_ring_tree_fingerprint(tree2);
    CHECK(f2.levels == 2);
    CHECK(f2.leaves < plan.tree.leaves);
    CHECK_FALSE(f2 == plan.tree);
    CHECK_THROWS_AS(mlfmm::PmlRingOperator(c->problem, tree2, plan), std::invalid_argument);
    CHECK_THROWS_WITH(mlfmm::PmlRingOperator(c->problem, tree2, plan),
                      ContainsSubstring("another octree"));
    // Equal level and leaf counts but another permutation (hash only).
    mlfmm::PmlRingPlan forged = plan;
    forged.tree.hash ^= 1;
    CHECK_THROWS_WITH(mlfmm::PmlRingOperator(c->problem, tree3, std::move(forged)),
                      ContainsSubstring("another octree"));
    CHECK_NOTHROW(mlfmm::PmlRingOperator(c->problem, tree3b, plan));
}

// ---- WP-PML5c review: completeness of the translation-class key ----

namespace {

/// Region parameters of a Problem (as op::make_setup: exterior, object).
std::array<kernels::RegionParams, 2> region_params(const op::Problem& p) {
    std::array<kernels::RegionParams, 2> r{};
    const std::array<const material::Material*, 2> m = {&p.exterior, &p.object};
    for (std::size_t i = 0; i < 2; ++i)
        r[i] = {m[i]->wavenumber(p.omega), m[i]->wave_impedance(p.omega), p.omega,
                constants::eps0 * m[i]->eps_r, constants::mu0 * m[i]->mu_r};
    return r;
}

/// Non-identical pairs t < s sharing a vertex index with a stretched triangle.
std::vector<std::array<Index, 2>> stretched_touching_pairs(const Case& c) {
    const Triangles& T = c.om.mesh.triangles();
    std::vector<std::array<Index, 2>> out;
    for (Index t = 0; t < T.rows(); ++t) {
        for (Index s = t + 1; s < T.rows(); ++s) {
            bool shared = false;
            for (Eigen::Index i = 0; i < 3; ++i)
                for (Eigen::Index j = 0; j < 3; ++j) shared = shared || T(t, i) == T(s, j);
            if (shared && (kernels::is_stretched_triangle(c.pml, c.om.mesh, t) ||
                           kernels::is_stretched_triangle(c.pml, c.om.mesh, s)))
                out.push_back({t, s});
        }
    }
    return out;
}

}  // namespace

TEST_CASE(
    "pml_ring: translation classes stay bitwise with vertices on the strip boundary (m = 0.5)",
    "[pml_ring]") {
    // Review W1b: the PML starts exactly on a grid line (the start is a vertex coordinate, bit for
    // bit) and the profile order m = 0.5 makes sigma visible one ulp beyond the start
    // (sigma_max (ulp / W)^0.5 ~ 1e-8). Pairs with a vertex on (or within the margin of) the strip
    // boundary must not be copies of interior pairs.
    const OpenSurfaceMesh mesh = tiny_surface();
    const Vertices& V = mesh.mesh.vertices();
    Real start = V(0, 1);  // the vertex coordinate closest to pml_start (x and y share the lines)
    for (Eigen::Index i = 0; i < V.rows(); ++i) {
        if (std::abs(V(i, 1) - mesh.pml_start) < std::abs(start - mesh.pml_start))
            start = V(i, 1);
    }
    REQUIRE(std::abs(start - mesh.pml_start) < 1e-3 * (mesh.outer_half - mesh.pml_start));
    Index on_boundary = 0;
    for (Eigen::Index i = 0; i < V.rows(); ++i)
        on_boundary += std::abs(V(i, 0)) == start || std::abs(V(i, 1)) == start ? 1 : 0;
    REQUIRE(on_boundary > 0);
    // The start on the grid line and one ulp beyond it (the boundary vertices then lie one ulp
    // inside the strip, within the margin).
    for (const Real s0 : {start, std::nextafter(start, 1.0)}) {
        INFO("start " << (s0 == start ? "on the grid line" : "one ulp beyond"));
        const geometry::StretchProfile prof{s0, mesh.outer_half - s0, 2.0, 0.5};
        Case c(mesh, material::Material{Complex(2.25, -0.5), 1.0}, formulation::Kind::PMCHWT,
               CoordinateStretch(prof, prof));
        // A keyed pair keeps the margin from the boundary of its strip (|u| = start along the
        // translation axis u); pairs the 5c rule (|u| <= start) grouped but that touch the
        // boundary have no key.
        const std::array<kernels::RegionParams, 2> reg = region_params(c.problem);
        const std::array<bool, 2> active = {true, true};
        kernels::StretchedTouchingKey key;
        Index keyed = 0, boundary = 0;
        for (const auto& [t, s] : stretched_touching_pairs(c)) {
            std::array<Real, 2> umax{};  // max |x|, |y| over the six vertices
            for (const Index tri : {t, s})
                for (Eigen::Index i = 0; i < 3; ++i)
                    for (std::size_t d = 0; d < 2; ++d)
                        umax[d] = std::max(umax[d], std::abs(V(c.om.mesh.triangles()(tri, i),
                                                               static_cast<Eigen::Index>(d))));
            const bool has = kernels::stretched_touching_key(c.space, t, s, reg, active,
                                                             c.problem.kernel_options, c.pml, key);
            if (has) {
                ++keyed;
                CHECK(umax[sz(key.axis)] <= s0 - 1e-12 * s0);
            } else if ((umax[1] <= s0 || umax[0] <= s0) && c.space.support(t).count > 0 &&
                       c.space.support(s).count > 0) {
                ++boundary;  // grouped by the 5c rule, on (or one ulp inside) the boundary
                CHECK((umax[1] == start || umax[0] == start));
            }
        }
        CHECK(boundary > 0);
        CHECK(keyed > 0);
        if (s0 != start)
            continue;  // the assembly once (unit-test budget)
        // The class result is bitwise the own evaluation of every pair (canonical cache: bitwise
        // the per-ordering 5a path, tested above).
        const mlfmm::Octree tree(c.space, kLambda, mlfmm::OctreeParams{1, 3, 0.0});
        const mlfmm::PmlRingPlan plan = mlfmm::plan_pml_ring(tree, c.space, c.pml);
        op::SparseStretchedStats s_tr;
        op::SparseStretchedOptions o_tr, o_ref;
        o_tr.stats = &s_tr;
        o_ref.evaluation = op::StretchedPairEvaluation::canonical_cache;
        const auto Zt = op::assemble_sparse_stretched(c.problem, plan.split.ring, o_tr);
        const auto Zr = op::assemble_sparse_stretched(c.problem, plan.split.ring, o_ref);
        INFO("classes " << s_tr.translation_classes << ", copies " << s_tr.translated_pairs);
        CHECK(s_tr.translated_pairs > 0);
        CHECK(bitwise_equal(Zt->matrix(), Zr->matrix()));
    }
}

TEST_CASE("pml_ring: the translation key holds the fastmath decision and the shared-index pattern",
          "[pml_ring]") {
    // Review W1a: the fastmath choice is decided from absolute coordinates (not translation-exact),
    // so the key holds the decision. Every keyed coplanar pair of the tiny flat case: the same pair
    // with fast_plain_kernel off has another key; translates with equal keys have bitwise equal
    // blocks after the slot mapping.
    const auto c = tiny_case();
    const std::array<kernels::RegionParams, 2> reg = region_params(c->problem);
    const std::array<bool, 2> active = {true, true};
    kernels::OperatorOptions slow = c->problem.kernel_options;
    slow.fast_plain_kernel = false;
    kernels::StretchedTouchingKey k_fast, k_slow;
    std::map<std::vector<std::uint64_t>, std::array<Index, 2>> seen;
    Index keyed = 0, compared = 0;
    for (const auto& [t, s] : stretched_touching_pairs(*c)) {
        const bool has = kernels::stretched_touching_key(c->space, t, s, reg, active,
                                                         c->problem.kernel_options, c->pml, k_fast);
        REQUIRE(kernels::stretched_touching_key(c->space, t, s, reg, active, slow, c->pml,
                                                k_slow) == has);
        if (!has)
            continue;
        ++keyed;
        CHECK(k_fast.axis == k_slow.axis);
        CHECK(k_fast.words.size() == k_slow.words.size());
        CHECK(k_fast.words != k_slow.words);  // flat mesh: every touching pair coplanar (K = 0)
        const auto [it, inserted] = seen.emplace(k_fast.words, std::array<Index, 2>{t, s});
        if (inserted || compared >= 4)
            continue;
        // A class member: its blocks equal the representative's with the slots mapped.
        ++compared;
        kernels::StretchedTouchingKey k_rep;
        const auto [tr, sr] = it->second;
        REQUIRE(kernels::stretched_touching_key(c->space, tr, sr, reg, active,
                                                c->problem.kernel_options, c->pml, k_rep));
        std::array<Eigen::Matrix<Complex, 3, 3>, 2> L, K, Lr, Kr;
        kernels::element_blocks_both_regions(c->space, t, s, reg, active, c->problem.kernel_options,
                                             &c->pml, L, K);
        kernels::element_blocks_both_regions(c->space, tr, sr, reg, active,
                                             c->problem.kernel_options, &c->pml, Lr, Kr);
        const auto slot_of = [](const std::array<int, 3>& m, int j) {
            for (int x = 0; x < 3; ++x)
                if (m[static_cast<std::size_t>(x)] == j)
                    return x;
            return -1;
        };
        bool equal = true;
        for (int a = 0; a < c->space.support(t).count; ++a) {
            for (int b = 0; b < c->space.support(s).count; ++b) {
                const int ar = slot_of(k_rep.slot_vertex[0], k_fast.slot_vertex[0][sz(a)]);
                const int br = slot_of(k_rep.slot_vertex[1], k_fast.slot_vertex[1][sz(b)]);
                REQUIRE(ar >= 0);
                REQUIRE(br >= 0);
                for (std::size_t i = 0; i < 2; ++i)
                    equal = equal && L[i](a, b) == Lr[i](ar, br) && K[i](a, b) == Kr[i](ar, br);
            }
        }
        CHECK(equal);
    }
    CHECK(keyed > 0);
    CHECK(compared > 0);
    CHECK(seen.size() < static_cast<std::size_t>(keyed));

    // Review W1c: equal coordinates imply an equal shared-index pattern only without duplicate
    // vertices. Two meshes with the same coordinates (local order) where t and s overlap along
    // the segment AB but share by index only A (mesh P, B duplicated) or only B (mesh Q, A
    // duplicated): both shared-vertex pairs, different keys.
    const Real h = 100e-9;
    const auto build = [&](bool share_a) {
        Vertices V(7, 3);
        V << 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.5, 1.0, 0.0, 1.5, -1.0, 0.0, 0.5, 1.0, 0.0, 0.5, -1.0,
            0.0, share_a ? 2.0 : 1.0, 0.0, 0.0;
        V *= h;
        Triangles T(4, 3);
        const Index A = 0, B = 1, C = 2, D = 3, E = 4, G = 5, X = 6;  // X: the duplicate
        if (share_a)
            T << A, B, C, X, A, D, A, C, E, D, A, G;  // s = (B', A, D), its neighbour across A D
        else
            T << A, B, C, B, X, D, A, C, E, D, X, G;  // s = (B, A', D), its neighbour across A' D
        return geometry::TriangleMesh(V, T);
    };
    const geometry::StretchProfile px{0.5 * h, 4.0 * h, 1.0, 2.0};
    const geometry::StretchProfile py{0.0, 1.0, 0.0, 2.0};  // no stretch along y
    const CoordinateStretch strip(px, py);
    std::array<std::vector<std::uint64_t>, 2> words;
    for (const bool share_a : {true, false}) {
        INFO("mesh " << (share_a ? "P (shares A)" : "Q (shares B)"));
        const geometry::TriangleMesh m = build(share_a);
        const basis::RwgSpace space(m);
        REQUIRE(space.size() == 2);
        REQUIRE(kernels::classify(m, 0, 1, 1.0) == kernels::Proximity::shared_vertex);
        kernels::StretchedTouchingKey k;
        REQUIRE(kernels::stretched_touching_key(space, 0, 1, reg, active, c->problem.kernel_options,
                                                strip, k));
        CHECK(k.axis == 1);
        words[share_a ? 0 : 1] = k.words;
    }
    CHECK(words[0] != words[1]);
    CHECK(words[0].size() == words[1].size());
}

// ---------------------------------------------------------------------------------------------
// WP-PML5d: triangle-pair memo of the ACA far ring.
// ---------------------------------------------------------------------------------------------

namespace {

/// The applied far ring as a matrix: column k = ring e_k (bitwise comparisons of the stored
/// blocks through apply(); a far-only ring).
/// Applied with one thread (apply() is thread-count independent, unit test above; hundreds of
/// tiny parallel regions are slow on a loaded machine).
std::vector<VectorXc> applied_columns(const mlfmm::PmlRingOperator& r, Index n2) {
#ifdef SPECKLEBEM_HAVE_OPENMP
    const int saved = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    std::vector<VectorXc> cols;
    cols.reserve(sz(n2));
    for (Index k = 0; k < n2; ++k) {
        VectorXc e = VectorXc::Zero(n2);
        e(k) = Complex(1.0, 0.0);
        cols.push_back(r * e);
    }
#ifdef SPECKLEBEM_HAVE_OPENMP
    omp_set_num_threads(saved);
#endif
    return cols;
}

bool same_columns(const std::vector<VectorXc>& a, const std::vector<VectorXc>& b) {
    if (a.size() != b.size())
        return false;
    for (std::size_t k = 0; k < a.size(); ++k)
        if (!(a[k].array() == b[k].array()).all())
            return false;
    return true;
}

/// Block metadata that the memo must not change (everything but the evaluation counters).
bool same_blocks(const std::vector<mlfmm::PmlRingFarBlock>& a,
                 const std::vector<mlfmm::PmlRingFarBlock>& b) {
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const mlfmm::PmlRingFarBlock& x = a[i];
        const mlfmm::PmlRingFarBlock& y = b[i];
        if (x.row_box != y.row_box || x.box != y.box || x.kind != y.kind || x.rank != y.rank ||
            x.region != y.region || x.generated_rows != y.generated_rows ||
            x.generated_columns != y.generated_columns || x.aca_skipped != y.aca_skipped ||
            x.pair_distinct != y.pair_distinct || x.pair_reductions != y.pair_reductions ||
            x.pair_skips != y.pair_skips)
            return false;
    }
    return true;
}

}  // namespace

TEST_CASE("pml_ring: the triangle-pair memo leaves the stored far blocks bitwise unchanged",
          "[pml_ring]") {
    // WP-PML5d: the far ring with memo budgets 0 (no memo: the evaluation without the memo),
    // small (strips refused) and unlimited, on the tiny flat and rough surfaces and a lossy
    // object and exterior (per-pair drops, skips), with forced ACA (low-rank blocks) and the
    // default parameters (rank-cap fallbacks and blocks without an ACA attempt: full() after the
    // attempt): the applied far ring e_k for every k (all stored ACA and exact entries) is
    // bitwise the same, so are the ranks, kinds and generated rows / columns. Every kernel
    // evaluation saved is a memo hit; unlimited, each distinct pair is evaluated once.
    struct Variant {
        const char* name;
        bool rough;
        Complex eps;
    };
    const std::array<Variant, 3> variants = {{
        {"flat", false, Complex(2.25, 0.0)},
        {"rough", true, Complex(2.25, 0.0)},
        {"flat lossy", false, Complex(-30.0, -20.0)},
    }};
    constexpr std::size_t kSmall = 30000;  // a few strips of the tiny blocks (584 B per pair)
    Index limited = 0, saved = 0, skips = 0, fallbacks = 0;
    for (const Variant& v : variants) {
        Case c(v.rough ? tiny_rough_surface() : tiny_surface(), material::Material{v.eps, 1.0}, 1.0,
               formulation::Kind::PMCHWT);
        if (v.eps.imag() < 0.0)  // both regions damped: pairs with no region left exist
            c.problem.exterior = material::Material{v.eps, 1.0};
        const Index n2 = 2 * c.space.size();
        const mlfmm::Octree tree = ring_tree(c.space, 3);
        for (const bool forced : {true, false}) {
            mlfmm::PmlRingParams p = far_only(forced ? forced_aca() : mlfmm::PmlRingParams{});
            p.pair_memo_bytes = 0;
            const mlfmm::PmlRingOperator base(c.problem, tree, p);
            const mlfmm::PmlRingFarStats& bs = base.far_stats();
            const std::vector<VectorXc> cb = applied_columns(base, n2);
            CHECK(bs.pair_memo_hits == 0);
            CHECK(bs.memo_bytes_max == 0);
            CHECK(bs.memo_limited_blocks == 0);
            CHECK(bs.pair_evaluations >= bs.pair_distinct);
            if (forced)
                CHECK(bs.aca_blocks > 0);
            fallbacks += bs.fallback_blocks;
            skips += bs.pair_skips;
            for (const std::size_t budget : {kSmall, std::numeric_limits<std::size_t>::max()}) {
                p.pair_memo_bytes = budget;
                const mlfmm::PmlRingOperator r(c.problem, tree, p);
                const mlfmm::PmlRingFarStats& st = r.far_stats();
                INFO(v.name << (forced ? ", forced ACA" : ", default") << ", budget " << budget
                            << ": " << r.describe());
                CHECK(same_columns(applied_columns(r, n2), cb));
                CHECK(same_blocks(r.far_blocks(), base.far_blocks()));
                CHECK(st.aca_bytes == bs.aca_bytes);
                CHECK(st.exact_bytes == bs.exact_bytes);
                CHECK(st.memo_budget == budget);
                CHECK(st.pair_evaluations + st.pair_memo_hits == bs.pair_evaluations);
                CHECK(st.pair_distinct == bs.pair_distinct);
                CHECK(st.pair_reductions == bs.pair_reductions);
                CHECK(st.pair_skips == bs.pair_skips);
                CHECK(st.memo_bytes_max <= budget);
                if (budget == kSmall) {
                    limited += st.memo_limited_blocks;
                } else {
                    CHECK(st.memo_limited_blocks == 0);
                    CHECK(st.pair_evaluations == st.pair_distinct);
                    saved += bs.pair_evaluations - st.pair_evaluations;
                }
                for (const mlfmm::PmlRingFarBlock& b : r.far_blocks())
                    CHECK(b.memo_bytes <= budget);
            }
        }
    }
    CHECK(limited > 0);    // the small budget exercised the refused-strip path
    CHECK(saved > 0);      // the memo saved evaluations
    CHECK(skips > 0);      // the lossy object exercised the memoised skip decisions
    CHECK(fallbacks > 0);  // full() after an ACA attempt read the memo
}

TEST_CASE("pml_ring: the far ring with the pair memo is thread-count independent", "[pml_ring]") {
    // WP-PML5d: assembly (forced ACA) with 1 / 2 / 3 / all threads, with the default memo and with
    // a small budget (strips refused; the memo is per block and filled by one thread in the block's
    // own order): bitwise the same applied far ring and statistics.
    Case c(tiny_rough_surface(), material::Material{Complex(2.25, 0.0), 1.0}, 1.0,
           formulation::Kind::PMCHWT);
    const Index n2 = 2 * c.space.size();
    const mlfmm::Octree tree = ring_tree(c.space, 3);
    for (const std::size_t budget : {mlfmm::PmlRingParams{}.pair_memo_bytes, std::size_t{30000}}) {
        mlfmm::PmlRingParams p = far_only(forced_aca());
        p.pair_memo_bytes = budget;
#ifdef SPECKLEBEM_HAVE_OPENMP
        const int saved = omp_get_max_threads();
        const int most = std::max(saved, omp_get_num_procs());
        std::vector<std::vector<VectorXc>> cols;
        std::vector<mlfmm::PmlRingFarStats> stats;
        std::vector<std::vector<mlfmm::PmlRingFarBlock>> blocks;
        for (const int t : {1, 2, 3, most}) {
            omp_set_num_threads(t);
            const mlfmm::PmlRingOperator r(c.problem, tree, p);
            cols.push_back(applied_columns(r, n2));
            stats.push_back(r.far_stats());
            blocks.push_back(r.far_blocks());
        }
        omp_set_num_threads(saved);
        REQUIRE(stats[0].pair_memo_hits > 0);
        for (std::size_t i = 1; i < cols.size(); ++i) {
            INFO("budget " << budget << ", variant " << i);
            CHECK(same_columns(cols[i], cols[0]));
            CHECK(same_blocks(blocks[i], blocks[0]));
            CHECK(stats[i].pair_evaluations == stats[0].pair_evaluations);
            CHECK(stats[i].pair_memo_hits == stats[0].pair_memo_hits);
            CHECK(stats[i].memo_bytes_max == stats[0].memo_bytes_max);
            CHECK(stats[i].memo_limited_blocks == stats[0].memo_limited_blocks);
        }
#else
        const mlfmm::PmlRingOperator r(c.problem, tree, p);
        CHECK(r.far_stats().pair_memo_hits > 0);
#endif
    }
}
