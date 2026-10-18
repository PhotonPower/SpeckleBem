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

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
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
        : om(std::move(m)),
          space(om.mesh),
          form(formulation::make_formulation(kind)),
          pml(geometry::make_pml_stretch(om, sigma_max)) {
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
    CHECK(ring.estimate().far_bytes == ring.far_bytes());
    CHECK(ring.estimate().near_bytes == ring.near_bytes());
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
    // Memory warning (review W3): above the available memory only; 0 = unknown.
    CHECK(mlfmm::warn_pml_ring_memory(est, 1));
    CHECK_FALSE(mlfmm::warn_pml_ring_memory(est, est.far_bytes));
    CHECK_FALSE(mlfmm::warn_pml_ring_memory(est, 0));
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
