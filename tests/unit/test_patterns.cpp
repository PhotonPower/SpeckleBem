#include "specklebem/compression/mlfmm/patterns.hpp"
#include "specklebem/geometry/rough_surface.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/kernels/quadrature.hpp"
#include "specklebem/material/material.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

using namespace specklebem;
using mlfmm::Box;
using mlfmm::Octree;
using mlfmm::RadiationPatterns;
using mlfmm::SphereSampling;

namespace {

constexpr Real kLambda = 500e-9;
const Complex kJ{0.0, 1.0};

Real omega0() {
    return 2.0 * constants::pi * constants::c0 / kLambda;
}

kernels::RegionParams region_of(const material::Material& m) {
    const Real w = omega0();
    return {m.wavenumber(w), m.wave_impedance(w), w, constants::eps0 * m.eps_r,
            constants::mu0 * m.mu_r};
}

material::Material dielectric() {
    return {Complex(2.25, 0.0), Complex(1.0, 0.0)};
}

Real max_edge(const geometry::TriangleMesh& mesh) {
    Real h = 0.0;
    for (Index e = 0; e < mesh.num_edges(); ++e) h = std::max(h, mesh.edge_length(e));
    return h;
}

Real max_abs(std::span<const Complex> v) {
    Real m = 0.0;
    for (const Complex& z : v) m = std::max(m, std::abs(z));
    return m;
}

/// Leaves on level 2 (box edge = root / 4), the coarsest level with interaction lists.
mlfmm::OctreeParams level2_params() {
    return mlfmm::OctreeParams{1, 3, 0.0};
}

/// Dense oracle: Galerkin L, K^PV blocks of the bases of leaf boxes a (rows) and b (columns)
/// summed from element_blocks over the support triangles, as DenseStrategy assembles them.
struct DenseBlock {
    MatrixXc L, K;
};

DenseBlock dense_block(const basis::RwgSpace& space, const Octree& tree, Index a, Index b,
                       const kernels::RegionParams& region, Real accuracy) {
    kernels::OperatorOptions opt;
    opt.target_accuracy = accuracy;  // near/far degree selection (estimate 2-20x above actual)
    const Box& A = tree.boxes()[static_cast<std::size_t>(a)];
    const Box& B = tree.boxes()[static_cast<std::size_t>(b)];
    const std::vector<Index>& perm = tree.permutation();
    std::vector<Index> row(static_cast<std::size_t>(space.size()), -1);
    std::vector<Index> col(static_cast<std::size_t>(space.size()), -1);
    std::vector<Index> ta, tb;
    for (Index i = 0; i < A.num_elements; ++i) {
        const Index n = perm[static_cast<std::size_t>(A.first_element + i)];
        row[static_cast<std::size_t>(n)] = i;
        ta.push_back(space.plus_triangle(n));
        ta.push_back(space.minus_triangle(n));
    }
    for (Index j = 0; j < B.num_elements; ++j) {
        const Index n = perm[static_cast<std::size_t>(B.first_element + j)];
        col[static_cast<std::size_t>(n)] = j;
        tb.push_back(space.plus_triangle(n));
        tb.push_back(space.minus_triangle(n));
    }
    for (std::vector<Index>* v : {&ta, &tb}) {
        std::sort(v->begin(), v->end());
        v->erase(std::unique(v->begin(), v->end()), v->end());
    }
    DenseBlock out{MatrixXc::Zero(A.num_elements, B.num_elements),
                   MatrixXc::Zero(A.num_elements, B.num_elements)};
    Eigen::Matrix<Complex, 3, 3> L, K;
    for (const Index t : ta) {
        const basis::RwgSpace::Support st = space.support(t);
        for (const Index s : tb) {
            const basis::RwgSpace::Support ss = space.support(s);
            kernels::element_blocks(space, t, s, region, opt, L, K);
            for (int x = 0; x < st.count; ++x) {
                const Index i = row[static_cast<std::size_t>(st.n[x])];
                if (i < 0)
                    continue;
                for (int y = 0; y < ss.count; ++y) {
                    const Index j = col[static_cast<std::size_t>(ss.n[y])];
                    if (j < 0)
                        continue;
                    out.L(i, j) += L(x, y);
                    out.K(i, j) += K(x, y);
                }
            }
        }
    }
    return out;
}

/// Interaction-list pairs (a, b) of the leaf level: for the first leaves with a non-empty
/// list, the nearest and the farthest partner.
std::vector<std::pair<Index, Index>> test_pairs(const Octree& tree, std::size_t count) {
    std::vector<std::pair<Index, Index>> pairs;
    for (const Index a : tree.boxes_at_level(tree.leaf_level())) {
        const Box& A = tree.boxes()[static_cast<std::size_t>(a)];
        if (A.interaction_list.empty())
            continue;
        const auto dist = [&](Index b) {
            return (tree.boxes()[static_cast<std::size_t>(b)].center - A.center).norm();
        };
        const auto [lo, hi] =
            std::minmax_element(A.interaction_list.begin(), A.interaction_list.end(),
                                [&](Index x, Index y) { return dist(x) < dist(y); });
        pairs.emplace_back(a, *lo);
        if (pairs.size() >= count)
            break;
        pairs.emplace_back(a, *hi);
        if (pairs.size() >= count)
            break;
    }
    return pairs;
}

Real rel_fro(const MatrixXc& x, const MatrixXc& ref) {
    return (x - ref).norm() / ref.norm();
}

struct SingleLevelResult {
    Real err_l = 0, err_k = 0, err_k_flipped = 0;
    int order = 0, sampling_order = 0;
    bool achievable = false;
};

/// Single-level FMM vs dense for the given pairs: max relative Frobenius errors. If the order
/// search reports the target as not achievable, the errors are evaluated at its best order
/// only with `always` (for the sweep's table).
SingleLevelResult single_level(const basis::RwgSpace& space, const Octree& tree,
                               const kernels::RegionParams& region, Real digits,
                               const std::vector<std::pair<Index, Index>>& pairs,
                               bool always = false) {
    SingleLevelResult r;
    const mlfmm::LeafSampling ls = mlfmm::leaf_sampling(space, tree, region.k, digits);
    r.order = ls.truncation_order;
    r.sampling_order = ls.sampling_order;
    r.achievable = ls.search.achievable;
    if (!r.achievable && !always)
        return r;
    mlfmm::PatternOptions popt;
    popt.target_accuracy = 1e-2 * std::pow(10.0, -digits);  // quadrature well below 10^-d0
    const RadiationPatterns pat(space, tree, region.k, SphereSampling(ls.sampling_order), popt);
    for (const auto& [a, b] : pairs) {
        const DenseBlock d = dense_block(space, tree, a, b, region, 1e-2 * std::pow(10.0, -digits));
        const mlfmm::FarBlock f = mlfmm::far_block(pat, a, b, ls.truncation_order, region);
        r.err_l = std::max(r.err_l, rel_fro(f.L, d.L));
        r.err_k = std::max(r.err_k, rel_fro(f.K, d.K));
        r.err_k_flipped = std::max(r.err_k_flipped, rel_fro(-f.K, d.K));
    }
    return r;
}

void check_single_level(const std::string& name, const basis::RwgSpace& space, const Octree& tree,
                        const kernels::RegionParams& region, Real digits, std::size_t npairs) {
    const auto pairs = test_pairs(tree, npairs);
    REQUIRE(pairs.size() == npairs);
    const SingleLevelResult r = single_level(space, tree, region, digits, pairs);
    INFO(name << ": a = " << tree.box_size(tree.leaf_level()) / kLambda << " lambda0, d0 = "
              << digits << ", L = " << r.order << ", L_leaf = " << r.sampling_order
              << ", achievable = " << r.achievable << ", err L = " << r.err_l
              << ", err K = " << r.err_k << ", flipped K = " << r.err_k_flipped);
    REQUIRE(r.achievable);  // all test configurations are achievable (sweep table)
    CHECK(r.err_l <= std::pow(10.0, -digits));
    CHECK(r.err_k <= std::pow(10.0, -digits));
    CHECK(r.err_k_flipped > 1.0);
}

}  // namespace

TEST_CASE("patterns: quadrature degree calibration", "[.][patterns_calib]") {
    const geometry::TriangleMesh mesh = geometry::make_icosphere(1.0, 1);
    const basis::RwgSpace space(mesh);
    const Octree tree(space, 1.0, mlfmm::OctreeParams{1000, 1, 0.0});
    const SphereSampling s(8);
    const Real h = max_edge(mesh);
    for (const Complex n : {Complex(1.0, 0.0), Complex(4.3, -0.07), Complex(0.05, -3.13)}) {
        for (const Real kh : {0.25, 0.5, 1.0, 2.0, 3.0, 4.0}) {
            const Complex k = n * (kh / std::abs(n)) / h;
            mlfmm::PatternOptions ref_opt;
            ref_opt.quad_degree = 20;
            const RadiationPatterns ref(space, tree, k, s, ref_opt);
            const Real scale = max_abs(ref.data());
            std::string line;
            for (const int d : {1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 17, 19}) {
                mlfmm::PatternOptions o;
                o.quad_degree = d;
                const RadiationPatterns p(space, tree, k, s, o);
                Real e = 0.0;
                for (std::size_t i = 0; i < p.data().size(); ++i) {
                    e = std::max(e, std::abs(p.data()[i] - ref.data()[i]));
                }
                line += " " + std::to_string(std::lround(10.0 * std::log10(e / scale)));
            }
            WARN("n = " << n << " kh = " << kh << line);
        }
    }
}

TEST_CASE("patterns: quadrature converges with the degree", "[patterns]") {
    // Icosphere n = 1 (one box): |k| h up to 3, real and Si-like complex k. Error relative to
    // the largest pattern entry, against degree 20.
    const geometry::TriangleMesh mesh = geometry::make_icosphere(1.0, 1);
    const basis::RwgSpace space(mesh);
    const Octree tree(space, 1.0, mlfmm::OctreeParams{1000, 1, 0.0});
    const SphereSampling s(3);
    const Real h = max_edge(mesh);
    for (const Complex n : {Complex(1.0, 0.0), Complex(4.3, -0.07)}) {
#ifdef NDEBUG
        for (const Real kh : {0.5, 1.5, 3.0}) {
#else
        for (const Real kh : {0.5, 3.0}) {  // budget of the sanitizer build
#endif
            const Complex k = n * (kh / std::abs(n)) / h;
            mlfmm::PatternOptions ref_opt;
            ref_opt.quad_degree = 20;
            const RadiationPatterns ref(space, tree, k, s, ref_opt);
            const Real scale = max_abs(ref.data());
            const auto error = [&](const mlfmm::PatternOptions& o) {
                const RadiationPatterns p(space, tree, k, s, o);
                Real e = 0.0;
                for (std::size_t i = 0; i < p.data().size(); ++i) {
                    e = std::max(e, std::abs(p.data()[i] - ref.data()[i]));
                }
                return e / scale;
            };
            Real prev = 1.0;
            for (const int d : {2, 5, 9, 13}) {
                mlfmm::PatternOptions o;
                o.quad_degree = d;
                const Real e = error(o);
                INFO("n = " << n << ", kh = " << kh << ", degree " << d << ": " << e);
                CHECK(e < prev);
                prev = std::max(e, 1e-14);
            }
            for (const Real target : {1e-4, 1e-6, 1e-8}) {
                mlfmm::PatternOptions o;
                o.target_accuracy = target;
                const Real e = error(o);
                INFO("n = " << n << ", kh = " << kh << ", target " << target << ": " << e);
                CHECK(e <= target);
            }
        }
    }
}

TEST_CASE("patterns: radiation and antipodal receiving patterns match direct evaluation",
          "[patterns]") {
    const geometry::TriangleMesh mesh = geometry::make_icosphere(kLambda, 1);
    const basis::RwgSpace space(mesh);
    const Octree tree(space, kLambda, level2_params());
    const SphereSampling s(7);
    for (const Complex k :
         {region_of(material::vacuum()).k, region_of(material::silicon_500nm()).k}) {
        mlfmm::PatternOptions opt;
        opt.quad_degree = 8;
        const RadiationPatterns pat(space, tree, k, s, opt);
        const kernels::TriangleRule& rule = kernels::triangle_rule(8);
        const std::vector<Index>& perm = tree.permutation();
        Real err_v = 0.0, err_r = 0.0, scale = 0.0;
        for (Index p = 0; p < space.size(); p += 7) {
            const Index n = perm[static_cast<std::size_t>(p)];
            const Vec3 c = tree.boxes()[static_cast<std::size_t>(pat.leaf_box(p))].center;
            for (Index q = 0; q < s.size(); q += 5) {
                const Vec3 kq = s.directions().row(q).transpose();
                Vec3c v = Vec3c::Zero(), r = Vec3c::Zero();
                for (const Index t : {space.plus_triangle(n), space.minus_triangle(n)}) {
                    const Triangles& tri = mesh.triangles();
                    for (std::size_t i = 0; i < rule.weights.size(); ++i) {
                        const Vec3& l = rule.barycentric[i];
                        const Vec3 x = l(0) * mesh.vertices().row(tri(t, 0)).transpose() +
                                       l(1) * mesh.vertices().row(tri(t, 1)).transpose() +
                                       l(2) * mesh.vertices().row(tri(t, 2)).transpose();
                        const Vec3 f = rule.weights[i] * mesh.area(t) * space.value(n, t, x);
                        const Real ph = kq.dot(x - c);
                        v += f.cast<Complex>() * std::exp(kJ * k * ph);
                        r += f.cast<Complex>() * std::exp(-kJ * k * ph);
                    }
                }
                for (int cmp = 0; cmp < 2; ++cmp) {
                    const Vec3 e = cmp == 0 ? Vec3(s.theta_hat().row(q).transpose())
                                            : Vec3(s.phi_hat().row(q).transpose());
                    const Complex vd = e.cast<Complex>().cwiseProduct(v).sum();
                    const Complex rd = e.cast<Complex>().cwiseProduct(r).sum();
                    err_v = std::max(err_v, std::abs(pat.radiation(p, q, cmp) - vd));
                    err_r = std::max(err_r, std::abs(pat.receiving(p, q, cmp) - rd));
                    scale = std::max(scale, std::abs(vd));
                }
                const Vec3 ka = s.directions().row(pat.antipode(q)).transpose();
                CHECK((ka + kq).norm() <= 1e-14);
            }
        }
        INFO("k = " << k << ": radiation " << err_v / scale << ", receiving " << err_r / scale);
        CHECK(err_v <= 1e-13 * scale);
        CHECK(err_r <= 1e-13 * scale);
    }
}

// Single-level FMM vs dense (8 interaction-list pairs: nearest and farthest partner of the
// first four leaves with a non-empty list). The block error at the searched order depends on
// a / lambda (lambda in the medium) and rmax / a; measured with 8 pairs (slow sweep), it meets
// 10^-3 from a ~ 0.75 lambda and 10^-5 from a ~ 1.5 lambda, while at a = lambda / 2 (vacuum,
// rmax / a = 0.29) the nearest pairs reach 1.3e-3 (d0 = 3) and 3.6e-4 (d0 = 5) although the
// statistical order search reports both as achievable. The regular cases test d0 = 3 at
// a >= 0.75 lambda (release, icosphere subdivision 3, rmax / a = 0.29); the d0 = 5 cases need
// L ~ 35-50 and take several seconds, so they are in the slow sweep. Unoptimised + sanitizer
// builds run subdivision 2 (rmax / a = 0.56) with 3 pairs as a code-path check.
namespace {

#ifdef NDEBUG
constexpr int kSub = 3;
constexpr std::size_t kPairs = 8;
#else
constexpr int kSub = 2;
constexpr std::size_t kPairs = 3;
#endif

void check_icosphere(const std::string& name, Real radius, const material::Material& m) {
    const geometry::TriangleMesh mesh = geometry::make_icosphere(radius, kSub);
    const basis::RwgSpace space(mesh);
    const Octree tree(space, kLambda, level2_params());
    check_single_level(name, space, tree, region_of(m), 3.0, kPairs);
}

geometry::TriangleMesh small_rough_box(Real mesh_size) {
    geometry::RoughSurfaceParams p;
    p.edge_length_L = 1e-6;
    p.rms_roughness = 30e-9;
    p.correlation_length = 250e-9;
    p.mesh_size = mesh_size;
    p.box_depth = 0.3e-6;
    p.box_mesh_size = mesh_size;
    p.seed = 7;
    return geometry::make_rough_surface_mesh(p);
}

}  // namespace

TEST_CASE("patterns: single-level FMM vs dense, icosphere, vacuum", "[patterns]") {
#ifdef NDEBUG
    check_icosphere("icosphere vacuum", 2.0 * kLambda, material::vacuum());  // a = lambda
#else
    check_icosphere("icosphere vacuum", kLambda, material::vacuum());  // a = lambda / 2
#endif
}

TEST_CASE("patterns: single-level FMM vs dense, icosphere, dielectric n = 1.5", "[patterns]") {
    check_icosphere("icosphere n = 1.5", kLambda, dielectric());  // a = 0.75 lambda
}

TEST_CASE("patterns: single-level FMM vs dense, icosphere, Si", "[patterns]") {
    // a = lambda_0 / 4 = 1.07 lambda_Si, |k| h ~ 2 (release) / 3.5 (debug).
    check_icosphere("icosphere Si", 0.5 * kLambda, material::silicon_500nm());
}

TEST_CASE("patterns: single-level FMM vs dense, rough-surface box", "[patterns]") {
    // 1 um x 1 um x 0.3 um box, leaves 0.25 um: media n = 1.5 and 2 (a = 0.75 and 1 lambda),
    // 4 pairs (dense oracle cost).
    // In vacuum (a = lambda / 2) the nearest pairs give 9.7e-4 at d0 = 3 and at d0 = 5 3.3e-4,
    // with a minimum over L of 1.3e-4 (independent of the mesh size 62.5 / 31.25 nm): the flat
    // surface fills the facing box faces (near-corner configurations).
#ifdef NDEBUG
    const geometry::TriangleMesh mesh = small_rough_box(62.5e-9);
#else
    const geometry::TriangleMesh mesh = small_rough_box(125e-9);
#endif
    const basis::RwgSpace space(mesh);
    const Octree tree(space, kLambda, level2_params());
#ifdef NDEBUG
    for (const Real n : {1.5, 2.0}) {
#else
    for (const Real n : {2.0}) {  // budget of the sanitizer build
#endif
        const material::Material m{Complex(n * n, 0.0), Complex(1.0, 0.0)};
        check_single_level("rough box", space, tree, region_of(m), 3.0,
                           std::min<std::size_t>(kPairs, 4));
    }
}

TEST_CASE("patterns: invalid input", "[patterns]") {
    const geometry::TriangleMesh mesh = geometry::make_icosphere(kLambda, 1);
    const basis::RwgSpace space(mesh);
    const Octree tree(space, kLambda, level2_params());
    const kernels::RegionParams vac = region_of(material::vacuum());
    const SphereSampling s(10);
    CHECK_THROWS_AS(mlfmm::pattern_quadrature_degree(-1.0, 1e-6), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::pattern_quadrature_degree(1.0, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::pattern_quadrature_degree(1.0, 1.0), std::invalid_argument);
    CHECK_THROWS_AS(RadiationPatterns(space, tree, Complex(1.0, 0.1), s), std::invalid_argument);
    CHECK_THROWS_AS(RadiationPatterns(space, tree, Complex(-1.0, 0.0), s), std::invalid_argument);
    mlfmm::PatternOptions bad;
    bad.quad_degree = 21;
    CHECK_THROWS_AS(RadiationPatterns(space, tree, vac.k, s, bad), std::invalid_argument);
    bad.quad_degree = 0;
    bad.target_accuracy = 0.0;
    CHECK_THROWS_AS(RadiationPatterns(space, tree, vac.k, s, bad), std::invalid_argument);
    const geometry::TriangleMesh other_mesh = geometry::make_icosphere(kLambda, 2);
    const basis::RwgSpace other(other_mesh);
    CHECK_THROWS_AS(RadiationPatterns(other, tree, vac.k, s), std::invalid_argument);

    const RadiationPatterns pat(space, tree, vac.k, s);
    const std::vector<Index>& leaves = tree.boxes_at_level(tree.leaf_level());
    const Box& A = tree.boxes()[static_cast<std::size_t>(leaves.front())];
    REQUIRE(!A.interaction_list.empty());
    const Index b = A.interaction_list.front();
    CHECK_NOTHROW(mlfmm::far_block(pat, leaves.front(), b, 8, vac));
    // Self and near-list pairs, non-leaf boxes, indices out of range.
    CHECK_THROWS_AS(mlfmm::far_block(pat, leaves.front(), leaves.front(), 8, vac),
                    std::invalid_argument);
    if (!A.near_list.empty()) {
        CHECK_THROWS_AS(mlfmm::far_block(pat, leaves.front(), A.near_list.front(), 8, vac),
                        std::invalid_argument);
    }
    CHECK_THROWS_AS(mlfmm::far_block(pat, 0, b, 8, vac), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::far_block(pat, -1, b, 8, vac), std::out_of_range);
    CHECK_THROWS_AS(
        mlfmm::far_block(pat, leaves.front(), static_cast<Index>(tree.boxes().size()), 8, vac),
        std::out_of_range);
    // Truncation order above the sampling order, mismatched region.
    CHECK_THROWS_AS(mlfmm::far_block(pat, leaves.front(), b, 11, vac), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::far_block(pat, leaves.front(), b, -1, vac), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::far_block(pat, leaves.front(), b, 8, region_of(dielectric())),
                    std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::translator(vac.k, Vec3(1e-6, 0, 0), s, 11), std::invalid_argument);
    CHECK_THROWS_AS(mlfmm::leaf_sampling(space, tree, Complex(1.0, 1.0), 3.0),
                    std::invalid_argument);
}

TEST_CASE("patterns: single-level accuracy sweep", "[.][patterns_sweep]") {
#ifndef NDEBUG
    SKIP("release-only sweep");
#endif
    // Table of the single-level FMM block errors (8 pairs) at the searched orders. Checked
    // where the error is expected below 10^-d0: a >= 0.75 lambda (d0 = 3) and a >= 1.5 lambda
    // (d0 = 5), lambda in the medium; smaller boxes are reported only (see the regular cases).
    const auto report = [](const std::string& name, const basis::RwgSpace& space,
                           const Octree& tree, const material::Material& m) {
        const auto pairs = test_pairs(tree, 8);
        const Real a = std::abs(m.refractive_index()) * tree.box_size(2) / kLambda;
        for (const Real d0 : {3.0, 5.0}) {
            const SingleLevelResult r = single_level(space, tree, region_of(m), d0, pairs, true);
            WARN(name << ": a = " << a
                      << " lambda, rmax/a = " << mlfmm::max_support_radius(space) / tree.box_size(2)
                      << ", d0 = " << d0 << ", achievable = " << r.achievable << ", L = " << r.order
                      << ", L_leaf = " << r.sampling_order << ", err L = " << r.err_l
                      << ", err K = " << r.err_k);
            if (a >= (d0 <= 3.0 ? 0.74 : 1.49)) {
                CHECK(r.achievable);
                CHECK(r.err_l <= std::pow(10.0, -d0));
                CHECK(r.err_k <= std::pow(10.0, -d0));
            }
        }
    };
    // Icospheres: leaf a = R / 2 (lambda_0 / 4 ... lambda_0); Si only up to R = lambda_0
    // (pattern memory at L ~ 80).
    for (const Real radius : {0.5 * kLambda, kLambda, 2.0 * kLambda}) {
        for (const int sub : {3, 4}) {
            const geometry::TriangleMesh mesh = geometry::make_icosphere(radius, sub);
            const basis::RwgSpace space(mesh);
            const Octree tree(space, kLambda, level2_params());
            const std::string name = "icosphere R = " + std::to_string(radius / kLambda) +
                                     " lambda0, subdivision " + std::to_string(sub);
            report(name + ", vacuum", space, tree, material::vacuum());
            report(name + ", n = 1.5", space, tree, dielectric());
            if (radius <= kLambda && sub == 3) {
                report(name + ", Si", space, tree, material::silicon_500nm());
            }
        }
    }
    // Rough box (62.5 nm mesh, leaves 0.25 um): media n = 1, 1.5, 2, 3.
    const geometry::TriangleMesh mesh = small_rough_box(62.5e-9);
    const basis::RwgSpace space(mesh);
    const Octree tree(space, kLambda, level2_params());
    for (const Real n : {1.0, 1.5, 2.0, 3.0}) {
        report("rough box, n = " + std::to_string(n), space, tree,
               material::Material{Complex(n * n, 0.0), Complex(1.0, 0.0)});
    }
}
