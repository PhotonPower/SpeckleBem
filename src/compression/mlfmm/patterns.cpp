/// @file patterns.cpp
/// RWG radiation patterns and single-level FMM far blocks (docs/04_theory_mlfmm.md, ADR 0008).
///
/// Far-field form of the Galerkin entries (exp(+jwt), G = e^{-jkR} / (4 pi R)). The addition
/// theorem of plane_wave.hpp for observer r near C_A and source r' near C_B gives
///   G(r, r') = a \oint e^{-jk khat.(r - C_A)} T(khat) e^{+jk khat.(r' - C_B)} d^2khat,
///   a = (1 / 4pi)(-jk / 4pi) = -jk / (16 pi^2),   T = T_L(k, C_A - C_B, khat).
/// Write F_m(khat) = int f_m e^{-jk khat.(r - C_A)} dS (unprojected receiving integral) and
/// F'_n(khat) = int f_n e^{+jk khat.(r' - C_B)} dS' (unprojected radiation integral).
///
/// L. Mixed-potential Galerkin entry (kernels/operators.hpp)
///   L_mn = j w mu <f_m, G f_n> + (1 / (j w eps)) <div f_m, G div' f_n>.
/// Surface divergence theorem on the supports (the normal flux of an RWG vanishes on the outer
/// support edges and is continuous across its own edge):
///   int (div f_m) e^{-jk khat.(r - C_A)} dS   = -int f_m . grad e^{...} dS = +jk khat . F_m,
///   int (div' f_n) e^{+jk khat.(r' - C_B)} dS' = -int f_n . grad' e^{...} dS' = -jk khat . F'_n.
/// The product is k^2 (khat . F_m)(khat . F'_n), and k^2 / (j w eps) = w^2 mu eps / (j w eps) =
/// -j w mu (k^2 = w^2 mu eps also for complex eps), so
///   L_mn = j w mu a \oint T [F_m . F'_n - (khat . F_m)(khat . F'_n)] = j w mu a \oint T
///          F_m . (I - khat khat) . F'_n = (w mu k / 16 pi^2) \oint T R_m . V_n,
/// since (I - khat khat) is a symmetric projector: R_m = (I - khat khat) F_m = V_m(-khat), V_n =
/// (I - khat khat) F'_n. Check: the projector removes the longitudinal part, as the far field of
/// a current element must (no radiation along the current).
///
/// K. K_mn = <f_m, int grad' G x f_n dS'> with the source-point gradient (docs/03). In the
/// expansion G depends on r' only through e^{+jk khat.(r' - C_B)}, so grad' -> +jk khat:
///   K_mn = a (jk) \oint T F_m . (khat x F'_n) = (k^2 / 16 pi^2) \oint T R_m . W_n,
/// W_n = khat x V_n (khat x F'_n is transverse, so only the transverse part R_m of F_m counts).
/// Check: grad' G = (1 + jkR) G (r - r') / R^2 -> +jk G Rhat in the far zone, and the
/// stationary direction of e^{-jk khat.(r - r')} is khat = Rhat; the test with the flipped sign
/// (tests/unit/test_patterns.cpp) gives O(1) errors.
///
/// Components: with khat x theta_hat = phi_hat and khat x phi_hat = -theta_hat,
/// W_theta = -V_phi, W_phi = V_theta, and R . V = R_theta V_theta + R_phi V_phi.
#include "specklebem/compression/mlfmm/patterns.hpp"

#include "specklebem/compression/mlfmm/interpolation.hpp"
#include "specklebem/kernels/quadrature.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <string>

namespace specklebem::mlfmm {

namespace {

void check_k(Complex k, const char* where) {
    if (!std::isfinite(k.real()) || !std::isfinite(k.imag()) || k.real() <= 0.0 || k.imag() > 0.0) {
        throw std::invalid_argument(std::string(where) +
                                    ": need finite k with Re k > 0 and Im k <= 0 (exp(+jwt))");
    }
}

/// Positive-interior Dunavant degrees (kernels::triangle_rule_is_positive_interior).
constexpr std::array<int, 13> kDegrees = {1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 17, 19};

/// Empirical envelope of the relative pattern quadrature error of Dunavant degree d at
/// kappa = |k| h: E_d = 3 (kappa / 4)^{d+1} / (d+1)! (Taylor-remainder shape).
/// Calibration (hidden test "[patterns_calib]"): icosphere n = 1, all RWG patterns at the 162
/// directions of SphereSampling(8), max error relative to the largest entry against degree 20,
/// kappa = 0.25 ... 4, n = 1, 4.3 - 0.07j, 0.05 - 3.13j. Measured log10 errors, e.g. kappa = 3
/// (real k): degree 4: -3.3, 6: -5.7, 8: -7.5, 10: -9.9, 12: -12.4, 14: -14.6 (round-off);
/// kappa = 1: 4: -5.3, 6: -8.7, 8: -11.2; kappa = 4: 6: -4.8, 10: -8.7, 14: -13.1. The
/// envelope lies above every measured value of the positive-interior ladder (closest:
/// kappa = 0.25, degree 4: 2.4e-8 vs 2.0e-8; kappa = 4, degree 12: 1.6e-10 vs 1.3e-11) and is
/// conservative by about 1 to 2 digits at kappa >= 2. Complex k (same |k|) is not harder:
/// the Si-like values equal the real ones within 0.1 digit, the Ag-like ones are smaller.
Real error_estimate(int d, Real kappa) {
    Real e = 3.0;
    for (int i = 1; i <= d + 1; ++i) {
        e *= 0.25 * kappa / static_cast<Real>(i);
    }
    return e;
}

/// Per-thread capture of the first exception inside an OpenMP region.
class ExceptionSlot {
public:
    void capture() {
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp critical(specklebem_patterns_exception)
#endif
        {
            if (!ptr_)
                ptr_ = std::current_exception();
        }
    }
    void rethrow() const {
        if (ptr_)
            std::rethrow_exception(ptr_);
    }

private:
    std::exception_ptr ptr_;
};

}  // namespace

int pattern_quadrature_degree(Real kh, Real target_accuracy) {
    if (!std::isfinite(kh) || kh < 0.0) {
        throw std::invalid_argument("pattern_quadrature_degree: need finite kh >= 0");
    }
    if (!std::isfinite(target_accuracy) || target_accuracy <= 0.0 || target_accuracy >= 1.0) {
        throw std::invalid_argument("pattern_quadrature_degree: target_accuracy must be in (0, 1)");
    }
    for (const int d : kDegrees) {
        if (error_estimate(d, kh) <= target_accuracy)
            return d;
    }
    return kDegrees.back();
}

Real max_support_radius(const basis::RwgSpace& space) {
    if (space.size() == 0) {
        throw std::invalid_argument("max_support_radius: empty RWG space");
    }
    const geometry::TriangleMesh& mesh = space.mesh();
    const Vertices& v = mesh.vertices();
    Real r = 0.0;
    for (Index n = 0; n < space.size(); ++n) {
        const Vec3 a = v.row(mesh.edges()(n, 0)).transpose();
        const Vec3 b = v.row(mesh.edges()(n, 1)).transpose();
        const Vec3 mid = 0.5 * (a + b);
        const Vec3 pp = v.row(space.plus_free_vertex(n)).transpose();
        const Vec3 pm = v.row(space.minus_free_vertex(n)).transpose();
        r = std::max({r, (a - mid).norm(), (pp - mid).norm(), (pm - mid).norm()});
    }
    return r;
}

LeafSampling leaf_sampling(const basis::RwgSpace& space, const Octree& tree, Complex k,
                           Real digits) {
    check_k(k, "leaf_sampling");
    LeafSampling s;
    const Real a = tree.box_size(tree.leaf_level());
    s.enlarged_diagonal = std::sqrt(3.0) * a + 2.0 * max_support_radius(space);
    TruncationSearchOptions opt;
    opt.box_diagonal = s.enlarged_diagonal;
    s.search = search_truncation_order(k, a, digits, opt);
    s.truncation_order = s.search.order;
    s.interpolation_order = interpolation_order(digits);
    s.sampling_order = leaf_sampling_order(s.truncation_order, s.interpolation_order);
    return s;
}

RadiationPatterns::RadiationPatterns(const basis::RwgSpace& space, const Octree& tree, Complex k,
                                     const SphereSampling& sampling, const PatternOptions& opt)
    : k_(k), tree_(tree), sampling_(sampling) {
    check_k(k, "RadiationPatterns");
    if (opt.quad_degree < 0 || opt.quad_degree > 20) {
        throw std::invalid_argument("RadiationPatterns: quad_degree must be in 0..20");
    }
    if (opt.quad_degree == 0 && !(opt.target_accuracy > 0.0 && opt.target_accuracy < 1.0)) {
        throw std::invalid_argument("RadiationPatterns: target_accuracy must be in (0, 1)");
    }
    num_basis_ = space.size();
    if (num_basis_ != static_cast<Index>(tree.permutation().size()) || num_basis_ == 0) {
        throw std::invalid_argument(
            "RadiationPatterns: the RWG space does not match the octree (element count)");
    }
    const geometry::TriangleMesh& mesh = space.mesh();
    const Vertices& vert = mesh.vertices();
    const Index nd = sampling_.size();
    // Sizes in std::size_t from clamped values (GCC -O3 -Wnull-dereference false positives).
    const auto nbasis = static_cast<std::size_t>(std::max<Index>(num_basis_, 1));
    const auto ndir = static_cast<std::size_t>(std::max<Index>(nd, 1));
    data_.assign(nbasis * ndir * 2, Complex(0.0, 0.0));
    leaf_box_.assign(nbasis, -1);

    // Antipodal map: theta -> pi - theta (GL nodes are symmetric), phi -> phi + pi.
    const int nt = sampling_.num_theta();
    const int np = sampling_.num_phi();
    antipode_.resize(ndir);
    for (int it = 0; it < nt; ++it) {
        for (int ip = 0; ip < np; ++ip) {
            antipode_[static_cast<std::size_t>(sampling_.index(it, ip))] =
                sampling_.index(nt - 1 - it, (ip + np / 2) % np);
        }
    }

    // Quadrature degree per triangle (rules fetched before the parallel loop).
    const Index nf = mesh.num_triangles();
    std::vector<int> degree(static_cast<std::size_t>(std::max<Index>(nf, 1)), opt.quad_degree);
    const Real absk = std::abs(k);
    for (Index t = 0; t < nf; ++t) {
        if (opt.quad_degree == 0) {
            const Triangles& tri = mesh.triangles();
            Real h = 0.0;
            for (int e = 0; e < 3; ++e) {
                h = std::max(h, (vert.row(tri(t, e)) - vert.row(tri(t, (e + 1) % 3))).norm());
            }
            degree[static_cast<std::size_t>(t)] =
                pattern_quadrature_degree(absk * h, opt.target_accuracy);
        }
        max_degree_ = std::max(max_degree_, degree[static_cast<std::size_t>(t)]);
        (void)kernels::triangle_rule(degree[static_cast<std::size_t>(t)]);
    }

    const std::vector<Index>& leaves = tree.boxes_at_level(tree.leaf_level());
    const std::vector<Index>& perm = tree.permutation();
    const auto& khat = sampling_.directions();
    const auto& th = sampling_.theta_hat();
    const auto& ph = sampling_.phi_hat();
    const auto nleaves = static_cast<Index>(leaves.size());
    ExceptionSlot error;
#ifdef SPECKLEBEM_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (Index ib = 0; ib < nleaves; ++ib) {
        try {
            const Box& box =
                tree.boxes()[static_cast<std::size_t>(leaves[static_cast<std::size_t>(ib)])];
            const Vec3 c = box.center;
            // Support triangles of the box's bases (unique, ascending).
            std::vector<Index> tris;
            for (Index p = box.first_element; p < box.first_element + box.num_elements; ++p) {
                const Index n = perm[static_cast<std::size_t>(p)];
                tris.push_back(space.plus_triangle(n));
                tris.push_back(space.minus_triangle(n));
            }
            std::sort(tris.begin(), tris.end());
            tris.erase(std::unique(tris.begin(), tris.end()), tris.end());
            // Quadrature points relative to c and weights (area included), per triangle.
            std::vector<std::size_t> offset(tris.size() + 1, 0);
            std::vector<std::array<Real, 4>> pts;  // (r - c, weight x area), plain arrays
            for (std::size_t j = 0; j < tris.size(); ++j) {
                const Index t = tris[j];
                const kernels::TriangleRule& rule =
                    kernels::triangle_rule(degree[static_cast<std::size_t>(t)]);
                const Triangles& tri = mesh.triangles();
                const Vec3 v0 = vert.row(tri(t, 0)).transpose();
                const Vec3 v1 = vert.row(tri(t, 1)).transpose();
                const Vec3 v2 = vert.row(tri(t, 2)).transpose();
                const Real area = mesh.area(t);
                for (std::size_t q = 0; q < rule.weights.size(); ++q) {
                    const Vec3& l = rule.barycentric[q];
                    const Vec3 r = l(0) * v0 + l(1) * v1 + l(2) * v2 - c;
                    pts.push_back({r(0), r(1), r(2), rule.weights[q] * area});
                }
                offset[j + 1] = pts.size();
            }
            // Per basis: local triangle slots, div / 2 and c - p_free on both triangles.
            struct Side {
                std::size_t tri;
                Real half_div;
                Vec3 c_minus_p;
            };
            std::vector<std::array<Side, 2>> sides;
            for (Index p = box.first_element; p < box.first_element + box.num_elements; ++p) {
                const Index n = perm[static_cast<std::size_t>(p)];
                std::array<Side, 2> s{};
                for (int side = 0; side < 2; ++side) {
                    const Index t = side == 0 ? space.plus_triangle(n) : space.minus_triangle(n);
                    const Index fv =
                        side == 0 ? space.plus_free_vertex(n) : space.minus_free_vertex(n);
                    const auto it = std::lower_bound(tris.begin(), tris.end(), t);
                    s[static_cast<std::size_t>(side)] = {
                        static_cast<std::size_t>(it - tris.begin()), 0.5 * space.divergence(n, t),
                        c - vert.row(fv).transpose()};
                }
                sides.push_back(s);
                leaf_box_[static_cast<std::size_t>(p)] = leaves[static_cast<std::size_t>(ib)];
            }
            // Moments S0 = int e dS, S1 = int (r - c) e dS per triangle and direction.
            // e^{+jk s} = e^{-Im(k) s} (cos(Re(k) s) + j sin(Re(k) s)), s = khat . (r - c);
            // scalar loops (also fast in the unoptimised build).
            std::vector<std::array<Complex, 4>> mom(tris.size());
            const Real kr = k.real();
            const Real ki = k.imag();
            for (Index q = 0; q < nd; ++q) {
                const Real kx = khat(q, 0);
                const Real ky = khat(q, 1);
                const Real kz = khat(q, 2);
                for (std::size_t j = 0; j < tris.size(); ++j) {
                    std::array<Real, 8> m{};
                    for (std::size_t i = offset[j]; i < offset[j + 1]; ++i) {
                        const std::array<Real, 4>& x = pts[i];
                        const Real sp = kx * x[0] + ky * x[1] + kz * x[2];
                        const Real amp = ki == 0.0 ? x[3] : x[3] * std::exp(-ki * sp);
                        const Real er = amp * std::cos(kr * sp);
                        const Real ei = amp * std::sin(kr * sp);
                        m[0] += er;
                        m[1] += ei;
                        m[2] += er * x[0];
                        m[3] += ei * x[0];
                        m[4] += er * x[1];
                        m[5] += ei * x[1];
                        m[6] += er * x[2];
                        m[7] += ei * x[2];
                    }
                    mom[j] = {Complex(m[0], m[1]), Complex(m[2], m[3]), Complex(m[4], m[5]),
                              Complex(m[6], m[7])};
                }
                for (std::size_t b = 0; b < sides.size(); ++b) {
                    std::array<Complex, 3> v{};
                    for (const Side& s : sides[b]) {
                        const std::array<Complex, 4>& m = mom[s.tri];
                        for (std::size_t a = 0; a < 3; ++a) {
                            v[a] += s.half_div *
                                    (m[a + 1] + s.c_minus_p(static_cast<Eigen::Index>(a)) * m[0]);
                        }
                    }
                    const auto p = static_cast<std::size_t>(box.first_element) + b;
                    const std::size_t at =
                        (p * static_cast<std::size_t>(nd) + static_cast<std::size_t>(q)) * 2;
                    data_[at] = th(q, 0) * v[0] + th(q, 1) * v[1] + th(q, 2) * v[2];
                    data_[at + 1] = ph(q, 0) * v[0] + ph(q, 1) * v[1] + ph(q, 2) * v[2];
                }
            }
        } catch (...) {
            error.capture();
        }
    }
    error.rethrow();
}

FarBlock far_block(const RadiationPatterns& patterns, Index box_a, Index box_b,
                   int truncation_order, const kernels::RegionParams& region) {
    const Octree& tree = patterns.octree();
    const auto nboxes = static_cast<Index>(tree.boxes().size());
    if (box_a < 0 || box_a >= nboxes || box_b < 0 || box_b >= nboxes) {
        throw std::out_of_range("far_block: box index outside the octree");
    }
    const Box& A = tree.boxes()[static_cast<std::size_t>(box_a)];
    const Box& B = tree.boxes()[static_cast<std::size_t>(box_b)];
    if (A.level != tree.leaf_level() || B.level != tree.leaf_level()) {
        throw std::invalid_argument("far_block: both boxes must be leaf boxes");
    }
    if (!std::binary_search(A.interaction_list.begin(), A.interaction_list.end(), box_b)) {
        throw std::invalid_argument("far_block: box_b is not in the interaction list of box_a");
    }
    if (truncation_order < 0 || truncation_order > patterns.sampling().order()) {
        throw std::invalid_argument("far_block: truncation_order must be in 0..sampling order");
    }
    const Complex k = patterns.k();
    if (!(std::abs(region.k - k) <= 1e-12 * std::abs(k))) {
        throw std::invalid_argument("far_block: region.k differs from the patterns' k");
    }
    if (!std::isfinite(region.omega) || !std::isfinite(region.mu.real()) ||
        !std::isfinite(region.mu.imag())) {
        throw std::invalid_argument("far_block: omega and mu must be finite");
    }
    const SphereSampling& s = patterns.sampling();
    const Index nd = s.size();
    const VectorXc t = translator(k, A.center - B.center, s, truncation_order);
    const Index na = A.num_elements;
    const Index nb = B.num_elements;
    MatrixXc rw(na, 2 * nd);
    MatrixXc vm(nb, 2 * nd);
    MatrixXc wm(nb, 2 * nd);
    for (Index q = 0; q < nd; ++q) {
        const Complex sq = s.weights()(q) * t(q);
        for (Index i = 0; i < na; ++i) {
            const Index p = A.first_element + i;
            rw(i, q) = sq * patterns.receiving(p, q, 0);
            rw(i, nd + q) = sq * patterns.receiving(p, q, 1);
        }
        for (Index j = 0; j < nb; ++j) {
            const Index p = B.first_element + j;
            const Complex vt = patterns.radiation(p, q, 0);
            const Complex vp = patterns.radiation(p, q, 1);
            vm(j, q) = vt;
            vm(j, nd + q) = vp;
            wm(j, q) = -vp;      // W_theta = -V_phi
            wm(j, nd + q) = vt;  // W_phi = V_theta
        }
    }
    const Real c16 = 16.0 * constants::pi * constants::pi;
    FarBlock out;
    out.L = (region.omega * region.mu * k / c16) * (rw * vm.transpose());
    out.K = (k * k / c16) * (rw * wm.transpose());
    return out;
}

}  // namespace specklebem::mlfmm
