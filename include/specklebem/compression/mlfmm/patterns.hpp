#pragma once
/// @file patterns.hpp
/// RWG radiation / receiving patterns of the MLFMM leaf level and the single-level FMM far
/// blocks (docs/04_theory_mlfmm.md "Radiation and receiving patterns", ADR 0008 and its
/// amendments).
///
/// Patterns (one region, wavenumber k, Im k <= 0; c = centre of the leaf box of f_n):
///   radiation  V_n(khat) = int (I - khat khat) f_n(r') e^{+jk khat.(r' - c)} dS'
///   K pattern  W_n(khat) = khat x V_n(khat)
///              (not stored: W_theta = -V_phi, W_phi = V_theta)
///   receiving  R_m(khat) = V_m(-khat)   (no conjugation, also for complex k)
/// stored as the (theta_hat, phi_hat) components of V_n at every direction of a SphereSampling.
///
/// Far-field form of the Galerkin entries (derivation in src/compression/mlfmm/patterns.cpp).
/// With G = e^{-jkR}/(4 pi R) = (1/4pi)(-jk/4pi) \oint e^{-jk khat.(r - C_A)} T_L(k, C_A - C_B,
/// khat) e^{+jk khat.(r' - C_B)} d^2khat (plane_wave.hpp) and S_q = w_q T_L(k, C_A - C_B, khat_q):
///   <f_m, L f_n> = j w mu (-jk / 16 pi^2) sum_q S_q R_m(khat_q) . V_n(khat_q)
///                = (w mu k / 16 pi^2) sum_q S_q R_m . V_n,
///   <f_m, K f_n> = (-jk / 16 pi^2) (+jk) sum_q S_q R_m(khat_q) . W_n(khat_q)
///                = (k^2 / 16 pi^2) sum_q S_q R_m . W_n,
/// for L the scalar-potential term (1/(j w eps)) <div f_m, G div' f_n> becomes -j w mu (khat .
/// F_m)(khat . F_n), which turns f_m . f_n into the projector (I - khat khat); for K the
/// source-point gradient grad' G (docs/03) becomes +jk khat G.
#include "specklebem/basis/rwg.hpp"
#include "specklebem/compression/mlfmm/octree.hpp"
#include "specklebem/compression/mlfmm/plane_wave.hpp"
#include "specklebem/core/types.hpp"
#include "specklebem/kernels/operators.hpp"

#include <span>
#include <vector>

namespace specklebem::mlfmm {

/// Dunavant degree of the pattern quadrature on a triangle with |k| h = kh (h its longest edge):
/// the smallest positive-interior degree (1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 17, 19) whose
/// estimated relative error of int f e^{+jk khat.(r - c)} dS is <= target_accuracy (empirical
/// envelope, calibration in src/compression/mlfmm/patterns.cpp); 19 if none reaches it.
/// @throws std::invalid_argument for kh < 0, non-finite kh or target_accuracy outside (0, 1).
[[nodiscard]] int pattern_quadrature_degree(Real kh, Real target_accuracy);

struct PatternOptions {
    /// Fixed Dunavant degree (1..20) for every triangle; 0: pattern_quadrature_degree per
    /// triangle from |k| h_T and target_accuracy.
    int quad_degree = 0;
    /// Target relative quadrature error of the automatic degree choice, in (0, 1). The default
    /// is conservative; for an accuracy target of d0 digits 0.01 x 10^-d0 suffices (the tests
    /// use it: the single-level errors do not change).
    Real target_accuracy = 1e-8;
};

/// Largest support radius of the RWG functions: max over n of the distance from the midpoint
/// of edge n to the four vertices of its two support triangles [m]. The truncation of every
/// level uses the box diagonal enlarged by twice this value (ADR 0008 §1).
/// @throws std::invalid_argument for an empty space.
[[nodiscard]] Real max_support_radius(const basis::RwgSpace& space);

/// Truncation and sampling order of the leaf level of `tree` for one region.
struct LeafSampling {
    Real enlarged_diagonal = 0;  ///< sqrt(3) a_leaf + 2 max_support_radius [m]
    TruncationSearch search;     ///< search_truncation_order result (statistical check)
    /// = search.order. The statistical check is a pre-screen only: achievable == false means the
    /// level must not be used; achievable == true is no guarantee — RWG far blocks can exceed
    /// 10^-d0 below ~0.75 lambda (d0 = 3) / ~1.5 lambda (d0 = 5) leaves (ADR 0008, WP19b
    /// amendment: block-based level check).
    int truncation_order = 0;
    int interpolation_order = 0;  ///< interpolation_order(digits)
    int sampling_order = 0;       ///< leaf_sampling_order(truncation_order, interpolation_order)
};

/// ADR 0008 §1/§2 with its amendments: truncation order of the leaf level from
/// search_truncation_order(k, a_leaf, digits) with box_diagonal = enlarged_diagonal, sampling
/// order L_leaf = leaf_sampling_order(L, interpolation_order(digits)). The caller must check
/// search.achievable (otherwise the leaf level must not be used for this region / digits); a
/// true value still needs the block-based check of ADR 0008 (WP19b amendment).
/// @throws std::invalid_argument for invalid k or digits (as the called functions) or an empty
///         space; std::underflow_error from search_truncation_order.
[[nodiscard]] LeafSampling leaf_sampling(const basis::RwgSpace& space, const Octree& tree,
                                         Complex k, Real digits);

/// Radiation patterns of every RWG function of one region on the leaf level of an Octree.
///
/// Storage (ADR 0005): one contiguous array indexed (basis, direction, component), basis in the
/// permuted (Morton) order of the octree: entry ((p * num_directions() + q) * 2 + c) holds the
/// theta_hat (c = 0) or phi_hat (c = 1) component of V_{perm[p]}(khat_q) relative to the
/// centre of the leaf box containing permuted position p.
///
/// Quadrature: per leaf box and direction, the moments int e^{+jk khat.(r - c)} dS and
/// int (r - c) e^{+jk khat.(r - c)} dS of every support triangle are formed with the Dunavant
/// rule of degree pattern_quadrature_degree(|k| h_T, target_accuracy) (or the fixed degree),
/// then combined into f_n = (div f_n / 2)(r - p_free) on each support triangle. For complex k
/// the factor grows by up to e^{|Im k| |r - c|} (|r - c| <= half the enlarged diagonal):
/// harmless at leaf scale, included in the |k| h degree choice. OpenMP over leaf boxes, each
/// writing only its own bases: deterministic for any thread count.
/// Holds a non-owning reference to the octree (must outlive the patterns).
class RadiationPatterns {
public:
    /// @param space the RWG space the octree was built on
    /// @throws std::invalid_argument for k outside Re k > 0, Im k <= 0 (or not finite), invalid
    ///         options, or a space whose size differs from the octree's element count.
    RadiationPatterns(const basis::RwgSpace& space, const Octree& tree, Complex k,
                      const SphereSampling& sampling, const PatternOptions& opt = {});
    RadiationPatterns(const basis::RwgSpace&, const Octree&&, Complex, const SphereSampling&,
                      const PatternOptions& = {}) = delete;

    [[nodiscard]] Complex k() const { return k_; }
    [[nodiscard]] const Octree& octree() const { return tree_; }
    [[nodiscard]] const SphereSampling& sampling() const { return sampling_; }
    [[nodiscard]] Index num_basis() const { return num_basis_; }
    [[nodiscard]] Index num_directions() const { return sampling_.size(); }
    /// Largest Dunavant degree used on a triangle.
    [[nodiscard]] int max_quad_degree() const { return max_degree_; }
    /// All patterns, size num_basis() * num_directions() * 2 (layout above).
    [[nodiscard]] std::span<const Complex> data() const { return data_; }

    /// Component c (0: theta_hat, 1: phi_hat) of V at permuted position p, direction q.
    [[nodiscard]] Complex radiation(Index p, Index q, int c) const {
        return data_[static_cast<std::size_t>((p * num_directions() + q) * 2 + c)];
    }
    /// Component c of R_p(khat_q) = V_p(-khat_q) in the (theta_hat, phi_hat) basis of khat_q.
    /// The antipode q' of q has theta' = pi - theta, phi' = phi + pi; there theta_hat(q') =
    /// theta_hat(q) and phi_hat(q') = -phi_hat(q), so R_theta = V_theta(q'), R_phi = -V_phi(q').
    [[nodiscard]] Complex receiving(Index p, Index q, int c) const {
        const Complex v = radiation(p, antipode_[static_cast<std::size_t>(q)], c);
        return c == 0 ? v : -v;
    }
    /// Index of -khat_q in the sampling.
    [[nodiscard]] Index antipode(Index q) const { return antipode_[static_cast<std::size_t>(q)]; }
    /// Leaf box (index into octree().boxes()) of permuted position p.
    [[nodiscard]] Index leaf_box(Index p) const { return leaf_box_[static_cast<std::size_t>(p)]; }

private:
    Complex k_;
    const Octree& tree_;
    SphereSampling sampling_;
    Index num_basis_ = 0;
    int max_degree_ = 0;
    std::vector<Complex> data_;
    std::vector<Index> antipode_;
    std::vector<Index> leaf_box_;
};

/// L and K far blocks of one region between two leaf boxes: rows = the bases of box_a
/// (observer, permuted positions first_element ... of box_a, i.e. basis indices
/// octree().permutation()[first_element + i]), columns = the bases of box_b (source).
struct FarBlock {
    MatrixXc L;  ///< <f_m, L f_n>
    MatrixXc K;  ///< <f_m, K^PV f_n> (no jump term: far pairs never touch)
};

/// Single-level FMM (two leaf boxes, docs/04 "Validation hooks"): the Galerkin blocks
/// <f_m, L f_n>, <f_m, K f_n> of the region of `patterns` from the far-field form in the file
/// comment, with T_L(k, C_A - C_B) of order `truncation_order` sampled at the patterns'
/// directions. Cost O(|A| |B| num_directions) (two complex matrix products).
/// @throws std::out_of_range for box indices outside the octree;
///         std::invalid_argument if a box is not a leaf, box_b is not in box_a's interaction
///         list, truncation_order is outside 0..sampling().order(), or region.k differs from
///         patterns.k() (relative 1e-12) or omega / mu are not finite; std::overflow_error
///         from translator.
[[nodiscard]] FarBlock far_block(const RadiationPatterns& patterns, Index box_a, Index box_b,
                                 int truncation_order, const kernels::RegionParams& region);

}  // namespace specklebem::mlfmm
