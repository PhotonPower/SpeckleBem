#pragma once
/// @file interpolation.hpp
/// Local Lagrange interpolation of sampled functions on the unit sphere between two
/// SphereSamplings (ADR 0008 §3 and its amendments): child (order L_c) -> parent (order L_p) in
/// the aggregation pass, and its exact transpose (anterpolation) in the disaggregation pass.
///
/// Separable: first p-point Lagrange in phi (periodic, uniform nodes) from every source theta
/// row to the target phi grid, then p-point Lagrange in theta (Gauss-Legendre nodes, theta
/// continued across the poles by f(-theta, phi) = s f(theta, phi + pi) and f(2pi - theta, phi) =
/// s f(theta, phi + pi)). The parity s is +1 for scalars and Cartesian components and -1 for
/// the spherical components along theta_hat / phi_hat (both change sign across a pole). The
/// stencils are stored as contiguous index/weight tables (ADR 0005); apply cost
/// O(p (n_theta_src + n_theta_tgt) n_phi_tgt), no allocation.
///
/// Choosing p and the leaf sampling (ADR 0008 amendments): p = interpolation_order(d0) from the
/// measured table; the finest (leaf) level is sampled at leaf_sampling_order(L, p) =
/// max(L, p - 1) instead of its truncation order L, so that small leaf boxes (lambda/4: L = 8 at
/// d0 = 3, 11 at d0 = 5) have at least p theta nodes. Coarser levels keep their own L; their
/// stencils may extend over the pole-reflected nodes (any p <= 2 n_theta_src is accepted).
///
/// Accuracy (measured, tests/unit/test_plane_wave.cpp): for exp(-jk khat.d), |d| = half the
/// child diagonal D_c, L_c = truncation_order(k, D_c, d0) -> L_p = truncation_order(k, 2 D_c,
/// d0), real k, child edge lambda/2 ... 2 lambda, the max error is ~1e-3 ... 1e-2 for p = 6,
/// 1e-4 ... 7e-4 for p = 10, <= 3.3e-5 for p = 14 and <= 4e-7 for p = 22 (d0 = 5, 1 ... 2
/// lambda). The error grows with the box size (the ADR sampling approaches 4 points per
/// wavelength of the pattern), so 0.1 x 10^-d0 needs p ~ 12-14 for d0 = 3 and ~ 20-22 for
/// d0 = 5. Lambda/4 leaves with the oversampled leaf sampling: d0 = 3 (p = 14): L_leaf = 13 =
/// the parent order (identity, 1e-15); d0 = 5 (p = 22): L_leaf = 21 -> 15, <= 7.5e-12. Without
/// oversampling (stencils over the reflected nodes): 9.8e-5 (d0 = 3, n_theta = 9) and 3.4e-7
/// (d0 = 5, n_theta = 12), i.e. at or just below the 0.1 x 10^-d0 target.
///
/// Weight convention for the disaggregation (WP20): the plain transpose I^T is the exact
/// adjoint of I with respect to the *unweighted* bilinear form sum_q a_q b_q. In the downward
/// pass the incoming field of a level must therefore carry the quadrature weights of the level
/// it lives on: multiply the translated parent field by the parent weights w_p *before*
/// anterpolating, and do NOT multiply by the child weights afterwards (the result already is
/// the child-level weighted field, since sum_p w_p (I f)_p g_p = sum_c f_c (I^T (w_p g))_c);
/// the leaf receive step then sums the receiving pattern times this field without weights.
/// Unweighted fields would need W_c^{-1} I^T W_p instead, which is not provided.
#include "specklebem/compression/mlfmm/plane_wave.hpp"
#include "specklebem/core/types.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace specklebem::mlfmm {

/// Lagrange interpolation order p for `digits` = d0 from the measured table of the ADR 0008
/// amendment (interpolation error <= 0.1 x 10^-d0 between consecutive levels with the §2
/// sampling, box edges lambda/4 ... 2 lambda): p = 14 for d0 <= 3, p = 22 for 3 < d0 <= 5.
/// d0 > 5 is outside the measured range (no extrapolation: the needed p grows faster than
/// linearly near the 4-points-per-wavelength sampling limit) and must be measured first.
/// @throws std::invalid_argument for non-finite digits, digits <= 0 or digits > 5.
[[nodiscard]] int interpolation_order(Real digits);

/// Sampling order of the finest (leaf) level: max(truncation_order, interpolation_order - 1),
/// so that the leaf sampling has at least p theta nodes (ADR 0008 review amendment).
/// @throws std::invalid_argument for truncation_order < 0 or interpolation_order < 2.
[[nodiscard]] int leaf_sampling_order(int truncation_order, int interpolation_order);

/// +1 (even) for scalar fields and Cartesian components, -1 (odd) for theta/phi components.
enum class PoleParity { even, odd };

class SphereInterpolator {
public:
    /// `order` p: number of Lagrange nodes per direction, normally interpolation_order(d0); no
    /// default (the order must follow the accuracy target). The theta stencil of p nodes
    /// centred on the target node lies inside the pole-reflected extended node range
    /// [-n_theta_src, 2 n_theta_src) iff ceil(p / 2) <= n_theta_src, i.e. p <= 2 n_theta_src;
    /// the phi stencil holds distinct nodes iff p <= n_phi_src = 2 n_theta_src. Both are the
    /// same bound.
    /// @throws std::invalid_argument if order < 2 or order > 2 * source.num_theta().
    SphereInterpolator(const SphereSampling& source, const SphereSampling& target, int order);

    [[nodiscard]] int order() const { return order_; }
    [[nodiscard]] Index source_size() const { return Index{nt_src_} * np_src_; }
    [[nodiscard]] Index target_size() const { return Index{nt_tgt_} * np_tgt_; }
    /// Complex scratch entries needed by interpolate / anterpolate.
    [[nodiscard]] Index workspace_size() const { return Index{nt_src_} * np_tgt_; }
    /// Bytes of the stencil tables.
    [[nodiscard]] std::size_t memory_bytes() const {
        return sizeof(*this) + phi_index_.size() * sizeof(Index) +
               phi_weight_.size() * sizeof(Real) + theta_index_.size() * sizeof(Index) +
               theta_weight_.size() * sizeof(Real) + theta_flip_.size();
    }

    /// out (target_size) = I in (source_size). Not aliasing; workspace >= workspace_size().
    /// @throws std::invalid_argument on size mismatch.
    void interpolate(std::span<const Complex> in, std::span<Complex> out,
                     std::span<Complex> workspace, PoleParity parity = PoleParity::even) const;
    /// out (source_size) = I^T in (target_size): the exact transpose of interpolate (real
    /// weights, so also the adjoint for the unweighted inner product).
    void anterpolate(std::span<const Complex> in, std::span<Complex> out,
                     std::span<Complex> workspace, PoleParity parity = PoleParity::even) const;

    /// Convenience overloads that allocate the result and the workspace.
    [[nodiscard]] VectorXc interpolate(const VectorXc& in,
                                       PoleParity parity = PoleParity::even) const;
    [[nodiscard]] VectorXc anterpolate(const VectorXc& in,
                                       PoleParity parity = PoleParity::even) const;

private:
    void check_sizes(Index in, Index want_in, Index out, Index want_out, Index ws) const;

    int order_;
    int nt_src_, np_src_, nt_tgt_, np_tgt_;
    // phi stencil per target phi j: source phi indices / weights at [j * order_ + b].
    std::vector<Index> phi_index_;
    std::vector<Real> phi_weight_;
    // theta stencil per target theta i: source theta row, weight and pole flip at [i*order_+a].
    std::vector<Index> theta_index_;
    std::vector<Real> theta_weight_;
    std::vector<unsigned char> theta_flip_;
};

}  // namespace specklebem::mlfmm
