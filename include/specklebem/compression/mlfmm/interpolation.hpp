#pragma once
/// @file interpolation.hpp
/// Local Lagrange interpolation of sampled functions on the unit sphere between two
/// SphereSamplings (ADR 0008 §3): child (coarse, order L_c) -> parent (fine, order L_p) in the
/// aggregation pass, and its exact transpose (anterpolation) in the disaggregation pass.
///
/// Separable: first p-point Lagrange in phi (periodic, uniform nodes) from every source theta
/// row to the target phi grid, then p-point Lagrange in theta (Gauss-Legendre nodes, theta
/// continued across the poles by f(-theta, phi) = s f(theta, phi + pi) and f(2pi - theta, phi) =
/// s f(theta, phi + pi)). The parity s is +1 for scalars and Cartesian components and -1 for
/// the spherical components along theta_hat / phi_hat (both change sign across a pole). The
/// stencils are stored as contiguous index/weight tables (ADR 0005); apply cost
/// O(p (n_theta_src + n_theta_tgt) n_phi_tgt), no allocation.
///
/// Accuracy (measured, tests/unit/test_plane_wave.cpp): for exp(-jk khat.d), |d| <= half the
/// child diagonal D_c, L_c = truncation_order(k, D_c, d0) -> L_p = truncation_order(k, 2 D_c,
/// d0), real k, child edge lambda/2 ... 2 lambda, the max error is ~1e-3 ... 1e-2 for p = 6,
/// 1e-4 ... 7e-4 for p = 10, <= 3.3e-5 for p = 14 and <= 4e-7 for p = 22 (d0 = 5, 1 ... 2
/// lambda). The error grows with the box size (the ADR sampling approaches 4 points per
/// wavelength of the pattern), so 0.1 x 10^-d0 needs p ~ 12-14 for d0 = 3 and ~ 20-22 for
/// d0 = 5; the default p = 6 meets only ~10^-2.
#include "specklebem/compression/mlfmm/plane_wave.hpp"
#include "specklebem/core/types.hpp"

#include <span>
#include <vector>

namespace specklebem::mlfmm {

/// +1 (even) for scalar fields and Cartesian components, -1 (odd) for theta/phi components.
enum class PoleParity { even, odd };

class SphereInterpolator {
public:
    /// @throws std::invalid_argument if order < 2 or order exceeds the number of source theta
    ///         or phi nodes.
    SphereInterpolator(const SphereSampling& source, const SphereSampling& target, int order = 6);

    [[nodiscard]] int order() const { return order_; }
    [[nodiscard]] Index source_size() const { return Index{nt_src_} * np_src_; }
    [[nodiscard]] Index target_size() const { return Index{nt_tgt_} * np_tgt_; }
    /// Complex scratch entries needed by interpolate / anterpolate.
    [[nodiscard]] Index workspace_size() const { return Index{nt_src_} * np_tgt_; }

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
