#pragma once
/// @file plane_wave.hpp
/// Plane-wave (diagonal) machinery of the MLFMM (docs/04_theory_mlfmm.md, ADR 0008 §2, §4, §6):
/// truncation order, sampling of the unit sphere, spherical Hankel functions h_l^{(2)} and
/// Legendre polynomials, the diagonal translator and the a-priori expansion-error check.
///
/// Convention exp(+jwt) (docs/06): outgoing waves exp(-jkR), Im k <= 0. Addition theorem
/// (Gegenbauer, conjugated from the exp(-iwt) form), for X = C_o - C_s and |d| < |X| with
/// d = (o - C_o) + (C_s - s):
///
///   exp(-jk|o - s|) / |o - s|  =  (-jk / 4pi) \oint exp(-jk khat.(o - C_o)) T_L(k, X, khat)
///                                                    exp(-jk khat.(C_s - s)) d^2khat,
///   T_L(k, X, khat) = sum_{l=0}^{L} (-j)^l (2l + 1) h_l^{(2)}(k|X|) P_l(khat . Xhat),
///
/// with the integral replaced by the SphereSampling quadrature (exact for spherical harmonics
/// of degree <= 2L + 1).
#include "specklebem/core/types.hpp"

#include <vector>

namespace specklebem::mlfmm {

/// Truncation order L = ceil(kD + 1.8 d0^{2/3} (kD)^{1/3}) with k -> Re k for the bandwidth
/// (excess-bandwidth formula, ADR 0008 §2). Measured accuracy (tests/unit/test_plane_wave.cpp):
/// for random points inside two boxes of edge a, D = sqrt(3) a, centres >= 2a apart, the
/// expansion error is <= 10^-d0 for real k and ka >= ~3 (a >= lambda/2); at a = lambda/4 it is
/// ~1e-3 (d0 = 3) / ~2e-4 (d0 = 5). Corner-to-corner points of the nearest interaction-list
/// pair (d/X = 0.87) converge much more slowly (see expansion_error()).
/// @throws std::invalid_argument for non-finite input, Re k <= 0, Im k > 0, box_diagonal <= 0,
///         digits <= 0, or an order above 100000.
[[nodiscard]] int truncation_order(Complex k, Real box_diagonal, Real digits);

/// Sample directions of the unit sphere for truncation order L: (L+1) Gauss-Legendre nodes in
/// cos(theta) (theta ascending, i.e. cos(theta) descending) x 2(L+1) uniform phi_j = 2 pi j /
/// (2L+2). Direction index i = i_theta * num_phi() + i_phi (theta-major, contiguous). Weights
/// w_i = w_GL(i_theta) * 2 pi / num_phi sum to 4 pi; the rule integrates spherical harmonics of
/// degree <= 2L + 1 exactly. num_phi() is even, so phi + pi is a grid point (pole reflection).
class SphereSampling {
public:
    /// Column-major (SoA): column c holds the c-th Cartesian component of every direction.
    using DirectionArray = Eigen::Matrix<Real, Eigen::Dynamic, 3>;

    /// @throws std::invalid_argument if `order` < 0 or > 100000.
    explicit SphereSampling(int order);

    [[nodiscard]] int order() const { return order_; }
    [[nodiscard]] int num_theta() const { return order_ + 1; }
    [[nodiscard]] int num_phi() const { return 2 * (order_ + 1); }
    [[nodiscard]] Index size() const { return Index{num_theta()} * num_phi(); }
    [[nodiscard]] Index index(int i_theta, int i_phi) const {
        return Index{i_theta} * num_phi() + i_phi;
    }
    [[nodiscard]] const VectorXr& theta() const { return theta_; }  ///< ascending, in (0, pi)
    [[nodiscard]] const VectorXr& phi() const { return phi_; }      ///< in [0, 2 pi)
    [[nodiscard]] const VectorXr& weights() const { return weights_; }
    [[nodiscard]] const DirectionArray& directions() const { return khat_; }  ///< unit khat
    [[nodiscard]] const DirectionArray& theta_hat() const { return theta_hat_; }
    [[nodiscard]] const DirectionArray& phi_hat() const { return phi_hat_; }

private:
    int order_;
    VectorXr theta_, phi_, weights_;
    DirectionArray khat_, theta_hat_, phi_hat_;
};

/// Spherical Hankel functions of the second kind h_l^{(2)}(z) = j_l(z) - j y_l(z) (outgoing
/// for exp(+jwt): h_0^{(2)}(z) = j e^{-jz}/z), l = 0..max_order, by upward recurrence
/// h_{l+1} = (2l+1)/z h_l - h_{l-1} from the closed forms of h_0, h_1.
/// Domain: Re z > 0, Im z <= 0 (real z included). The recurrence is stable for h^{(2)} (no
/// minimal solution is sought; perturbations stay O(eps |h_l|) in the oscillatory range l < |z|
/// and are dominated by the growing y_l part for l > |z|, where |h_l| ~ (2l-1)!!/|z|^{l+1}).
/// Measured relative error of the complex value h_l <= 1.4e-14 for l <= 120 and |z| in
/// [0.05, 80] (real, and Im z down to -3.1), against the Bessel-polynomial closed form, a long
/// double recurrence and the Wronskian (tests/unit/test_plane_wave.cpp). The real part j_l(z)
/// is NOT accurate where |j_l| << |y_l| (l >> |z|); the translator needs only h_l.
/// @throws std::invalid_argument outside the domain or for max_order < 0;
///         std::overflow_error if |h_l| overflows double (e.g. |z| = 0.1, l > ~150).
[[nodiscard]] std::vector<Complex> spherical_hankel2(int max_order, Complex z);

/// Legendre polynomials P_l(x), l = 0..max_order, by the three-term recurrence
/// (l+1) P_{l+1} = (2l+1) x P_l - l P_{l-1} (stable on [-1, 1]).
/// @throws std::invalid_argument for max_order < 0 or x outside [-1, 1] (or not finite).
[[nodiscard]] std::vector<Real> legendre_p(int max_order, Real x);

/// Diagonal translator T_L(k, r, khat) at every direction of `sampling`, with L =
/// sampling.order() (see file comment; the addition-theorem prefactor -jk/4pi is NOT included).
/// Cost O(L * sampling.size()), no allocation per direction.
/// @throws std::invalid_argument for |r| = 0 or k outside the spherical_hankel2 domain.
[[nodiscard]] VectorXc translator(Complex k, const Vec3& r, const SphereSampling& sampling);

struct ExpansionErrorOptions {
    int order = 0;            ///< 0: truncation_order(k, box_diagonal, digits)
    Real box_diagonal = 0.0;  ///< 0: sqrt(3) * box_size; WP21 passes the enlarged diagonal
};

struct ExpansionError {
    Real max_relative_error = 0;  ///< max |expansion - exact| / |exact|
    int order = 0;                ///< truncation order used
    Vec3 worst_offset;            ///< C_o - C_s of the worst pair, in units of box_size
};

/// A-priori worst-case check of the expansion (ADR 0008 §6, building block of the lossy-region
/// policy): the plane-wave form of the addition theorem with the quadrature of
/// SphereSampling(order) and T_L is compared with exp(-jkR)/R for the nearest interaction-list
/// configurations (centre offsets (2,0,0), (2,1,0), (2,1,1), (2,2,0), (2,2,1), (2,2,2) box
/// sizes) and observer/source points at all 8 x 8 corner pairs of cubes of diagonal
/// box_diagonal around the two centres (|d| up to box_diagonal). Real k: measured
/// 1e-3 ... 1e-1 at the ADR order for box sizes lambda/4 ... 4 lambda (d/X = 0.87 converges
/// slowly and low-frequency breakdown limits small boxes to ~1e-2 for any order); complex k
/// with large |Im k| D (Ag-like) gives errors >> 1 (exponentially ill-conditioned sampling).
/// @throws std::invalid_argument for invalid k, box_size <= 0, digits <= 0, or
///         box_diagonal < 0; std::overflow_error from spherical_hankel2.
[[nodiscard]] ExpansionError expansion_error(Complex k, Real box_size, Real digits,
                                             const ExpansionErrorOptions& options = {});

}  // namespace specklebem::mlfmm
