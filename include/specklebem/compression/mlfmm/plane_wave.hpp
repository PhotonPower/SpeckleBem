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

#include <cstdint>
#include <vector>

namespace specklebem::mlfmm {

/// Truncation order L = ceil(kD + 1.8 d0^{2/3} (kD)^{1/3}) with k -> Re k for the bandwidth
/// (excess-bandwidth formula, ADR 0008 §2). Measured accuracy (tests/unit/test_plane_wave.cpp):
/// for random points inside two boxes of edge a, D = sqrt(3) a, centres >= 2a apart, the
/// expansion error is <= 10^-d0 for real k and ka >= ~3 (a >= lambda/2); at a = lambda/4 it is
/// up to ~2e-3 (d0 = 3) / ~3e-4 (d0 = 5) depending on the points: the formula is too small at
/// low kD, hence search_truncation_order (ADR 0008 amendment) starting from this value.
/// Corner-to-corner points of the nearest interaction-list pair (d/X = 0.87) converge much
/// more slowly (see expansion_error()).
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
/// Domain: Re z > 0, Im z <= 0 (real z included), and |h_0(z)| = e^{Im z}/|z| >= 1e3 DBL_MIN
/// (about Im z >= -701 - ln|z|), so that h_0 and h_1 are normal doubles (|h_l| >= |h_0| for
/// l >= 1 in this half plane); outside, e^{-jz} underflows and the values would silently become
/// zero. The recurrence is stable for h^{(2)} (no minimal solution is sought; perturbations stay
/// O(eps |h_l|) in the oscillatory range l < |z| and are dominated by the growing y_l part for
/// l > |z|, where |h_l| ~ (2l-1)!!/|z|^{l+1}).
/// Measured relative error of the complex value h_l <= 1.4e-14 for l <= 120 and |z| in
/// [0.05, 80] (real, and Im z down to -3.1), against the Bessel-polynomial closed form, a long
/// double recurrence, the Wronskian for real z, and for complex z up to |z| = 80 against
/// j_l (Miller downward recurrence) - j y_l (upward) in long double with their Wronskian
/// (tests/unit/test_plane_wave.cpp). The real part j_l(z) is NOT accurate where
/// |j_l| << |y_l| (l >> |z|); the translator needs only h_l.
/// @throws std::invalid_argument outside the domain Re z > 0, Im z <= 0 or for max_order < 0;
///         std::underflow_error if |h_0(z)| < 1e3 DBL_MIN (e.g. z = 10 - 740j);
///         std::overflow_error if |h_l| overflows double (e.g. |z| = 0.1, l > ~150).
[[nodiscard]] std::vector<Complex> spherical_hankel2(int max_order, Complex z);

/// Legendre polynomials P_l(x), l = 0..max_order, by the three-term recurrence
/// (l+1) P_{l+1} = (2l+1) x P_l - l P_{l-1} (stable on [-1, 1]).
/// @throws std::invalid_argument for max_order < 0 or x outside [-1, 1] (or not finite).
[[nodiscard]] std::vector<Real> legendre_p(int max_order, Real x);

/// Diagonal translator T_L(k, r, khat) at every direction of `sampling`, with L =
/// sampling.order() (see file comment; the addition-theorem prefactor -jk/4pi is NOT included).
/// Cost O(L * sampling.size()), no allocation per direction.
/// @throws std::invalid_argument for |r| = 0 or k outside the spherical_hankel2 domain;
///         std::underflow_error / std::overflow_error from spherical_hankel2;
///         std::overflow_error if a coefficient c_l = (-j)^l (2l+1) h_l(k|r|) or a translator
///         value is not finite (|h_L| close to DBL_MAX, low-frequency breakdown).
[[nodiscard]] VectorXc translator(Complex k, const Vec3& r, const SphereSampling& sampling);

/// Point sets of the a-priori expansion check (expansion_error).
enum class ExpansionCheck {
    /// All 8 x 8 corner pairs of the two cubes: deterministic worst case, very conservative
    /// (d/X = 0.87 for offset (2,0,0); never meets 10^-d0 at the formula order, see below).
    corners,
    /// `pairs` seeded random observer/source pairs per offset, uniform in the two cubes: the
    /// statistical per-level check of the ADR 0008 amendment (WP18 measurements).
    random,
};

struct ExpansionErrorOptions {
    int order = 0;            ///< 0: truncation_order(k, box_diagonal, digits)
    Real box_diagonal = 0.0;  ///< 0: sqrt(3) * box_size; WP21 passes the enlarged diagonal
    ExpansionCheck mode = ExpansionCheck::corners;
    int pairs = 8;  ///< random mode: observer/source pairs per offset (>= 1; 6 offsets)
    /// random mode: seed of the std::mt19937_64 stream; uniforms are formed from its raw 64-bit
    /// output (not std::uniform_real_distribution), so the point set is identical on every
    /// standard library, and the same for every `order` (the order search compares like with
    /// like).
    std::uint64_t seed = 20261009;
};

struct ExpansionError {
    Real max_relative_error = 0;  ///< max |expansion - exact| / |exact| (inf if not finite)
    int order = 0;                ///< truncation order used
    Vec3 worst_offset;            ///< C_o - C_s of the worst pair, in units of box_size
};

/// A-priori check of the expansion (ADR 0008 §2/§6 and its WP18 amendment): the plane-wave form
/// of the addition theorem with the quadrature of SphereSampling(order) and T_L is compared with
/// exp(-jkR)/R for the nearest interaction-list configurations (centre offsets X = (2,0,0),
/// (2,1,0), (2,1,1), (2,2,0), (2,2,1), (2,2,2) x box_size), with observer/source points in the
/// cubes of diagonal box_diagonal (edge box_diagonal / sqrt 3) around the two centres.
///
/// - mode corners: all 8 x 8 corner pairs (|d| up to box_diagonal). Real k: measured
///   9e-2 ... 1e-3 (d0 = 3) and 4e-2 ... 2e-4 (d0 = 5) at the formula order for box sizes
///   lambda/4 ... 4 lambda, i.e. above 10^-d0 everywhere; raising the order helps only for
///   ka >~ 10 (minimum over L ~1e-2 at lambda/4, ~1e-3 at lambda, ~2e-4 at 2 lambda: low-
///   frequency breakdown).
/// - mode random: `pairs` seeded random pairs per offset (the norm-wise error that the docs/05
///   matvec criterion measures). Real k at the formula order: <= 10^-d0 for box edges
///   >= lambda/2 (d0 = 3, 5); at lambda/4 up to one digit worse (see search_truncation_order).
///
/// Complex k with large |Im k| (Ag-like) gives errors >> 1 at the formula order in both modes:
/// the formula sees only Re k; search_truncation_order finds usable orders for small boxes,
/// and from |Im k| D ~ 17 on the sampling is exponentially ill-conditioned at every order.
/// @throws std::invalid_argument for invalid k, box_size <= 0, digits <= 0, box_diagonal < 0,
///         order < 0, or pairs < 1 in random mode; std::underflow_error / std::overflow_error
///         from spherical_hankel2 / translator.
[[nodiscard]] ExpansionError expansion_error(Complex k, Real box_size, Real digits,
                                             const ExpansionErrorOptions& options = {});

struct TruncationSearchOptions {
    Real box_diagonal = 0.0;        ///< 0: sqrt(3) * box_size (as ExpansionErrorOptions)
    int pairs = 8;                  ///< random pairs per offset of the statistical check
    std::uint64_t seed = 20261009;  ///< seed of the statistical check
    int max_order = 0;              ///< highest order tried; 0: 2 * formula + 20
    /// Breakdown detection: stop after this many consecutive orders without a new minimum of
    /// the error, or as soon as the error exceeds 10 x its minimum so far.
    int patience = 4;
};

struct TruncationSearch {
    bool achievable = false;  ///< an order meeting 10^-d0 was found
    int order = 0;            ///< achievable: the smallest such order; else the best order tried
    Real error = 0;           ///< statistical-check error at `order`
    int formula_order = 0;    ///< truncation_order(k, box_diagonal, digits), the search start
    int orders_tried = 0;     ///< number of orders evaluated
    /// Not achievable: true if the search stopped because the error grew again (breakdown
    /// onset, including a translator overflow), false if it reached max_order.
    bool breakdown = false;
};

/// Truncation order per level (ADR 0008 amendment, WP18 measurements): the smallest L >=
/// truncation_order(k, box_diagonal, digits) for which the statistical check
/// (expansion_error, mode random, same seeded points for every L) meets 10^-digits. If the
/// error stops decreasing first (no new minimum for `patience` orders, an error above 10 x the
/// minimum, or a translator overflow), the target is reported as not achievable (`achievable`
/// false, `breakdown` true; `order` / `error` are the best seen): the level must not be used as
/// a leaf level (coarsen it), or for complex k the ADR 0008 §6 alternatives apply.
/// Measured (tests/unit/test_plane_wave.cpp, default options): real k: the formula order for
/// box edges >= lambda/2 (d0 = 3, 5) and lambda/4 at d0 = 3; lambda/4 at d0 = 5: formula + 1;
/// lambda/4 at d0 = 7 and lambda/8 at d0 = 6: not achievable (minimum ~4e-7 / ~1e-5).
/// Si-like k (n = 4.3 - 0.07j) from lambda_0/4: the formula order. Ag-like k (n = 0.05 -
/// 3.13j), d0 = 3: a = 0.05 ... 0.25 lambda_0 at L = 9 ... 22 (formula 2 - 3); from
/// 0.5 lambda_0 not achievable; d0 = 5 not achievable.
/// @throws std::invalid_argument for invalid arguments (as expansion_error, plus patience < 1
///         or 0 < max_order < formula order); std::underflow_error from spherical_hankel2
///         when e^{Im k |X|} underflows (an interaction far below any accuracy target).
[[nodiscard]] TruncationSearch search_truncation_order(Complex k, Real box_size, Real digits,
                                                       const TruncationSearchOptions& options = {});

}  // namespace specklebem::mlfmm
