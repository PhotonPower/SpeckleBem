#pragma once
/// @file coordinate_stretch.hpp
/// Complex coordinate stretching (perfectly matched layer, PML) for the open rough surface
/// of ADR 0009, decision 2.
///
/// The map r~ = S(r) = (x~(x), y~(y), z) acts separately on x and y. For the time convention
/// exp(+j w t) (docs/06_conventions.md):
///
///   x~ = x                                         for |x| <= x_p,
///   x~ = x - j sgn(x) Int_{x_p}^{|x|} sigma(t) dt   for |x| >  x_p,
///   sigma(t) = sigma_max ((t - x_p) / W)^m          for x_p < t <= x_p + W,
///   sigma(t) = sigma_max                            for t > x_p + W,
///
/// with x_p = StretchProfile::start, W = StretchProfile::width and m = StretchProfile::order.
/// Hence Im x~ = -sgn(x) sigma_max W / (m + 1) at |x| = x_p + W, and an outgoing wave
/// exp(-j k sgn(x) x) (k > 0) is damped by exp(-k |Im x~|) in both directions. The lead's
/// exp(-i w t) code uses x~ = x + i Int sigma; the sign flips with the time convention.
///
/// Beyond x_p + W the stretch keeps growing linearly with slope sigma_max (sigma stays at its
/// end value, so x~ is C^1 at x_p + W for every m; at x_p it is C^1 for m > 0 and only C^0 for
/// m = 0, where sigma jumps: a step profile that reflects once discretised; for 0 < m < 1
/// sigma' is unbounded at x_p). The open surface mesh ends at x_p + W
/// (OpenSurfaceMesh::outer_half), so the continuation only matters for points outside the
/// mesh (e.g. near-field evaluation points); it keeps the map smooth and monotone in Im.
///
/// The Jacobian is diagonal: DS = diag(s_x, s_y, 1), s_x = dx~/dx = 1 - j sigma(|x|) (the
/// sgn(x) of the map cancels against d|x|/dx). The Galerkin integrals of the stretched
/// problem keep the real mesh, the real quadrature and the real RWG functions; they only gain
/// the complex distance |S(r) - S(r')| and the DS factors of the Piola transform (ADR 0009,
/// decision 2; kernels are WP-PML2).

#include "specklebem/core/types.hpp"

#include <optional>

namespace specklebem::geometry {

/// Polynomial PML profile along one axis (see coordinate_stretch.hpp). All lengths in metres.
struct StretchProfile {
    Real start{};      ///< x_p >= 0: |t| where the stretch begins (OpenSurfaceMesh::pml_start)
    Real width{};      ///< W > 0: PML width (OpenSurfaceMesh::outer_half - pml_start)
    Real sigma_max{};  ///< >= 0, dimensionless, typical 3-5 (ADR 0009 lead decision 2)
    Real order = 2;    ///< polynomial order m >= 0 (m < 1: see above; accepted for profile scans)
};

/// Complex coordinate stretch r~ = S(r) = (x~(x), y~(y), z) for exp(+j w t).
///
/// Every member is cheap (a few flops and one pow per stretched axis) and thread-safe; the
/// map does not check its argument (non-finite input propagates).
class CoordinateStretch {
public:
    /// @throws std::invalid_argument if a profile has a non-finite field, start < 0,
    ///         width <= 0, sigma_max < 0 or order < 0.
    CoordinateStretch(const StretchProfile& x, const StretchProfile& y);

    /// No stretching anywhere: map() returns r with zero imaginary parts (bit-identical real
    /// parts), jacobian_diag() returns (1, 1, 1) and in_pml() is false everywhere.
    static CoordinateStretch identity();

    /// True for identity() and for profiles with sigma_max = 0 on both axes (map() then
    /// returns r exactly, as for identity(); in_pml() still reports the geometric zone).
    [[nodiscard]] bool is_identity() const;

    /// r~ = (x~(x), y~(y), z).
    [[nodiscard]] Vec3c map(const Vec3& r) const;

    /// Diagonal of DS: (dx~/dx, dy~/dy, 1) = (1 - j sigma(|x|), 1 - j sigma(|y|), 1).
    [[nodiscard]] Vec3c jacobian_diag(const Vec3& r) const;

    /// True if |x| > x.start or |y| > y.start (strictly; the start line itself is unstretched).
    /// Always false for identity().
    [[nodiscard]] bool in_pml(const Vec3& r) const;

    /// Profile along x (empty for identity()).
    [[nodiscard]] const std::optional<StretchProfile>& profile_x() const { return x_; }
    /// Profile along y (empty for identity()).
    [[nodiscard]] const std::optional<StretchProfile>& profile_y() const { return y_; }

private:
    CoordinateStretch() = default;

    std::optional<StretchProfile> x_;
    std::optional<StretchProfile> y_;
    bool identity_ = true;
};

namespace detail {

/// Internal, exposed for tests; not part of the public API.
/// sigma(|t|) of the profile p (0 for |t| <= p.start).
[[nodiscard]] Real stretch_sigma(const StretchProfile& p, Real t);
/// Int_{p.start}^{|t|} sigma(u) du (0 for |t| <= p.start), so t~ = t - j sgn(t) of it.
[[nodiscard]] Real stretch_integral(const StretchProfile& p, Real t);

}  // namespace detail

}  // namespace specklebem::geometry
