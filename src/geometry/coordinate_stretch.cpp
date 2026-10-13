/// @file coordinate_stretch.cpp
/// Complex coordinate stretch of ADR 0009 (see coordinate_stretch.hpp for the equations).

#include "specklebem/geometry/coordinate_stretch.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace specklebem::geometry {

namespace {

void validate_profile(const StretchProfile& p, const char* axis) {
    const auto fail = [axis](const std::string& what) {
        std::ostringstream os;
        os << "CoordinateStretch: profile " << axis << ": " << what;
        throw std::invalid_argument(os.str());
    };
    if (!std::isfinite(p.start) || !std::isfinite(p.width) || !std::isfinite(p.sigma_max) ||
        !std::isfinite(p.order)) {
        fail("all fields must be finite");
    }
    if (p.start < 0.0) {
        fail("start must be >= 0 (got " + std::to_string(p.start) + " m)");
    }
    if (p.width <= 0.0) {
        fail("width must be > 0 (got " + std::to_string(p.width) + " m)");
    }
    if (p.sigma_max < 0.0) {
        fail("sigma_max must be >= 0 (got " + std::to_string(p.sigma_max) + ")");
    }
    if (p.order < 0.0) {
        fail("order must be >= 0 (got " + std::to_string(p.order) + ")");
    }
}

/// t~ = t - j sgn(t) Int sigma for one axis; the real part is t itself (bit-identical).
Complex stretch_coordinate(const std::optional<StretchProfile>& p, Real t) {
    if (!p || p->sigma_max == 0.0) {
        return {t, 0.0};
    }
    const Real integral = detail::stretch_integral(*p, t);
    if (integral == 0.0) {
        return {t, 0.0};
    }
    return {t, t > 0.0 ? -integral : integral};
}

Complex stretch_derivative(const std::optional<StretchProfile>& p, Real t) {
    if (!p || p->sigma_max == 0.0) {
        return {1.0, 0.0};
    }
    return {1.0, -detail::stretch_sigma(*p, t)};
}

}  // namespace

namespace detail {

Real stretch_sigma(const StretchProfile& p, Real t) {
    const Real d = std::abs(t) - p.start;
    if (!(d > 0.0)) {
        return 0.0;
    }
    if (d >= p.width) {
        return p.sigma_max;
    }
    return p.sigma_max * std::pow(d / p.width, p.order);
}

Real stretch_integral(const StretchProfile& p, Real t) {
    const Real d = std::abs(t) - p.start;
    if (!(d > 0.0)) {
        return 0.0;
    }
    const Real m1 = p.order + 1.0;
    if (d >= p.width) {
        return p.sigma_max * (p.width / m1 + (d - p.width));
    }
    return p.sigma_max * p.width * std::pow(d / p.width, m1) / m1;
}

}  // namespace detail

CoordinateStretch::CoordinateStretch(const StretchProfile& x, const StretchProfile& y) {
    validate_profile(x, "x");
    validate_profile(y, "y");
    x_ = x;
    y_ = y;
    identity_ = false;
}

CoordinateStretch CoordinateStretch::identity() {
    return CoordinateStretch{};
}

bool CoordinateStretch::is_identity() const {
    return identity_ || (x_->sigma_max == 0.0 && y_->sigma_max == 0.0);
}

Vec3c CoordinateStretch::map(const Vec3& r) const {
    if (identity_) {
        return {Complex{r.x(), 0.0}, Complex{r.y(), 0.0}, Complex{r.z(), 0.0}};
    }
    return {stretch_coordinate(x_, r.x()), stretch_coordinate(y_, r.y()), Complex{r.z(), 0.0}};
}

Vec3c CoordinateStretch::jacobian_diag(const Vec3& r) const {
    if (identity_) {
        return {Complex{1.0, 0.0}, Complex{1.0, 0.0}, Complex{1.0, 0.0}};
    }
    return {stretch_derivative(x_, r.x()), stretch_derivative(y_, r.y()), Complex{1.0, 0.0}};
}

bool CoordinateStretch::in_pml(const Vec3& r) const {
    if (identity_) {
        return false;
    }
    return std::abs(r.x()) > x_->start || std::abs(r.y()) > y_->start;
}

}  // namespace specklebem::geometry
