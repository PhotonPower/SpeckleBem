#include "specklebem/compression/mlfmm/plane_wave.hpp"

#include "specklebem/kernels/quadrature.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace specklebem::mlfmm {

namespace {

constexpr int kMaxOrder = 100000;
constexpr Complex kJ{0.0, 1.0};

void check_wavenumber(Complex k, const char* where) {
    if (!std::isfinite(k.real()) || !std::isfinite(k.imag()) || k.real() <= 0.0 || k.imag() > 0.0) {
        throw std::invalid_argument(std::string(where) +
                                    ": need finite k with Re k > 0 and Im k <= 0 (exp(+jwt))");
    }
}

bool is_finite(Complex z) {
    return std::isfinite(z.real()) && std::isfinite(z.imag());
}

void check_order(int order, const char* where) {
    if (order < 0 || order > kMaxOrder) {
        throw std::invalid_argument(std::string(where) + ": order must be in 0..100000");
    }
}

}  // namespace

int truncation_order(Complex k, Real box_diagonal, Real digits) {
    check_wavenumber(k, "truncation_order");
    if (!std::isfinite(box_diagonal) || box_diagonal <= 0.0 || !std::isfinite(digits) ||
        digits <= 0.0) {
        throw std::invalid_argument("truncation_order: need box_diagonal > 0 and digits > 0");
    }
    const Real kd = k.real() * box_diagonal;
    const Real l = std::ceil(kd + 1.8 * std::pow(digits, 2.0 / 3.0) * std::cbrt(kd));
    if (!(l <= kMaxOrder)) {
        throw std::invalid_argument("truncation_order: order above 100000 (box too large)");
    }
    return static_cast<int>(l);
}

SphereSampling::SphereSampling(int order) : order_(order) {
    check_order(order, "SphereSampling");
    const int nt = num_theta();
    const int np = num_phi();
    const kernels::LineRule gl = kernels::gauss_legendre(nt);
    theta_.resize(nt);
    phi_.resize(np);
    VectorXr wt(nt);
    for (int i = 0; i < nt; ++i) {  // nodes ascending in cos -> reverse for ascending theta
        const auto src = static_cast<std::size_t>(nt - 1 - i);
        theta_(i) = std::acos(gl.nodes[src]);
        wt(i) = gl.weights[src];
    }
    for (int j = 0; j < np; ++j) {
        phi_(j) = 2.0 * constants::pi * j / np;
    }
    const Index n = size();
    weights_.resize(n);
    khat_.resize(n, 3);
    theta_hat_.resize(n, 3);
    phi_hat_.resize(n, 3);
    for (int i = 0; i < nt; ++i) {
        const Real st = std::sin(theta_(i));
        const Real ct = std::cos(theta_(i));
        for (int j = 0; j < np; ++j) {
            const Real sp = std::sin(phi_(j));
            const Real cp = std::cos(phi_(j));
            const Index q = index(i, j);
            weights_(q) = wt(i) * 2.0 * constants::pi / np;
            khat_.row(q) << st * cp, st * sp, ct;
            theta_hat_.row(q) << ct * cp, ct * sp, -st;
            phi_hat_.row(q) << -sp, cp, 0.0;
        }
    }
}

std::vector<Complex> spherical_hankel2(int max_order, Complex z) {
    check_order(max_order, "spherical_hankel2");
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag()) || z.real() <= 0.0 || z.imag() > 0.0) {
        throw std::invalid_argument("spherical_hankel2: need finite z with Re z > 0, Im z <= 0");
    }
    // |h_0| = e^{Im z} / |z|; below 1e3 DBL_MIN, e^{-jz} is denormal or zero and every h_l
    // would silently lose all digits. Tested in log form (exp itself would underflow).
    constexpr Real kMinAbs = 1e3 * std::numeric_limits<Real>::min();
    if (z.imag() - std::log(std::abs(z)) < std::log(kMinAbs)) {
        throw std::underflow_error(
            "spherical_hankel2: |h_0(z)| = exp(Im z)/|z| underflows double (Im z too negative)");
    }
    std::vector<Complex> h(static_cast<std::size_t>(max_order) + 1);
    const Complex e = std::exp(-kJ * z);
    const Complex inv_z = 1.0 / z;
    h[0] = kJ * e * inv_z;
    if (max_order >= 1) {
        h[1] = e * (kJ * inv_z * inv_z - inv_z);
    }
    for (std::size_t l = 1; l + 1 < h.size(); ++l) {
        h[l + 1] = static_cast<Real>(2 * l + 1) * inv_z * h[l] - h[l - 1];
    }
    for (const Complex& v : h) {
        if (!is_finite(v)) {
            throw std::overflow_error("spherical_hankel2: |h_l(z)| overflows double");
        }
    }
    return h;
}

std::vector<Real> legendre_p(int max_order, Real x) {
    check_order(max_order, "legendre_p");
    if (!(x >= -1.0 && x <= 1.0)) {
        throw std::invalid_argument("legendre_p: x must be in [-1, 1]");
    }
    // push_back instead of indexing a freshly sized vector (GCC 16 -Wnull-dereference false
    // positive).
    std::vector<Real> p;
    p.reserve(static_cast<std::size_t>(max_order) + 1);
    p.push_back(1.0);
    if (max_order >= 1) {
        p.push_back(x);
    }
    for (int l = 1; l < max_order; ++l) {
        const auto lr = static_cast<Real>(l);
        const auto u = static_cast<std::size_t>(l);
        p.push_back(((2.0 * lr + 1.0) * x * p[u] - lr * p[u - 1]) / (lr + 1.0));
    }
    return p;
}

VectorXc translator(Complex k, const Vec3& r, const SphereSampling& sampling) {
    check_wavenumber(k, "translator");
    const Real rn = r.norm();
    if (!(rn > 0.0) || !std::isfinite(rn)) {
        throw std::invalid_argument("translator: need a finite, nonzero offset r");
    }
    const int order = sampling.order();
    // Coefficients c_l = (-j)^l (2l + 1) h_l^{(2)}(k|r|).
    std::vector<Complex> c = spherical_hankel2(order, k * rn);
    Complex mj{1.0, 0.0};
    for (std::size_t l = 0; l < c.size(); ++l) {
        c[l] *= mj * static_cast<Real>(2 * l + 1);
        mj *= -kJ;
        if (!is_finite(c[l])) {
            throw std::overflow_error(
                "translator: coefficient (-j)^l (2l+1) h_l(k|r|) overflows double (order too "
                "high for k|r|: low-frequency breakdown)");
        }
    }
    // Legendre recurrence P_{l+1} = a_l x P_l - b_l P_{l-1}, tables precomputed.
    std::vector<Real> a(c.size()), b(c.size());
    for (std::size_t l = 0; l < c.size(); ++l) {
        const auto lr = static_cast<Real>(l);
        a[l] = (2.0 * lr + 1.0) / (lr + 1.0);
        b[l] = lr / (lr + 1.0);
    }
    const VectorXr cos_gamma = sampling.directions() * (r / rn);
    const Index n = sampling.size();
    VectorXc t(n);
    for (Index q = 0; q < n; ++q) {
        const Real x = std::clamp(cos_gamma.data()[q], -1.0, 1.0);
        Real p_prev = 1.0;
        Real p_cur = x;
        Complex sum = c[0];
        for (std::size_t l = 1; l < c.size(); ++l) {
            sum += c[l] * p_cur;
            const Real p_next = a[l] * x * p_cur - b[l] * p_prev;
            p_prev = p_cur;
            p_cur = p_next;
        }
        if (!is_finite(sum)) {
            throw std::overflow_error("translator: value overflows double");
        }
        t.data()[q] = sum;
    }
    return t;
}

namespace {

/// The six nearest interaction-list centre offsets (units of box_size) up to symmetry.
constexpr std::array<std::array<Real, 3>, 6> kOffsets = {
    {{2, 0, 0}, {2, 1, 0}, {2, 1, 1}, {2, 2, 0}, {2, 2, 1}, {2, 2, 2}}};

Vec3 offset_vec(std::size_t i) {
    return {kOffsets[i][0], kOffsets[i][1], kOffsets[i][2]};
}

void check_expansion_args(Complex k, Real box_size, Real box_diagonal, const char* where) {
    check_wavenumber(k, where);
    if (!std::isfinite(box_size) || box_size <= 0.0 || !std::isfinite(box_diagonal) ||
        box_diagonal < 0.0) {
        throw std::invalid_argument(std::string(where) +
                                    ": need box_size > 0 and box_diagonal >= 0");
    }
}

/// Relative error, with 0/0, inf/inf and overflow reported as unusable (inf).
Real relative_error(Complex approx, Complex exact) {
    const Real err = std::abs(approx - exact) / std::abs(exact);
    return std::isfinite(err) ? err : std::numeric_limits<Real>::infinity();
}

/// Observer offsets o - C_o and source offsets s - C_s of the random check, `pairs` per centre
/// offset (offset-major), uniform in the cube of half edge h. Uniforms from the raw 64-bit
/// mt19937_64 output (53 bits), identical on every standard library.
struct RandomPoints {
    std::vector<Vec3> obs, src;
};

RandomPoints random_points(int pairs, std::uint64_t seed, Real h) {
    std::mt19937_64 rng(seed);
    const auto coord = [&rng, h] {
        return h * (static_cast<Real>(rng() >> 11) * 0x1.0p-52 - 1.0);  // [-h, h)
    };
    RandomPoints p;
    const std::size_t n = kOffsets.size() * static_cast<std::size_t>(pairs);
    p.obs.reserve(n);
    p.src.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const Real ox = coord(), oy = coord(), oz = coord();
        const Real sx = coord(), sy = coord(), sz = coord();
        p.obs.emplace_back(ox, oy, oz);
        p.src.emplace_back(sx, sy, sz);
    }
    return p;
}

/// Statistical check at one order with the given point set.
ExpansionError random_error(Complex k, Real box_size, int order, const RandomPoints& pts) {
    ExpansionError result;
    result.order = order;
    result.worst_offset = offset_vec(0);
    const SphereSampling s(order);
    const Index n = s.size();
    const Real* kx = s.directions().col(0).data();
    const Real* ky = s.directions().col(1).data();
    const Real* kz = s.directions().col(2).data();
    const Real* w = s.weights().data();
    const Complex pref = -kJ * k / (4.0 * constants::pi);
    const std::size_t pairs = pts.obs.size() / kOffsets.size();
    VectorXc wt(n);
    for (std::size_t io = 0; io < kOffsets.size(); ++io) {
        const Vec3 x = offset_vec(io) * box_size;
        const VectorXc t = translator(k, x, s);
        for (Index q = 0; q < n; ++q) {
            wt.data()[q] = w[q] * t.data()[q];
        }
        for (std::size_t m = io * pairs; m < (io + 1) * pairs; ++m) {
            // exp(-jk khat.(o - C_o)) exp(-jk khat.(C_s - s)) = exp(-jk khat.d), d = o' - s'.
            const Vec3 d = pts.obs[m] - pts.src[m];
            Complex sum{0.0, 0.0};
            for (Index q = 0; q < n; ++q) {
                const Real phase = kx[q] * d.x() + ky[q] * d.y() + kz[q] * d.z();
                sum += wt.data()[q] * std::exp(-kJ * k * phase);
            }
            const Real r = (x + d).norm();
            const Real err = relative_error(pref * sum, std::exp(-kJ * k * r) / r);
            if (err > result.max_relative_error) {
                result.max_relative_error = err;
                result.worst_offset = offset_vec(io);
            }
        }
    }
    return result;
}

/// Worst case over the 8 x 8 corner pairs of the cubes of diagonal `diag`.
ExpansionError corner_error(Complex k, Real box_size, Real diag, int order) {
    ExpansionError result;
    result.order = order;
    result.worst_offset = offset_vec(0);
    const SphereSampling s(order);
    const Index n = s.size();
    const SphereSampling::DirectionArray& dirs = s.directions();

    // Corner offsets of a cube with diagonal `diag`; plane waves exp(-jk khat.c) per corner
    // (observer: c = o - C_o; source: C_s - s = -c).
    const Real h = 0.5 * diag / std::sqrt(3.0);
    std::array<Vec3, 8> corners;
    MatrixXc out_wave(n, 8);
    MatrixXc in_wave(n, 8);
    for (int c = 0; c < 8; ++c) {
        corners[static_cast<std::size_t>(c)] =
            h * Vec3((c & 1) ? 1.0 : -1.0, (c & 2) ? 1.0 : -1.0, (c & 4) ? 1.0 : -1.0);
        const VectorXr phase = dirs * corners[static_cast<std::size_t>(c)];
        for (Index q = 0; q < n; ++q) {
            out_wave(q, c) = std::exp(-kJ * k * phase(q));
            in_wave(q, c) = std::exp(kJ * k * phase(q));
        }
    }
    const Complex pref = -kJ * k / (4.0 * constants::pi);
    VectorXc tmp(n);
    const Real* w = s.weights().data();
    for (std::size_t io = 0; io < kOffsets.size(); ++io) {
        const Vec3 x = offset_vec(io) * box_size;
        const VectorXc t = translator(k, x, s);
        for (int cs = 0; cs < 8; ++cs) {
            const Complex* in = in_wave.col(cs).data();
            for (Index q = 0; q < n; ++q) {
                tmp.data()[q] = w[q] * t.data()[q] * in[q];
            }
            for (int co = 0; co < 8; ++co) {
                const Complex* out = out_wave.col(co).data();
                Complex sum{0.0, 0.0};
                for (Index q = 0; q < n; ++q) {
                    sum += out[q] * tmp.data()[q];
                }
                const Real r = (x + corners[static_cast<std::size_t>(co)] -
                                corners[static_cast<std::size_t>(cs)])
                                   .norm();
                const Real err = relative_error(pref * sum, std::exp(-kJ * k * r) / r);
                if (err > result.max_relative_error) {
                    result.max_relative_error = err;
                    result.worst_offset = offset_vec(io);
                }
            }
        }
    }
    return result;
}

}  // namespace

ExpansionError expansion_error(Complex k, Real box_size, Real digits,
                               const ExpansionErrorOptions& options) {
    check_expansion_args(k, box_size, options.box_diagonal, "expansion_error");
    if (options.order < 0 || (options.mode == ExpansionCheck::random && options.pairs < 1)) {
        throw std::invalid_argument("expansion_error: need order >= 0 and pairs >= 1");
    }
    const Real diag = options.box_diagonal > 0.0 ? options.box_diagonal : std::sqrt(3.0) * box_size;
    const int order = options.order > 0 ? options.order : truncation_order(k, diag, digits);
    if (options.mode == ExpansionCheck::corners) {
        return corner_error(k, box_size, diag, order);
    }
    return random_error(k, box_size, order,
                        random_points(options.pairs, options.seed, 0.5 * diag / std::sqrt(3.0)));
}

TruncationSearch search_truncation_order(Complex k, Real box_size, Real digits,
                                         const TruncationSearchOptions& options) {
    check_expansion_args(k, box_size, options.box_diagonal, "search_truncation_order");
    if (options.pairs < 1 || options.patience < 1 || options.max_order < 0) {
        throw std::invalid_argument(
            "search_truncation_order: need pairs >= 1, patience >= 1, max_order >= 0");
    }
    const Real diag = options.box_diagonal > 0.0 ? options.box_diagonal : std::sqrt(3.0) * box_size;
    TruncationSearch result;
    result.formula_order = truncation_order(k, diag, digits);
    const int max_order = options.max_order > 0
                              ? options.max_order
                              : std::min(kMaxOrder, 2 * result.formula_order + 20);
    if (max_order < result.formula_order) {
        throw std::invalid_argument("search_truncation_order: max_order below the formula order");
    }
    const Real target = std::pow(10.0, -digits);
    const RandomPoints pts =
        random_points(options.pairs, options.seed, 0.5 * diag / std::sqrt(3.0));
    const Real inf = std::numeric_limits<Real>::infinity();
    Real best = inf;
    int since_best = 0;
    result.order = result.formula_order;
    result.error = inf;
    for (int order = result.formula_order; order <= max_order; ++order) {
        Real err = inf;
        try {
            err = random_error(k, box_size, order, pts).max_relative_error;
        } catch (const std::overflow_error&) {
            // h_L or T_L overflows double: the breakdown has set in; reported below, not silent.
        }
        ++result.orders_tried;
        if (err <= target) {
            result.achievable = true;
            result.order = order;
            result.error = err;
            return result;
        }
        if (err < best) {
            best = err;
            result.order = order;
            result.error = err;
            since_best = 0;
        } else if (++since_best >= options.patience || !(err <= 10.0 * best)) {
            result.breakdown = true;
            return result;
        }
    }
    return result;
}

}  // namespace specklebem::mlfmm
