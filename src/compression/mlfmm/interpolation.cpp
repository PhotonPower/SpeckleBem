#include "specklebem/compression/mlfmm/interpolation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace specklebem::mlfmm {

namespace {

/// Lagrange weights of the nodes x[0..p) at the point t.
void lagrange_weights(const std::vector<Real>& x, Real t, Real* w) {
    const std::size_t p = x.size();
    for (std::size_t a = 0; a < p; ++a) {
        Real v = 1.0;
        for (std::size_t b = 0; b < p; ++b) {
            if (b != a) {
                v *= (t - x[b]) / (x[a] - x[b]);
            }
        }
        w[a] = v;
    }
}

}  // namespace

int interpolation_order(Real digits) {
    if (!std::isfinite(digits) || digits <= 0.0 || digits > 5.0) {
        throw std::invalid_argument(
            "interpolation_order: digits must be in (0, 5] (measured range, ADR 0008)");
    }
    return digits <= 3.0 ? 14 : 22;
}

int leaf_sampling_order(int truncation_order, int interpolation_order) {
    if (truncation_order < 0 || interpolation_order < 2) {
        throw std::invalid_argument(
            "leaf_sampling_order: need truncation_order >= 0 and interpolation_order >= 2");
    }
    return std::max(truncation_order, interpolation_order - 1);
}

SphereInterpolator::SphereInterpolator(const SphereSampling& source, const SphereSampling& target,
                                       int order)
    : order_(order),
      nt_src_(source.num_theta()),
      np_src_(source.num_phi()),
      nt_tgt_(target.num_theta()),
      np_tgt_(target.num_phi()) {
    // Theta stencil: e = m - order/2 + a, a < order, with m in [0, n]; it stays in the extended
    // range [-n, 2n) iff order/2 <= n and order - order/2 <= n, i.e. order <= 2n. The phi
    // stencil needs order <= n_phi = 2n distinct periodic nodes: the same bound.
    if (order < 2 || order > 2 * nt_src_ || order > np_src_) {
        throw std::invalid_argument(
            "SphereInterpolator: order must be >= 2 and <= 2 x the number of source theta nodes");
    }
    const auto p = static_cast<std::size_t>(order);
    std::vector<Real> x(p);

    // phi: uniform periodic nodes e * dphi, e any integer, source index e mod np_src.
    const Real dphi = 2.0 * constants::pi / np_src_;
    phi_index_.resize(static_cast<std::size_t>(np_tgt_) * p);
    phi_weight_.resize(phi_index_.size());
    for (std::size_t j = 0; j < static_cast<std::size_t>(np_tgt_); ++j) {
        const Real t = target.phi()(static_cast<Index>(j));
        const auto m = static_cast<int>(std::floor(t / dphi)) + 1;  // nodes < m lie at or below t
        const int start = m - order / 2;
        for (std::size_t a = 0; a < p; ++a) {
            const int e = start + static_cast<int>(a);
            x[a] = e * dphi;
            phi_index_[j * p + a] = ((e % np_src_) + np_src_) % np_src_;
        }
        lagrange_weights(x, t, &phi_weight_[j * p]);
    }

    // theta: extended nodes e in [-n, 2n): e < 0 -> -theta_{-1-e}, e >= n -> 2pi - theta_{2n-1-e}
    // (both with phi + pi); order <= 2n keeps the stencil in that range (checked above).
    const VectorXr& ts = source.theta();
    const int n = nt_src_;
    theta_index_.resize(static_cast<std::size_t>(nt_tgt_) * p);
    theta_weight_.resize(theta_index_.size());
    theta_flip_.resize(theta_index_.size());
    for (std::size_t i = 0; i < static_cast<std::size_t>(nt_tgt_); ++i) {
        const Real t = target.theta()(static_cast<Index>(i));
        const auto m = static_cast<int>(std::upper_bound(ts.data(), ts.data() + n, t) - ts.data());
        const int start = m - order / 2;
        for (std::size_t a = 0; a < p; ++a) {
            const int e = start + static_cast<int>(a);
            const std::size_t q = i * p + a;
            if (e < 0) {
                theta_index_[q] = -1 - e;
                x[a] = -ts(-1 - e);
                theta_flip_[q] = 1;
            } else if (e >= n) {
                theta_index_[q] = 2 * n - 1 - e;
                x[a] = 2.0 * constants::pi - ts(2 * n - 1 - e);
                theta_flip_[q] = 1;
            } else {
                theta_index_[q] = e;
                x[a] = ts(e);
                theta_flip_[q] = 0;
            }
        }
        lagrange_weights(x, t, &theta_weight_[i * p]);
    }
}

void SphereInterpolator::check_sizes(Index in, Index want_in, Index out, Index want_out,
                                     Index ws) const {
    if (in != want_in || out != want_out || ws < workspace_size()) {
        throw std::invalid_argument("SphereInterpolator: input/output/workspace size mismatch");
    }
}

void SphereInterpolator::interpolate(std::span<const Complex> in, std::span<Complex> out,
                                     std::span<Complex> workspace, PoleParity parity) const {
    check_sizes(static_cast<Index>(in.size()), source_size(), static_cast<Index>(out.size()),
                target_size(), static_cast<Index>(workspace.size()));
    const auto p = static_cast<std::size_t>(order_);
    const auto npt = static_cast<std::size_t>(np_tgt_);
    const auto nps = static_cast<std::size_t>(np_src_);
    const std::size_t half = npt / 2;
    const Real flip_sign = parity == PoleParity::odd ? -1.0 : 1.0;
    // Pass 1 (phi): workspace(row, j) for every source theta row.
    for (std::size_t r = 0; r < static_cast<std::size_t>(nt_src_); ++r) {
        const Complex* src = in.data() + r * nps;
        Complex* dst = workspace.data() + r * npt;
        for (std::size_t j = 0; j < npt; ++j) {
            Complex v{0.0, 0.0};
            for (std::size_t b = 0; b < p; ++b) {
                v += phi_weight_[j * p + b] * src[phi_index_[j * p + b]];
            }
            dst[j] = v;
        }
    }
    // Pass 2 (theta), with the pole reflection phi -> phi + pi on the target phi grid.
    std::fill(out.begin(), out.end(), Complex{0.0, 0.0});
    for (std::size_t i = 0; i < static_cast<std::size_t>(nt_tgt_); ++i) {
        Complex* dst = out.data() + i * npt;
        for (std::size_t a = 0; a < p; ++a) {
            const std::size_t q = i * p + a;
            const Complex* row = workspace.data() + theta_index_[q] * static_cast<Index>(npt);
            const std::size_t shift = theta_flip_[q] ? half : 0;
            const Real w = theta_flip_[q] ? flip_sign * theta_weight_[q] : theta_weight_[q];
            for (std::size_t j = 0; j < npt; ++j) {
                const std::size_t js = j + shift < npt ? j + shift : j + shift - npt;
                dst[j] += w * row[js];
            }
        }
    }
}

void SphereInterpolator::anterpolate(std::span<const Complex> in, std::span<Complex> out,
                                     std::span<Complex> workspace, PoleParity parity) const {
    check_sizes(static_cast<Index>(in.size()), target_size(), static_cast<Index>(out.size()),
                source_size(), static_cast<Index>(workspace.size()));
    const auto p = static_cast<std::size_t>(order_);
    const auto npt = static_cast<std::size_t>(np_tgt_);
    const auto nps = static_cast<std::size_t>(np_src_);
    const std::size_t half = npt / 2;
    const Real flip_sign = parity == PoleParity::odd ? -1.0 : 1.0;
    // Transpose of pass 2.
    std::fill(workspace.begin(), workspace.begin() + workspace_size(), Complex{0.0, 0.0});
    for (std::size_t i = 0; i < static_cast<std::size_t>(nt_tgt_); ++i) {
        const Complex* src = in.data() + i * npt;
        for (std::size_t a = 0; a < p; ++a) {
            const std::size_t q = i * p + a;
            Complex* row = workspace.data() + theta_index_[q] * static_cast<Index>(npt);
            const std::size_t shift = theta_flip_[q] ? half : 0;
            const Real w = theta_flip_[q] ? flip_sign * theta_weight_[q] : theta_weight_[q];
            for (std::size_t j = 0; j < npt; ++j) {
                const std::size_t js = j + shift < npt ? j + shift : j + shift - npt;
                row[js] += w * src[j];
            }
        }
    }
    // Transpose of pass 1.
    std::fill(out.begin(), out.end(), Complex{0.0, 0.0});
    for (std::size_t r = 0; r < static_cast<std::size_t>(nt_src_); ++r) {
        const Complex* src = workspace.data() + r * npt;
        Complex* dst = out.data() + r * nps;
        for (std::size_t j = 0; j < npt; ++j) {
            for (std::size_t b = 0; b < p; ++b) {
                dst[phi_index_[j * p + b]] += phi_weight_[j * p + b] * src[j];
            }
        }
    }
}

VectorXc SphereInterpolator::interpolate(const VectorXc& in, PoleParity parity) const {
    VectorXc out(target_size());
    VectorXc ws(workspace_size());
    interpolate(std::span<const Complex>(in.data(), static_cast<std::size_t>(in.size())),
                std::span<Complex>(out.data(), static_cast<std::size_t>(out.size())),
                std::span<Complex>(ws.data(), static_cast<std::size_t>(ws.size())), parity);
    return out;
}

VectorXc SphereInterpolator::anterpolate(const VectorXc& in, PoleParity parity) const {
    VectorXc out(source_size());
    VectorXc ws(workspace_size());
    anterpolate(std::span<const Complex>(in.data(), static_cast<std::size_t>(in.size())),
                std::span<Complex>(out.data(), static_cast<std::size_t>(out.size())),
                std::span<Complex>(ws.data(), static_cast<std::size_t>(ws.size())), parity);
    return out;
}

}  // namespace specklebem::mlfmm
