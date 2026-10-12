#pragma once
/// @file krylov_spectrum.hpp
/// Spectral information from the GMRES Hessenberg matrix (solver::GmresParams::keep_hessenberg,
/// WP-A2): Ritz values, harmonic Ritz values and singular values of the (k + 1) x k upper
/// Hessenberg H_bar of the Arnoldi relation Op V_k = V_{k+1} H_bar. Shared by the unit tests of
/// solver::gmres and benchmarks/ag_absorption.cpp; no Catch2.
///
///  * Ritz values: eigenvalues of H_k (the leading k x k block); they approximate the exterior
///    eigenvalues of Op first.
///  * Harmonic Ritz values: eigenvalues of H_k + |h_{k+1,k}|^2 H_k^{-H} e_k e_k^T (equivalently
///    H_bar^H H_bar y = theta H_k^H y); they approximate the eigenvalues of Op closest to the
///    origin (the ones that slow GMRES down; Paige, Parlett & van der Vorst 1995).
///  * Singular values of H_bar: sigma_min(H_bar) >= sigma_min(Op) and decreases monotonically
///    with k (H_bar = V_{k+1}^H Op V_k), so a small value is a certified near-null direction of Op.
#include "specklebem/core/types.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace krylov_spectrum {

using specklebem::Complex;
using specklebem::Index;
using specklebem::MatrixXc;
using specklebem::Real;
using specklebem::VectorXc;
using specklebem::VectorXr;

/// Checks the shape (k + 1) x k with k >= 1. @throws std::invalid_argument otherwise.
inline Index hessenberg_dimension(const MatrixXc& hbar) {
    const Index k = hbar.cols();
    if (k < 1 || hbar.rows() != k + 1)
        throw std::invalid_argument("krylov_spectrum: expected a (k + 1) x k Hessenberg, k >= 1");
    return k;
}

/// Eigenvalues of a square matrix. @throws std::runtime_error if the QR iteration fails.
inline VectorXc eigenvalues(const MatrixXc& A) {
    const Eigen::ComplexEigenSolver<MatrixXc> es(A, false);
    if (es.info() != Eigen::Success)
        throw std::runtime_error("krylov_spectrum: eigenvalue iteration did not converge");
    return es.eigenvalues();
}

/// Ritz values (eigenvalues of H_k).
inline VectorXc ritz_values(const MatrixXc& hbar) {
    const Index k = hessenberg_dimension(hbar);
    return eigenvalues(hbar.topRows(k));
}

/// Harmonic Ritz values (eigenvalues of H_k + |h_{k+1,k}|^2 H_k^{-H} e_k e_k^T).
/// @throws std::runtime_error if H_k is singular (a zero harmonic Ritz value is undefined).
inline VectorXc harmonic_ritz_values(const MatrixXc& hbar) {
    const Index k = hessenberg_dimension(hbar);
    const MatrixXc Hk = hbar.topRows(k);
    const Eigen::PartialPivLU<MatrixXc> lu(Hk.adjoint());
    const VectorXc f = lu.solve(VectorXc::Unit(k, k - 1));
    if (!f.allFinite())
        throw std::runtime_error("krylov_spectrum: H_k is singular");
    MatrixXc G = Hk;
    G.col(k - 1) += std::norm(hbar(k, k - 1)) * f;
    return eigenvalues(G);
}

/// Singular values of H_bar in decreasing order.
inline VectorXr singular_values(const MatrixXc& hbar) {
    hessenberg_dimension(hbar);
    const Eigen::BDCSVD<MatrixXc> svd(hbar);
    return svd.singularValues();
}

/// The values sorted by increasing modulus.
inline std::vector<Complex> sorted_by_modulus(const VectorXc& v) {
    std::vector<Complex> out(v.data(), v.data() + v.size());
    std::sort(out.begin(), out.end(),
              [](const Complex& a, const Complex& b) { return std::abs(a) < std::abs(b); });
    return out;
}

}  // namespace krylov_spectrum
