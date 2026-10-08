#pragma once
/// @file assembler_test_support.hpp
/// Shared helpers of the dense-assembly tests (unit: tests/unit/test_assembler.cpp, validation:
/// tests/validation/test_assembler_mie.cpp): a sphere problem in vacuum with the plane wave of
/// docs/06 (k = +z, E along x), the exact Mie currents projected onto RWG functions
/// (fields_test_support.hpp), the residual of the Galerkin system for them, the symmetry
/// defect and the bistatic-RCS error of a dense solve.
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/kernels/operators.hpp"
#include "specklebem/operator/assembler.hpp"
#include "specklebem/operator/dense_operator.hpp"
#include "specklebem/postprocessing/scattering.hpp"
#include "specklebem/solver/direct.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "fields_test_support.hpp"

namespace assembler_test {

using namespace fields_test;

/// Sphere (radius kRadius) in vacuum, icosphere mesh, plane wave e0 = x, k_hat = z at the
/// vacuum wavelength `lambda` (default 500 nm), formulation `kind`; problem.omega is the
/// excitation's. The projected exact Mie currents are in setup.solution.currents. Icosphere
/// n = 1 at lambda = 1 um has the same h / lambda as n = 2 at 500 nm (fast unit tests).
struct SphereCase {
    SphereCase(const material::Material& sphere, int subdivisions, formulation::Kind kind,
               Real lambda = kLambda)
        : setup(sphere, subdivisions, lambda),
          wave(lambda, Vec3::UnitZ(), Vec3c(1.0, 0.0, 0.0)),
          form(formulation::make_formulation(kind)) {
        setup.problem.formulation = form.get();
        setup.problem.excitation = &wave;
        setup.problem.omega = wave.omega();
    }
    SphereCase(const SphereCase&) = delete;
    SphereCase& operator=(const SphereCase&) = delete;

    [[nodiscard]] op::Problem& problem() { return setup.problem; }
    [[nodiscard]] const VectorXc& exact() const { return setup.solution.currents; }
    [[nodiscard]] Index size() const { return setup.space.size(); }

    MieSetup setup;
    excitation::PlaneWave wave;
    std::unique_ptr<formulation::Formulation> form;
};

/// Dense matrix Z of the problem (moved out of the DenseOperator).
inline MatrixXc assemble(const op::Problem& p) {
    const std::shared_ptr<op::LinearOperator> op = op::DenseStrategy().build(p);
    auto* dense = dynamic_cast<op::DenseOperator*>(op.get());
    if (dense == nullptr) {
        throw std::logic_error("DenseStrategy::build did not return a DenseOperator");
    }
    return std::move(dense->matrix());
}

/// Relative residuals of the E rows (first N) and H rows (last N) of Z x - b, each normalised
/// by the norm of the corresponding half of `b_norm` (the halves have different units).
struct RowResiduals {
    Real e = 0, h = 0;
};

inline RowResiduals row_residuals(const MatrixXc& Z, const VectorXc& x, const VectorXc& b,
                                  const VectorXc& b_norm) {
    const Index N = Z.rows() / 2;
    const VectorXc r = Z * x - b;
    return {r.head(N).norm() / b_norm.head(N).norm(), r.tail(N).norm() / b_norm.tail(N).norm()};
}

/// max(|(Z x - b)_E| / |b_E|, |(Z x - b)_H| / |b_H|). The halves are normalised separately:
/// with the PMCHWT weights |b_E| / |b_H| ~ eta0, so the combined norm |Z x - b| / |b| would be
/// blind to errors in the H rows (MJ, MM blocks).
inline Real residual(const MatrixXc& Z, const VectorXc& x, const VectorXc& b) {
    const RowResiduals r = row_residuals(Z, x, b, b);
    return std::max(r.e, r.h);
}

/// |S - S^T|_F / |S|_F with S = diag(ch I, -ce I) Z (N x N blocks).
inline Real symmetry_defect(const MatrixXc& Z, Complex ch, Complex ce) {
    const Index N = Z.rows() / 2;
    MatrixXc S = Z;
    S.topRows(N) *= ch;
    S.bottomRows(N) *= -ce;
    const MatrixXc St = S.transpose();
    return (S - St).norm() / S.norm();
}

/// Bistatic-RCS error eps_rr of docs/05 (Eq. 7 of the paper) with E_ref = sqrt(sigma_Mie) and
/// E_sie = sqrt(sigma), over na angles theta in [0, pi] of a scattering plane through z: by
/// default the xz-plane (plane normal +y, Mie azimuth phi = 0); the yz-plane is plane normal -x
/// with phi = pi / 2 (post::bistatic_rcs maps theta to k_hat = (0, sin theta, cos theta) there).
inline Real rcs_eps_rr(const post::SurfaceSolution& s, const MieSolution& mie, Index na = 37,
                       const Vec3& plane_normal = Vec3::UnitY(), Real mie_phi = 0.0) {
    VectorXr angles(na);
    for (Index i = 0; i < na; ++i)
        angles(i) = kPi * static_cast<Real>(i) / static_cast<Real>(na - 1);
    const VectorXr sigma = post::bistatic_rcs(s, plane_normal, angles);
    Real sum = 0.0;
    Real max_ref = 0.0;
    for (Index i = 0; i < na; ++i) {
        const Real ref = std::sqrt(mie.bistatic_rcs(angles(i), mie_phi));
        const Real d = ref - std::sqrt(sigma(i));
        sum += d * d;
        max_ref = std::max(max_ref, ref);
    }
    return std::sqrt(sum / static_cast<Real>(na)) / max_ref;
}

/// Test-local formulation that keeps only the two tangential equations of one region
/// (weights a_i = eta_i, b_i = 1/eta_i for that region, 0 for the other; the assembler skips
/// the other region). The combined formulations of Table 1 cancel the jump terms
/// (assembler.cpp), so only a single-region system tests the sign of +-1/2 I.
class SingleRegion final : public formulation::Formulation {
public:
    explicit SingleRegion(int region) : region_(region) {}
    [[nodiscard]] formulation::Kind kind() const override { return formulation::Kind::PMCHWT; }
    [[nodiscard]] std::string name() const override { return "single-region"; }
    [[nodiscard]] formulation::Weights weights(Complex eta1, Complex eta2) const override {
        const Complex z(0.0, 0.0);
        if (region_ == 1)
            return {eta1, z, Complex(1.0) / eta1, z};
        return {z, eta2, z, Complex(1.0) / eta2};
    }

private:
    int region_;
};

/// Global Gram matrix of the jump term, G_mn = <f_m, n x f_n> (N x N).
inline MatrixXc jump_matrix(const basis::RwgSpace& space) {
    const Index N = space.size();
    MatrixXc G = MatrixXc::Zero(N, N);
    Eigen::Matrix<Complex, 3, 3> I;
    for (Index t = 0; t < space.mesh().num_triangles(); ++t) {
        kernels::jump_block(space, t, I);
        const basis::RwgSpace::Support s = space.support(t);
        for (int a = 0; a < s.count; ++a) {
            for (int b = 0; b < s.count; ++b) G(s.n[a], s.n[b]) += I(a, b);
        }
    }
    return G;
}

/// Row residuals of the exact Mie currents for each region's T-EFIE / T-MFIE alone (weights
/// 1), with the assembler's jump signs (ok) and with the jump sign of that region flipped
/// (flipped). Region 1 carries the incident field; region 2 is homogeneous and is normalised
/// by the region-1 right-hand side (E rows by <f, E_inc>, H rows by <f, H_inc>).
struct JumpSignResiduals {
    RowResiduals ok1, ok2, flipped1, flipped2;
};

inline JumpSignResiduals jump_sign_residuals(SphereCase& c) {
    const Index N = c.size();
    const MatrixXc G = jump_matrix(c.setup.space);
    const VectorXc& x = c.exact();
    const SingleRegion r1(1);
    const SingleRegion r2(2);
    op::Problem p1 = c.problem();
    p1.formulation = &r1;
    op::Problem p2 = c.problem();
    p2.formulation = &r2;
    const VectorXc b1 = op::assemble_rhs(p1);
    const VectorXc b2 = op::assemble_rhs(p2);  // zero: no incident field in R2
    MatrixXc Z1 = assemble(p1);
    MatrixXc Z2 = assemble(p2);
    JumpSignResiduals out;
    out.ok1 = row_residuals(Z1, x, b1, b1);
    out.ok2 = row_residuals(Z2, x, b2, b1);
    // Region 1: JM +G/2 -> -G/2, MJ -G/2 -> +G/2; region 2: JM -G/2 -> +G/2, MJ +G/2 -> -G/2.
    Z1.topRightCorner(N, N) -= G;
    Z1.bottomLeftCorner(N, N) += G;
    Z2.topRightCorner(N, N) += G;
    Z2.bottomLeftCorner(N, N) -= G;
    out.flipped1 = row_residuals(Z1, x, b1, b1);
    out.flipped2 = row_residuals(Z2, x, b2, b1);
    return out;
}

/// Dense solve of the case and its eps_rr against Mie.
struct SolveResult {
    Real eps_rr = 0;
    Real residual = 0;        ///< |Z x - b| / |b| of the LU solution
    Real exact_residual = 0;  ///< residual() of the projected exact Mie currents
};

inline SolveResult solve_and_compare(SphereCase& c) {
    const op::DenseOperator Z(assemble(c.problem()));
    const VectorXc b = op::assemble_rhs(c.problem());
    solver::DirectSolveInfo info;
    const VectorXc x = solver::solve_direct(Z, b, &info);
    const post::SurfaceSolution sol{&c.problem(), x, 0.0};
    return {rcs_eps_rr(sol, c.setup.mie), info.residual, residual(Z.matrix(), c.exact(), b)};
}

}  // namespace assembler_test
