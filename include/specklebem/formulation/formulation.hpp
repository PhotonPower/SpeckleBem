#pragma once
/// @file formulation.hpp
/// Linear combinations of the tangential EFIE / MFIE for both regions
/// (Eqs. 5-6, Table 1 of Fu et al. 2023):
///
///   (a1/eta1) T-EFIE_1 + (a2/eta2) T-EFIE_2 = 0
///    b1 eta1  T-MFIE_1 +  b2 eta2  T-MFIE_2 = 0
///
///   PMCHWT : a_i = eta_i,               b_i = 1/eta_i
///   ICTF   : a_i = 2 eta_i/(eta1+eta2), b_i = (eta1+eta2)/(2 eta_i)
///   MCTF   : a_i = eta_i,               b_i = eta1 eta2 / eta_i
///
/// The formulation only decides scalar weights (and optionally a diagonal left
/// preconditioner); it never touches the operator implementation, so any
/// compression scheme works with any formulation.
#include "specklebem/core/types.hpp"

#include <memory>
#include <string>

namespace specklebem::formulation {

enum class Kind { PMCHWT, ICTF, MCTF, JMCFIE /* reserved */ };

struct Weights {
    Complex a1, a2, b1, b2;
};

class Formulation {
public:
    virtual ~Formulation() = default;
    [[nodiscard]] virtual Kind kind() const = 0;
    [[nodiscard]] virtual std::string name() const = 0;
    /// Scalar weights from the wave impedances of the two regions.
    [[nodiscard]] virtual Weights weights(Complex eta1, Complex eta2) const = 0;
};

std::unique_ptr<Formulation> make_formulation(Kind kind);
Kind parse_kind(const std::string& name);

/// Material-based default from the paper's findings: ICTF without preconditioner for
/// dielectrics (Re eps_r > 0); diagonal preconditioner for metals (Re eps_r < 0). With the
/// Jacobi preconditioner PMCHWT, ICTF and MCTF are the same system (block-row scalings of
/// PMCHWT, docs/03, issue #15), so the returned kind is immaterial for metals; ICTF is
/// returned for uniformity.
struct Recommendation {
    Kind kind;
    bool diagonal_preconditioner;
};
[[nodiscard]] Recommendation recommend(Complex eps_r_object);

}  // namespace specklebem::formulation
