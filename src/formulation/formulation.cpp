#include "specklebem/formulation/formulation.hpp"

#include <stdexcept>

namespace specklebem::formulation {

namespace {

class Pmchwt final : public Formulation {
public:
    Kind kind() const override { return Kind::PMCHWT; }
    std::string name() const override { return "PMCHWT"; }
    Weights weights(Complex eta1, Complex eta2) const override {
        return {eta1, eta2, Complex(1) / eta1, Complex(1) / eta2};
    }
};

class Ictf final : public Formulation {
public:
    Kind kind() const override { return Kind::ICTF; }
    std::string name() const override { return "ICTF"; }
    Weights weights(Complex eta1, Complex eta2) const override {
        const Complex s = eta1 + eta2;
        return {Complex(2) * eta1 / s, Complex(2) * eta2 / s, s / (Complex(2) * eta1),
                s / (Complex(2) * eta2)};
    }
};

class Mctf final : public Formulation {
public:
    Kind kind() const override { return Kind::MCTF; }
    std::string name() const override { return "MCTF"; }
    Weights weights(Complex eta1, Complex eta2) const override {
        return {eta1, eta2, eta1 * eta2 / eta1, eta1 * eta2 / eta2};
    }
};

}  // namespace

std::unique_ptr<Formulation> make_formulation(Kind kind) {
    switch (kind) {
        case Kind::PMCHWT:
            return std::make_unique<Pmchwt>();
        case Kind::ICTF:
            return std::make_unique<Ictf>();
        case Kind::MCTF:
            return std::make_unique<Mctf>();
        case Kind::JMCFIE:
            throw std::logic_error("JMCFIE not implemented yet");
    }
    throw std::logic_error("unknown formulation");
}

Kind parse_kind(const std::string& name) {
    if (name == "PMCHWT" || name == "pmchwt")
        return Kind::PMCHWT;
    if (name == "ICTF" || name == "ictf")
        return Kind::ICTF;
    if (name == "MCTF" || name == "mctf")
        return Kind::MCTF;
    if (name == "JMCFIE" || name == "jmcfie")
        return Kind::JMCFIE;
    throw std::invalid_argument("unknown formulation: " + name);
}

Recommendation recommend(Complex eps_r_object) {
    // Fu et al. 2023, Sec. 4: ICTF (no preconditioner) for dielectrics,
    // diagonal left preconditioner for metals with large negative Re(eps_r).
    if (eps_r_object.real() < 0)
        return {Kind::ICTF, true};
    return {Kind::ICTF, false};
}

}  // namespace specklebem::formulation
