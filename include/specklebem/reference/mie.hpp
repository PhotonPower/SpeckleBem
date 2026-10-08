#pragma once
/// @file mie.hpp
/// Mie series for a homogeneous sphere (Bohren & Huffman conventions adapted
/// to exp(+jwt)). Used as the exact reference for solver validation:
/// near field, far field and bistatic RCS.
#include "specklebem/core/types.hpp"
#include "specklebem/material/material.hpp"

namespace specklebem::reference {

struct MieParams {
    Real radius;
    Real wavelength;
    material::Material sphere;
    material::Material medium = material::vacuum();
    int n_max = 0;  ///< 0 => Wiscombe criterion
};

class MieSolution {
public:
    explicit MieSolution(const MieParams& p);

    /// Scattered field for an x-polarised plane wave travelling in +z.
    [[nodiscard]] Vec3c scattered_E(const Vec3& r) const;
    [[nodiscard]] Vec3c scattered_H(const Vec3& r) const;
    [[nodiscard]] Vec3c internal_E(const Vec3& r) const;

    [[nodiscard]] Real bistatic_rcs(Real theta, Real phi) const;
    [[nodiscard]] Real scattering_cross_section() const;
    [[nodiscard]] Real extinction_cross_section() const;

    [[nodiscard]] const VectorXc& a_n() const { return a_; }
    [[nodiscard]] const VectorXc& b_n() const { return b_; }

private:
    MieParams p_;
    VectorXc a_, b_, c_, d_;
};

}  // namespace specklebem::reference
