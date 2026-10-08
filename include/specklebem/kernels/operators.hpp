#pragma once
/// @file operators.hpp
/// Galerkin matrix elements of the integro-differential operators L_i and K_i
/// of the tangential formulation (Eqs. 1-2 of the paper):
///
///   L_i X = j w mu_i  int G_i X dS'  -  (1 / j w eps_i) grad int G_i div' X dS'
///   K_i X = int grad G_i x X dS'      (principal value; the 1/2 n x X term is
///                                     added separately depending on the side)
///
/// Element routine: for a pair of triangles (t_test, t_src) return the
/// 3x3 block of <f_m, L_i f_n> and <f_m, K_i f_n> for all RWGs on them.
#include "specklebem/basis/rwg.hpp"
#include "specklebem/core/types.hpp"
#include "specklebem/material/material.hpp"

namespace specklebem::kernels {

struct RegionParams {
    Complex k;    ///< wavenumber
    Complex eta;  ///< wave impedance
    Real omega;
    Complex eps;  ///< absolute permittivity
    Complex mu;   ///< absolute permeability
};

struct OperatorOptions {
    int quad_degree_far = 3;    ///< Dunavant degree for well-separated pairs
    int quad_degree_near = 7;   ///< for near pairs (distance < factor * size)
    int quad_degree_sing = 10;  ///< for the smooth remainder after subtraction
    Real near_distance_factor = 2.0;
};

/// Computes the local L and K blocks (3x3 each) for one triangle pair.
void element_blocks(const basis::RwgSpace& space, Index t_test, Index t_src,
                    const RegionParams& region, const OperatorOptions& opt,
                    Eigen::Matrix<Complex, 3, 3>& L, Eigen::Matrix<Complex, 3, 3>& K);

}  // namespace specklebem::kernels
