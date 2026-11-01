#pragma once
/// @file pml_route_case.hpp
/// The fast PML route of the rough-surface studies (shared by benchmarks/preconditioner_study.cpp
/// and benchmarks/profile_study.cpp): the configuration of pml_rough_ag::make_config(c) switched
/// to compression "mlfmm" with full GMRES (no restart) at d0 digits and the given tolerance, as
/// WP-PML5b / WP-PC1 / WP-SC3..SC5 ran it. The preconditioner settings are left as make_config
/// leaves them (empty: Simulation chooses the route's default, since WP-SC4 the ILUT).
#include "pml_rough_ag_support.hpp"

namespace pml_route {

using namespace specklebem;

[[nodiscard]] inline SimulationConfig fast_config(const pml_rough_ag::RoughCase& c, Real digits,
                                                  Real tolerance, int max_iter) {
    SimulationConfig cfg = pml_rough_ag::make_config(c);
    cfg.compression = "mlfmm";
    cfg.solver = SolverKind::Gmres;
    cfg.mlfmm.accuracy_digits = digits;
    cfg.gmres.tolerance = tolerance;
    cfg.gmres.max_iter = max_iter;
    cfg.gmres.restart = 0;
    return cfg;
}

}  // namespace pml_route
