#pragma once
/// @file simulation.hpp
/// High-level driver that wires everything together. This is also the surface
/// exposed to Python: build a Simulation from a config, run, post-process.
#include "specklebem/compression/mlfmm/mlfmm_operator.hpp"
#include "specklebem/excitation/excitation.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/geometry/mesh.hpp"
#include "specklebem/material/material.hpp"
#include "specklebem/postprocessing/fields.hpp"
#include "specklebem/solver/gmres.hpp"

#include <memory>
#include <string>

namespace specklebem {

struct SimulationConfig {
    Real wavelength = 500e-9;
    material::Material exterior = material::vacuum();
    material::Material object = material::silicon_500nm();
    formulation::Kind formulation = formulation::Kind::ICTF;
    bool diagonal_preconditioner = false;
    std::string compression = "mlfmm";  ///< "dense" | "mlfmm" | "aca" | "hmatrix"
    mlfmm::MlfmmParams mlfmm;
    solver::GmresParams gmres;
    kernels::OperatorOptions kernels;
};

class Simulation {
public:
    Simulation(geometry::TriangleMesh mesh, std::shared_ptr<excitation::Excitation> excitation,
               SimulationConfig config);
    ~Simulation();

    /// Assemble operator and RHS (idempotent).
    void assemble();
    /// Run the iterative (or direct) solve; returns convergence information.
    solver::GmresResult solve();
    [[nodiscard]] const post::SurfaceSolution& solution() const;
    [[nodiscard]] std::string report() const;  ///< timings, memory, iterations

    [[nodiscard]] const geometry::TriangleMesh& mesh() const;
    [[nodiscard]] const SimulationConfig& config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace specklebem
