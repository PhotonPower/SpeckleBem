// Minimal example: query the reference materials and the formulation recommendation.
#include "specklebem/core/config.hpp"
#include "specklebem/formulation/formulation.hpp"
#include "specklebem/material/material.hpp"

#include <iostream>

int main() {
    using namespace specklebem;
    std::cout << "SpeckleBem " << version_string() << " (openmp=" << has_openmp()
              << ", cuda=" << has_cuda() << ")\n";
    for (auto [name, mat] :
         {std::pair{"Ag", material::silver_500nm()}, std::pair{"Si", material::silicon_500nm()}}) {
        auto rec = formulation::recommend(mat.eps_r);
        std::cout << name << ": n = " << mat.refractive_index() << "  -> "
                  << formulation::make_formulation(rec.kind)->name()
                  << (rec.diagonal_preconditioner ? " + diagonal preconditioner" : "") << "\n";
    }
}
