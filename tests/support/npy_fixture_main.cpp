/// @file npy_fixture_main.cpp
/// Test helper `specklebem_npy_fixture <dir>`: writes a fixed dataset with
/// io::open_npy_directory into <dir>. tests/python/test_result_writer.py runs it and reads the
/// files back with numpy.load and json.load (NumPy round trip of the .npy writer).
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/io/result_writer.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    using namespace specklebem;
    if (argc != 2) {
        std::cerr << "usage: specklebem_npy_fixture <output directory>\n";
        return 2;
    }
    try {
        const auto w = io::open_npy_directory(argv[1]);
        VectorXc v(5);
        for (Index k = 0; k < 5; ++k) {
            v(k) = Complex(static_cast<Real>(k) + 0.5, -static_cast<Real>(k) / 3.0);
        }
        MatrixXr mr(2, 3);
        for (Index i = 0; i < 2; ++i) {
            for (Index j = 0; j < 3; ++j) mr(i, j) = static_cast<Real>(10 * i + j) + 0.25;
        }
        MatrixXc mc(3, 2);
        for (Index i = 0; i < 3; ++i) {
            for (Index j = 0; j < 2; ++j) {
                mc(i, j) = Complex(static_cast<Real>(i + 1),
                                   0.5 * static_cast<Real>(j) - static_cast<Real>(i));
            }
        }
        w->write_vector("vector", v);
        w->write_vector("empty", VectorXc());
        w->write_matrix("matrix_real", mr);
        w->write_matrix("group/matrix_complex", mc);
        w->write_mesh("mesh", geometry::make_icosphere(0.5e-6, 1));
        w->write_attribute("name", "SpeckleBem \"npy\" \\ \xc3\xa9\n");
        w->write_attribute("wavelength", 500e-9);
        w->write_attribute("third", 1.0 / 3.0);
    } catch (const std::exception& e) {
        std::cerr << "specklebem_npy_fixture: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
