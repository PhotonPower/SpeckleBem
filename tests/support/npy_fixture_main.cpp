/// @file npy_fixture_main.cpp
/// Test helper `specklebem_npy_fixture <dir>`: writes a fixed dataset with
/// io::open_npy_directory into <dir>. tests/python/test_result_writer.py runs it and reads the
/// files back with numpy.load and json.load (NumPy round trip of the .npy writer).
#include "specklebem/core/path.hpp"
#include "specklebem/geometry/sphere.hpp"
#include "specklebem/io/result_writer.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// windows.h before shellapi.h
#include <shellapi.h>
#endif

namespace {
/// Output directory as UTF-8 (ADR 0007). On Windows the narrow argv is lossy (ANSI code page)
/// and libstdc++'s path(const char*) throws on non-ASCII bytes, so take the UTF-16 command line.
std::string output_dir_utf8([[maybe_unused]] char** argv) {
#ifdef _WIN32
    int wargc = 0;
    wchar_t** wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (wargv == nullptr || wargc < 2)
        throw std::runtime_error("cannot read the UTF-16 command line");
    const std::filesystem::path dir(wargv[1]);
    LocalFree(wargv);
    return specklebem::core::path_to_utf8(dir);
#else
    return argv[1];  // POSIX: narrow paths are UTF-8 bytes
#endif
}
}  // namespace

int main(int argc, char** argv) {
    using namespace specklebem;
    if (argc != 2) {
        std::cerr << "usage: specklebem_npy_fixture <output directory>\n";
        return 2;
    }
    try {
        const auto w = io::open_npy_directory(output_dir_utf8(argv));
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
