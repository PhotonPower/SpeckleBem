/// @file fft.cpp
/// 2D row-column DFT and the 5-smooth padding length (see fft.hpp). The code is the former
/// anonymous-namespace copy of geometry/rough_surface.cpp and postprocessing/
/// speckle_statistics.cpp (identical in both, WP-T3), unchanged so that the results stay
/// bitwise identical. The only additions, for shapes neither caller passes: the early return for
/// a matrix without entries and skipping the passes of length 1 (identity).
#include "specklebem/core/fft.hpp"

#include <unsupported/Eigen/FFT>

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <vector>

namespace specklebem::core {

void fft2(MatrixXc& a, bool inverse) {
    if (a.size() == 0)
        return;  // nothing to transform (a length-0 kissfft plan is not defined)
    Eigen::FFT<Real> fft;
    const Index rows = a.rows();
    const Index cols = a.cols();
    std::vector<Complex> in(static_cast<std::size_t>(std::max(rows, cols)));
    std::vector<Complex> out(in.size());
    // A length-1 DFT is the identity; Eigen's kissfft backend crashes on length 1 (seen in the
    // WP-T3 tests), so such a pass is skipped. Neither caller transforms length 1.
    for (Index c = 0; rows > 1 && c < cols; ++c) {
        for (Index r = 0; r < rows; ++r) in[static_cast<std::size_t>(r)] = a(r, c);
        if (inverse)
            fft.inv(out.data(), in.data(), rows);
        else
            fft.fwd(out.data(), in.data(), rows);
        for (Index r = 0; r < rows; ++r) a(r, c) = out[static_cast<std::size_t>(r)];
    }
    for (Index r = 0; cols > 1 && r < rows; ++r) {
        for (Index c = 0; c < cols; ++c) in[static_cast<std::size_t>(c)] = a(r, c);
        if (inverse)
            fft.inv(out.data(), in.data(), cols);
        else
            fft.fwd(out.data(), in.data(), cols);
        for (Index c = 0; c < cols; ++c) a(r, c) = out[static_cast<std::size_t>(c)];
    }
}

Index next_smooth(Index n) {
    for (Index m = std::max<Index>(n, 1);; ++m) {
        Index r = m;
        for (const Index p : {Index{2}, Index{3}, Index{5}}) {
            while (r % p == 0) r /= p;
        }
        if (r == 1)
            return m;
    }
}

}  // namespace specklebem::core
