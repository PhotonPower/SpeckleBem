#pragma once
/// @file common.hpp
/// Shared helpers of the pybind11 module: NumPy input validation (errors -> ValueError) and
/// vectorised evaluation of point functions with the GIL released.
#include "specklebem/core/types.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>
#include <utility>

namespace specklebem::python {

namespace py = pybind11;

using MatrixX3c = Eigen::Matrix<Complex, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatrixXrRow = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// Real (float or integer dtype) array -> C-contiguous float64 copy; ValueError otherwise.
inline py::array_t<double, py::array::c_style> real_array(const py::object& obj, const char* what) {
    const auto a = py::array::ensure(obj);
    if (!a || (a.dtype().kind() != 'f' && a.dtype().kind() != 'i' && a.dtype().kind() != 'u')) {
        throw py::value_error(std::string(what) + ": expected a real numeric array");
    }
    return py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(a);
}

/// (n, 3) real array -> Vertices (copy). With `single` != nullptr a (3,) array is accepted as
/// one point and *single is set to true.
inline Vertices points_from_array(const py::object& obj, const char* what, bool* single = nullptr) {
    const auto a = real_array(obj, what);
    const bool one = single != nullptr && a.ndim() == 1 && a.shape(0) == 3;
    if (single != nullptr) {
        *single = one;
    }
    if (!one && (a.ndim() != 2 || a.shape(1) != 3)) {
        throw py::value_error(std::string(what) + ": expected an (n, 3) array");
    }
    const Index n = one ? 1 : static_cast<Index>(a.shape(0));
    return Eigen::Map<const Vertices>(a.data(), n, 3);
}

/// (F, 3) integer array -> Triangles (copy); ValueError for other dtypes or shapes.
inline Triangles triangles_from_array(const py::object& obj) {
    const auto a = py::array::ensure(obj);
    if (!a || (a.dtype().kind() != 'i' && a.dtype().kind() != 'u') || a.ndim() != 2 ||
        a.shape(1) != 3) {
        throw py::value_error("triangles: expected an (F, 3) integer array");
    }
    const auto t = py::array_t<Index, py::array::c_style | py::array::forcecast>::ensure(a);
    return Eigen::Map<const Triangles>(t.data(), static_cast<Index>(t.shape(0)), 3);
}

/// Evaluates the vector function f(Vec3) -> Vec3c at every row of an (n, 3) array (or a
/// single (3,) point) with the GIL released; returns (n, 3) (or (3,)) complex128.
template <class F>
py::object evaluate_at_points(const py::object& points, F&& f) {
    bool single = false;
    const Vertices p = points_from_array(points, "points", &single);
    MatrixX3c out(p.rows(), 3);
    {
        py::gil_scoped_release release;
        for (Index i = 0; i < p.rows(); ++i) {
            out.row(i) = f(Vec3(p.row(i).transpose())).transpose();
        }
    }
    if (single) {
        return py::cast(Vec3c(out.row(0).transpose()));
    }
    return py::cast(std::move(out));
}

void bind_geometry(py::module_& m);
void bind_excitation(py::module_& m);

}  // namespace specklebem::python
