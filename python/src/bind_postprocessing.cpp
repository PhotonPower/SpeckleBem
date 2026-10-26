/// Bindings of the speckle statistics (postprocessing/speckle_statistics.hpp; WP-S1).
/// Intensities: real arrays of any shape (flattened in C order) for the sample statistics, 2-D
/// arrays (nu, nv) for the grid functions (I.reshape(nu, nv) of a plane_grid field).
#include "specklebem/postprocessing/speckle_statistics.hpp"

#include <sstream>
#include <string>

#include "common.hpp"

namespace specklebem::python {

namespace {

using post::ApertureShape;

/// Real array of any shape -> flattened VectorXr (copy, C order).
VectorXr flat_samples(const py::object& obj, const char* what) {
    const auto a = real_array(obj, what);
    return Eigen::Map<const VectorXr>(a.data(), static_cast<Index>(a.size()));
}

/// 2-D real array (nu, nv) -> column-major MatrixXr with entry (i, j) = a[i, j] (copy).
MatrixXr grid_samples(const py::object& obj, const char* what) {
    const auto a = real_array(obj, what);
    if (a.ndim() != 2) {
        throw py::value_error(std::string(what) + ": expected a 2-D array (nu, nv)");
    }
    return Eigen::Map<const MatrixXrRow>(a.data(), static_cast<Index>(a.shape(0)),
                                         static_cast<Index>(a.shape(1)));
}

/// "square" / "circular" / "gaussian" (any case) or an ApertureShape value; ValueError otherwise.
ApertureShape parse_shape(const py::object& obj) {
    if (py::isinstance<py::str>(obj)) {
        const std::string s = to_lower(obj.cast<std::string>());
        if (s == "square")
            return ApertureShape::Square;
        if (s == "circular" || s == "disc" || s == "disk")
            return ApertureShape::Circular;
        if (s == "gaussian")
            return ApertureShape::Gaussian;
    } else if (py::isinstance<ApertureShape>(obj)) {
        return obj.cast<ApertureShape>();
    }
    throw py::value_error(
        "shape: expected \"square\", \"circular\", \"gaussian\" or an ApertureShape value");
}

}  // namespace

void bind_postprocessing(py::module_& m) {
    py::enum_<ApertureShape>(m, "ApertureShape",
                             "Intensity distribution of the illuminated aperture "
                             "(speckle_size_expected).")
        .value("Square", ApertureShape::Square, "uniform square of side D")
        .value("Circular", ApertureShape::Circular, "uniform disc of diameter D")
        .value("Gaussian", ApertureShape::Gaussian,
               "Gaussian spot |P|^2 ~ exp(-2 r^2 / w0^2), D = 2 w0 (1/e^2 intensity diameter)");

    py::class_<post::KsResult>(m, "KsResult",
                               "Result of ks_test_exponential: statistic D_n, p_value, n, "
                               "modified_statistic; rejects(alpha) is the level-correct decision.")
        .def_readonly("statistic", &post::KsResult::statistic,
                      "D_n = sup |F_n(x) - (1 - exp(-x))|, x = I / <I>.")
        .def_readonly("p_value", &post::KsResult::p_value,
                      "Asymptotic Kolmogorov p-value Q_KS(sqrt(n) D_n). Conservative because the "
                      "mean is estimated from the sample (about 0.5 % false rejections at "
                      "nominal 5 %); use rejects() for a level-correct decision.")
        .def_readonly("n", &post::KsResult::n, "Number of samples.")
        .def_readonly("modified_statistic", &post::KsResult::modified_statistic,
                      "Stephens-modified statistic (D_n - 0.2 / n)(sqrt(n) + 0.26 + 0.5 / "
                      "sqrt(n)) for the exponential with estimated mean (Stephens 1974).")
        .def("rejects", &post::KsResult::rejects, py::arg("alpha") = 0.05,
             "Level-correct decision at level alpha in [0.01, 0.15]: modified_statistic > "
             "stephens_exponential_critical_value(alpha) (True = not negative-exponential). "
             "ValueError outside the table range.")
        .def("__iter__",
             [](const post::KsResult& r) {
                 // (statistic, p_value, n) as in WP-S1, so that existing unpacking keeps working.
                 return py::iter(py::make_tuple(r.statistic, r.p_value, r.n));
             })
        .def("__repr__", [](const post::KsResult& r) {
            std::ostringstream os;
            os.precision(6);
            os << "KsResult(statistic=" << r.statistic << ", p_value=" << r.p_value << ", n=" << r.n
               << ", modified_statistic=" << r.modified_statistic << ")";
            return os.str();
        });
    m.def("stephens_exponential_critical_value", &post::stephens_exponential_critical_value,
          py::arg("alpha"),
          "Upper-tail critical value of KsResult.modified_statistic (exponential, mean "
          "estimated; Stephens 1974): 0.926, 0.990, 1.094, 1.190, 1.308 at alpha = 0.15, 0.10, "
          "0.05, 0.025, 0.01, log-linear in alpha between them. ValueError outside [0.01, 0.15].");

    m.def(
        "speckle_contrast",
        [](const py::object& I) { return post::speckle_contrast(flat_samples(I, "intensity")); },
        py::arg("intensity"),
        "Speckle contrast C = std(I) / <I> (population moments; any array shape, flattened). 1 "
        "for fully developed polarised speckle. ValueError for < 2 samples, negative or "
        "non-finite values, or <I> = 0.");
    m.def(
        "correlation_coefficient",
        [](const py::object& I1, const py::object& I2) {
            return post::correlation_coefficient(flat_samples(I1, "I1"), flat_samples(I2, "I2"));
        },
        py::arg("I1"), py::arg("I2"),
        "Correlation coefficient <dI1 dI2> / sqrt(<dI1^2> <dI2^2>) of two patterns over "
        "corresponding samples (e.g. the angular correlation gamma_12); |mu|^2 for two jointly "
        "circular Gaussian fields with field correlation mu. ValueError for mismatched sizes, "
        "negative or non-finite values, or a constant pattern.");
    m.def(
        "intensity_pdf",
        [](const py::object& I, int bins) {
            const post::Histogram h = post::intensity_pdf(flat_samples(I, "intensity"), bins);
            return py::make_tuple(h.bin_centers, h.density);
        },
        py::arg("intensity"), py::arg("bins") = 64,
        "(bin_centers, density) histogram of x = I / <I> on [0, max x] (density integrates to "
        "1); compare with exp(-x).");
    m.def(
        "intensity_pdf_rms_error",
        [](const VectorXr& bin_centers, const VectorXr& density) {
            return post::intensity_pdf_rms_error(post::Histogram{bin_centers, density});
        },
        py::arg("bin_centers"), py::arg("density"),
        "RMS difference between an intensity_pdf histogram and exp(-x) at the bin centres "
        "(diagnostic).");
    m.def(
        "ks_test_exponential",
        [](const py::object& I) { return post::ks_test_exponential(flat_samples(I, "intensity")); },
        py::arg("intensity"), R"doc(
One-sample Kolmogorov-Smirnov test of x = I / <I> against 1 - exp(-x) (fully developed
polarised speckle).

Parameters
----------
intensity : array_like
    Non-negative intensities, any shape (flattened), n >= 50 independent samples: decimate a
    speckle pattern to about two speckle sizes per sample first (correlated pixels give
    spurious rejections; specklebem.speckle.ensemble_statistics does this).

Returns
-------
KsResult
    ``statistic`` (D_n), ``p_value`` (asymptotic Kolmogorov distribution,
    2 sum_k (-1)^(k-1) exp(-2 k^2 n D_n^2)), ``n``, ``modified_statistic`` (Stephens:
    (D_n - 0.2 / n)(sqrt(n) + 0.26 + 0.5 / sqrt(n))) and ``rejects(alpha=0.05)``; unpacks as the
    tuple (statistic, p_value, n). The mean is estimated from the sample, which makes the plain
    p-value strongly conservative (about 0.5 % false rejections at nominal 5 %; p-values cluster
    towards 1). ``rejects(alpha)`` is the level-correct test for this case (Stephens-modified
    statistic against stephens_exponential_critical_value(alpha); Stephens 1974, D'Agostino and
    Stephens 1986, chapter 4).

Raises
------
ValueError
    For n < 50, negative or non-finite values, or <I> = 0.
)doc");
    m.def("kolmogorov_survival", &post::kolmogorov_survival, py::arg("lam"),
          "Kolmogorov survival function Q_KS(lam) = 2 sum_k (-1)^(k-1) exp(-2 k^2 lam^2) = "
          "P(sqrt(n) D_n > lam) for n -> inf.");
    m.def(
        "intensity_autocorrelation",
        [](const py::object& I) {
            const MatrixXr g = grid_samples(I, "intensity");
            py::gil_scoped_release release;
            return post::intensity_autocorrelation(g);
        },
        py::arg("intensity"),
        "Normalised intensity autocorrelation Gamma_I(d) = <I(r) I(r + d)> / <I>^2 of a 2-D "
        "pattern (nu, nv) (linear correlation, averaged over the overlapping pixels). Same shape; "
        "entry [i, j] is the lag (i - nu // 2, j - nv // 2) pixels (numpy.fft.fftshift layout).");
    m.def(
        "field_correlation_from_intensity",
        [](const py::object& I) {
            const MatrixXr g = grid_samples(I, "intensity");
            py::gil_scoped_release release;
            return post::field_correlation_from_intensity(g);
        },
        py::arg("intensity"),
        "|mu_A(d)| = sqrt(max(Gamma_I(d) - 1, 0)) (Siegert relation, fully developed speckle) "
        "of a 2-D pattern (nu, nv); layout of intensity_autocorrelation, |mu_A(0)| = contrast.");
    m.def(
        "speckle_size",
        [](const py::object& mu_A, Real pixel_size, int axis) {
            return post::speckle_size(grid_samples(mu_A, "mu_A"), pixel_size, axis);
        },
        py::arg("mu_A"), py::arg("pixel_size"), py::arg("axis") = 0,
        "Speckle size [m]: FWHM of |mu_A|^2 through zero lag along axis 0 (u, rows) or 1 (v), "
        "half of the zero-lag value, pixel_size the grid spacing [m] along that axis. Compare "
        "with speckle_size_expected. RuntimeError if |mu_A|^2 does not drop below half within "
        "the grid.");
    m.def(
        "speckle_size_constant",
        [](const py::object& shape) { return post::speckle_size_constant(parse_shape(shape)); },
        py::arg("shape"),
        "c in FWHM(|mu_A|^2) = c lambda z / D: 0.885893 (square), 1.028994 (circular), "
        "1.060041 (gaussian, D = 2 w0).");
    m.def(
        "speckle_size_expected",
        [](Real wavelength, Real z, Real aperture, const py::object& shape) {
            return post::speckle_size_expected(wavelength, z, aperture, parse_shape(shape));
        },
        py::arg("wavelength"), py::arg("z"), py::arg("aperture"), py::arg("shape") = "square",
        R"doc(
Predicted speckle size c lambda z / D [m] in the convention of speckle_size (FWHM of |mu_A|^2).

Parameters
----------
wavelength : float
    Wavelength in the observation medium [m].
z : float
    Distance aperture - observation plane [m] (> 0; Fraunhofer zone z >> D^2 / lambda,
    paraxial observation).
aperture : float
    D [m]: side (square), diameter (circular) or 2 w0 (gaussian, 1/e^2 intensity diameter).
shape : {"square", "circular", "gaussian"} or ApertureShape
    c = 0.885893, 1.028994, 1.060041 (speckle_size_constant).
)doc");
}

}  // namespace specklebem::python
