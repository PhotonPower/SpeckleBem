"""SpeckleBem - rigorous light scattering from rough surfaces (SIE + fast BEM).

Headless Python front end for the C++ core. See docs/10_python_api.md.
"""
from ._specklebem import (  # noqa: F401
    Formulation,
    LogLevel,
    Material,
    __version__,
    has_blas_lapack,
    has_cuda,
    has_hdf5,
    has_openmp,
    recommend_formulation,
    set_log_level,
    silicon_500nm,
    silver_500nm,
    vacuum,
)

__all__ = [
    "Formulation", "LogLevel", "Material", "__version__",
    "has_blas_lapack", "has_cuda", "has_hdf5", "has_openmp",
    "recommend_formulation", "set_log_level", "silicon_500nm", "silver_500nm", "vacuum",
]
