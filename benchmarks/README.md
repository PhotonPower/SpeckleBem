# Benchmarks

Target reference points (Fu et al. 2023, 2x Xeon E5-2627, 12 threads):

| Case | N (unknowns) | Formulation | Iterations | Wall time |
|------|-------------:|-------------|-----------:|----------:|
| Ag sphere, d = 4 µm, λ/27 mesh | 393,216 | precond. | 424 | 188 min |
| Ag surface 10×10 µm², σ = 100 nm | ~240,000 | precond. | – | 46 min |
| Si surface 10×10 µm², σ = 100 nm | ~240,000 | ICTF | – | 12 min |
| Si surface 30×30 µm² | 2,160,000 | ICTF | – | (hardware limit) |

Complexity exponents to beat: γ ≈ 1.35 (Si rough), γ ≈ 1.9 (Ag rough).

Each benchmark writes a JSON record (N, levels, iterations, time per matvec,
peak memory) to `benchmarks/results/` so regressions are visible in CI history.
