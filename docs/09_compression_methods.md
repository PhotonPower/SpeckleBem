# Compression methods

The `CompressionStrategy` interface allows several ways of representing `Z`. This note records the candidates, why MLFMM is first, and the criteria for later choices.

## Candidates

| Method | Kernel knowledge | Complexity (matvec) | Setup | Strengths | Weaknesses |
|---|---|---|---|---|---|
| **MLFMM** (plane-wave) | explicit (Helmholtz) | O(N log N) | O(N log N) | best asymptotics at high frequency; mature for dielectric SIE; the paper's proven path | complex `k` instability for lossy interiors; low-frequency breakdown for very fine meshes; many tuning knobs |
| **ACA / H-matrix** | kernel-independent (needs matrix entries) | O(N log N) – O(N log² N) | O(N log² N) rank-dependent | trivial to apply to any kernel incl. complex `k`; enables approximate H-LU → strong preconditioner / direct solver | rank grows with frequency (oscillatory kernel) → poor for `kD ≫ 1`; memory |
| **H²-matrix / nested bases** | kernel-independent | O(N) | O(N) – O(N log N) | lower memory than H | same frequency limitation unless directional |
| **Directional H²** (Engquist–Ying, Bebendorf–Kuske–Venn) | partial | O(N log N) | O(N log N) | high-frequency capable, kernel-independent | implementation complexity |
| **Butterfly / MLMDA** | kernel-independent | O(N log² N) | O(N^1.5) – O(N log² N) | high-frequency capable, direct solvers possible | memory, complexity |
| **FFT-based (AIM / pFFT)** | explicit | O(N log N) | O(N log N) | simple, GPU-friendly | needs volumetric grid; poor for thin or sparse geometries |

## Why MLFMM first

- Demonstrated on exactly this problem class (paper: 2.16 M unknowns, γ ≈ 1.3–1.35 for Si).
- Best asymptotic cost for the 30–100 λ surfaces targeted.
- The octree built for it is reusable as the cluster tree of H-type methods.

## Planned second method: ACA/H-matrix

Motivations:
1. **Lossy interior operator** (`L2, K2` for Ag): kernel-independent compression avoids the plane-wave instability; the kernel decays so ranks stay small.
2. **Preconditioning**: an approximate H-LU of a low-accuracy H-matrix representation of the whole system is a known remedy for the slow convergence of metallic (and resonant) problems; it can be combined with the MLFMM matvec (H-LU as preconditioner, MLFMM as operator).
3. **Generality**: it works unchanged for the general geometries of Phase 8 and for other kernels (e.g. layered-medium Green's functions, if ever added).

## Selection criteria (to be measured in Phase 8 on identical problems)

- accuracy at fixed matvec tolerance (operator error vs dense)
- setup time, time per matvec, iterations to convergence (with and without H-LU preconditioner)
- peak memory
- scaling exponent γ over N ∈ [5·10⁴, 10⁶] for Si and Ag
- GPU speed-up achievable

## Interface requirements

All strategies implement `CompressionStrategy::build(Problem) → LinearOperator` and must:
- expose `memory_bytes()` and `describe()`,
- pass the matvec-vs-dense test,
- accept the same `kernels::OperatorOptions`, so element integration is identical across methods (otherwise comparisons are meaningless).
