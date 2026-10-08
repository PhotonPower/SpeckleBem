# Surface integral equations

This document fixes the equations the code implements. Notation follows the reference paper (Fu et al. 2023) with the conventions of docs/06_conventions.md (`exp(+jωt)`, normals into R1).

## Problem statement

A homogeneous object (region R2, `ε2, μ2`) is embedded in a homogeneous background (R1, `ε1, μ1`). An incident field `(E_inc, H_inc)` exists in R1. On the interface S we seek the equivalent surface currents
```
J = n̂ × H ,   M = −n̂ × E      (R1 side; J2 = −J, M2 = −M)
```
from which the scattered fields in both regions follow from the Stratton–Chu representation.

## Operators

For region i with wavenumber `k_i` and Green's function `G_i(r, r') = e^{−jk_i R}/(4πR)`:
```
L_i X (r) = jω μ_i ∫_S G_i X dS'  −  (1/(jω ε_i)) ∇ ∫_S G_i ∇'·X dS'
K_i X (r) = ∫_S ∇G_i × X dS'      (principal value)
```
Scattered fields in region i:
```
E_i^s = −L_i J_i + K_i M_i ,   H_i^s = −K_i J_i − (1/η_i²) L_i M_i     (up to the sign convention of J_i, M_i)
```

## Tangential equations (Eqs. 1–2 of the paper)

Enforcing tangential continuity on S from each side and using the boundary conditions gives, for i = 1, 2,
```
T-EFIE_i :  E_inc,i |_tan =  L_i J |_tan − K_i M |_tan
T-MFIE_i :  H_inc,i |_tan =  K_i J |_tan + (1/η_i²) L_i M |_tan
```
with `E_inc,2 = H_inc,2 = 0`. The principal-value `K` is accompanied by the jump term `±½ n̂ × X`; the sign depends on the side. The code keeps the jump term explicit in `kernels::element_blocks` (identity-like contribution on coincident triangles) and documents the sign for each region in the source.

## Combination (Eqs. 5–6, Table 1)

```
(a1/η1) T-EFIE_1 + (a2/η2) T-EFIE_2 = 0
 b1 η1  T-MFIE_1 +  b2 η2  T-MFIE_2 = 0
```

| Formulation | a_i | b_i | Behaviour (paper) |
|---|---|---|---|
| PMCHWT | η_i | 1/η_i | accurate, symmetric, **does not converge iteratively** |
| ICTF | 2η_i/(η1+η2) | (η1+η2)/(2η_i) | **best for dielectrics (Si)** without preconditioner |
| MCTF | η_i | η1η2/η_i | like ICTF for Si, slower for Ag |
| + diagonal left preconditioner | — | — | **best for metals (Ag)** |

Resulting block system (`x = [J; M]`, `b = [E-part; H-part]`):
```
[ Σ_i (a_i/η_i) L_i        −Σ_i (a_i/η_i) K_i     ] [J]   [ (a1/η1) E_inc |_tan ]
[ Σ_i b_i η_i K_i           Σ_i (b_i/η_i) L_i      ] [M] = [  b1 η1  H_inc |_tan ]
```

## Discretisation

- Flat triangles, RWG basis `f_n` on interior edges, currents `J = Σ x_n f_n`, `M = Σ x_{N+n} f_n`.
- Galerkin testing with the same functions: matrix entries `<f_m, L_i f_n>`, `<f_m, K_i f_n>`.
- The `L` operator is integrated in its mixed-potential form: the gradient is moved onto the test function (`<f_m, ∇φ> = −<∇·f_m, φ>`), so only `G` and `∇G` kernels appear and both are at most `1/R` or `1/R²` singular.
- Quadrature: Dunavant rules, degree chosen by proximity class (`far`, `near`, touching). Touching pairs use singularity subtraction (ADR 0004): the static part of `G` and `∇G` (`1/R`, `∇(1/R)` and, for better smoothness, also the `R` term) is integrated analytically over the source triangle (Hänninen et al. 2006; Wilton et al. 1984; Graglia 1993), the remainder numerically.
- Jump term of `K`: `±½ <f_m, n̂ × f_n>` on coincident triangles.

## Right-hand side

```
b_m         = (a1/η1) ∫ f_m · E_inc dS       (m = 1..N)
b_{N+m}     =  b1 η1  ∫ f_m · H_inc dS
```

## Preconditioning

Left Jacobi preconditioner `M^{-1} = diag(Z)^{-1}` (the paper's "preconditioned formulation"). Block-diagonal (leaf boxes) and Schur-complement variants are planned; see docs/01_project_plan.md.

## What "rigorous" means here

The only approximations are (i) the flat-triangle geometry, (ii) the finite RWG basis, (iii) numerical quadrature and (iv) the compression tolerance of the fast matvec. All four are *numerical* and controllable by refinement/tolerance, and all are measured in docs/05_validation.md. No Kirchhoff/physical-optics, no small-slope, no thin-layer, no periodicity assumptions enter the solver.
