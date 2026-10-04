# Open-Boundary Truncation

## The problem

Every exterior-field problem the solver handles — a current loop's stray field, a
transformer's leakage flux, a capacitor's fringing field — is posed on an
unbounded domain. The mesh is not unbounded. Magnetics currently closes it
with a distant homogeneous Dirichlet boundary (`A_phi = 0`).
Electrostatics also supports a Robin far-field approximation.

That condition is exact only at infinity. At any finite `D` it is wrong, and it
the sign depends on the formulation and drive. At fixed voltage, a sphere
of radius a inside a grounded sphere of radius R has
`C_R = 4 pi epsilon a R/(R-a) > C_infinity`: capacitance is **high**.
For closed magnetic current sources at fixed current, a distant tangential
Dirichlet closure generally lowers inductance.

This is a modelling error, not a discretization error. Refining the mesh does
not reduce it — you can drive the FE error to round-off and still be off by a
percent because the domain itself is wrong. That distinction matters when
interpreting any validation result against a closed-form reference.

## What we measured

Using the `examples/current_loop` geometry (loop radius `a = 0.1 m`, square
cross-section `w = 2 mm`), sweeping the far-field distance and comparing the
computed self-inductance against Grover's ring formula
(`L = mu_0 * a * (ln(8a/r_eq) - 2)` with `r_eq = 0.2235*(w+h)`, analytic value
`6.0277e-07 H`):

| Domain `D` [m] | `D/a` | `L` [H] | Error |
|---|---|---|---|
| 0.3 | 3 | 5.9615e-07 | -1.10% |
| 0.5 | 5 | 5.9983e-07 | -0.49% |
| 1.0 | 10 | 6.0158e-07 | -0.20% |
| 2.0 | 20 | 6.0212e-07 | -0.11% |
| 4.0 | 40 | 6.0238e-07 | -0.06% |
| 8.0 | 80 | 6.0252e-07 | -0.04% |
| 16.0 | 160 | 6.0261e-07 | -0.03% |

This historical table mixes discretization, truncation, and reference-model
errors; it has not been reproduced here. An empirical 1/D fit does not separate
them or prove the accuracy of the geometric-mean-distance approximation.
For a closed magnetic dipole, B decays as 1/r^3 and exterior energy as 1/D^3.
Self-similar distant-boundary expansion therefore gives a leading O(D^-3)
inductance truncation error, not a universal 1/D law.

`test/test_coaxial_rings.cpp` uses the dipole correction
`(L_tangent + 2 L_normal)/3`. Boundary expansion and mesh refinement must still
be studied separately before attributing the residual to either error source.

## Current state of the code

- **Electrostatics** supports the Robin far-field closure of option 1 below
  (`robin_coefficient = eps/R` on a sphere of radius `R`), in every geometry.
  It is exact for the monopole term; see the FAQ entry "How do I choose a
  Robin far-field coefficient?".
- **Magnetics:** Dirichlet-at-a-distance is still the only supported far-field
  closure; the magnetic solvers reject Robin during setup.
- The inductance regression tests in `test/test_solvers.cpp`
  (`Magnetostatic loop inductance matches the analytic ring value` and its MQS
  counterpart) use `D/a = 40` and assert 0.5%. That tolerance is set by the
  -0.06% truncation bias plus mesh effects, with headroom; it is not a limit of
  the formulation.

## Possible future fixes

In increasing order of implementation cost.

### 1. Asymptotic / Robin far-field condition

Replace `u = 0` at `D` with a condition encoding the known decay rate of the
exterior solution, e.g. `du/dn + (k/r) u = 0` with `k` chosen for the leading
multipole. For an axisymmetric current loop the exterior `A_phi` is
dipole-like (`A_phi` decays as 1/r^2). A suitable spherical closure removes
the leading dipole truncation; remaining multipoles and boundary shape set
the subsequent rate. The scalar electrostatic Robin term cannot simply be
reused as the magnetic natural boundary operator.

- **Pro:** by far the cheapest. The config plumbing and `RobinCoeff` field
  already exist; this is a boundary-integrator addition plus lifting the setup
  rejection. No mesh-generation changes.
- **Con:** assumes the sources are far from the boundary and that a single
  decay exponent dominates. Degrades when the geometry is elongated or when
  multiple well-separated source groups exist. Choosing `k` per-problem is a
  usability wart unless it can be inferred.
- **Best for:** terminal quantities (L, C, R) on compact source regions — which
  is most of what the coupling-matrix path produces.

### 2. Kelvin transformation

Mesh a second, inverted region representing the exterior via the map
`r -> R^2/r`, and couple it to the interior mesh by constraining matching DOFs
on the shared interface.

- **Pro:** exact for the exterior Laplace/Poisson problem — removes truncation
  error entirely rather than reducing its order. Well established for
  magnetostatics.
- **Con:** requires the mesh generator to emit a conforming companion region,
  and requires DOF-pairing machinery the codebase does not have. The
  axisymmetric measure needs care under the inversion.
- **Best for:** cases where exterior field values matter, not just terminal
  quantities.

### 3. Infinite elements / ballooning

Elements with shape functions that decay appropriately toward infinity, or
repeated outward mesh scaling.

- **Pro:** most accurate and most general.
- **Con:** most invasive — custom element types and integration rules, touching
  assembly throughout.

## Recommendation

Option 1 is the right next step *when* stray-field or leakage accuracy starts
limiting real work. Until then, moving the boundary out is adequate and honest,
provided the truncation bias is stated rather than silently absorbed into a
loose tolerance.

Option 2 is only worth the effort if exterior field distributions — not just
lumped terminal parameters — become a deliverable.

Whichever is chosen, rerun independent mesh refinement and self-similar domain
expansion studies. The historical 1/D fit is not an acceptance test.
