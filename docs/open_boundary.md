# Open-Boundary Truncation

## The problem

Every exterior-field problem the solver handles — a current loop's stray field, a
transformer's leakage flux, a capacitor's fringing field — is posed on an
unbounded domain. The mesh is not unbounded. Today the only way to close it is
to put the far boundary at some finite distance `D` and impose a homogeneous
Dirichlet condition (`A_phi = 0` or `V = 0`) there.

That condition is exact only at infinity. At any finite `D` it is wrong, and it
is wrong in a specific direction: forcing the potential to zero too early
removes field energy that physically extends past the boundary, so terminal
quantities derived from that energy come out **low**.

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

Three conclusions:

1. **The error is one-sided and decays as `1/D`.** Every entry is negative and
   halving the error requires doubling the domain. That is a poor exchange rate:
   the extra volume is nearly all air, and in 2D axisymmetric geometry the
   element count grows with it even under aggressive radial grading.

2. **The limit is the analytic value.** Richardson-extrapolating consecutive
   pairs under the assumed `L(D) = L_inf - C/D` model gives `L_inf = 6.0265e-07 H`,
   within **-0.013%** of Grover's formula. The solver reproduces the closed form
   to about a part in 10^4 once truncation is removed.

3. **The GMD approximation is not a meaningful contributor here.** An earlier
   reading of this example attributed the residual jointly to truncation and to
   the geometric-mean-distance substitution in the ring formula. The
   extrapolation above refutes that: the GMD form is accurate to ~0.01% for this
   aspect ratio, and essentially the entire observed deviation is boundary
   truncation. Do not blame the analytic reference for what the mesh boundary is
   doing.

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
dipole-like, so a correctly chosen `k` cancels the leading `1/D` term and
typically leaves `1/D^3`.

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

Whichever is chosen, the convergence study above should be re-run as the
acceptance test: the `1/D` sweep and its extrapolated limit are the evidence
that the closure works.
