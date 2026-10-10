# Open-Boundary Truncation

## The problem

Every exterior-field problem the solver handles — a current loop's stray field, a
transformer's leakage flux, a capacitor's fringing field — is posed on an
unbounded domain. The mesh is not unbounded. Today the only way to close it is
to put the far boundary at some finite distance `D` and impose a homogeneous
Dirichlet condition (`A_phi = 0` or `V = 0`) there.

That condition is exact only at infinity. At any finite `D` it changes the
answer, and the direction depends on the quantity:

- **Magnetics, `A_phi = 0` (n x A = 0):** flux may not cross the wall, so the
  returning flux is squeezed inside it and inductances come out **low**.
- **Electrostatics, `V = 0`:** a grounded wall at finite distance is a nearer
  return electrode, so a terminal's capacitance at fixed voltage comes out
  **high**. For a sphere of radius `a` inside a grounded sphere of radius `D`,
  `C_D = 4 pi eps a D / (D - a)`, above the isolated `4 pi eps a`.

For a compact source the error falls off with the source's leading
multipole. A current loop is a magnetic dipole: the wall adds a nearly uniform
field of order `mu0 m / D^3` at the source, so its inductance error decays as
`(a/D)^3`. A conductor with net charge is an electric monopole, whose
capacitance error decays only as `a/D` (the sphere formula above).

This is a modeling error, not a discretization error. Refining the mesh does
not reduce it, and a sweep over `D` that remeshes as it goes mixes the two:
separating them needs the source region's mesh held fixed while only the
exterior grows.

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

Every entry is low, as expected of `A_phi = 0`. But the sweep remeshed at
every `D`, so these are total errors, and they do not follow the dipole law:
from `D/a = 10` to `20` the error should fall eightfold and only halves, and
at `D/a = 160`, where truncation is about `(a/D)^3 ~ 2e-7`, it is still
-0.03%. Beyond `D/a ~ 5` the table is dominated by something other than
truncation -- the discretization of the remeshed domain, or the accuracy of
the ring formula itself -- and it cannot be extrapolated to a free-space limit
or used to judge the ring formula.

The controlled study is in `test/test_coaxial_rings.cpp`: the rings sit in a
sphere of radius `D`, and the two homogeneous walls bracket the free-space
value from either side (n x A = 0 low, n x H = 0 high, adding uniform fields
of -2 and +1 times `mu0 m / (4 pi D^3)`), so their weighted mean cancels the
`(a/D)^3` term.

## Current state of the code

- **Electrostatics** supports the Robin far-field closure of option 1 below
  (`robin_coefficient = eps/R` on a sphere of radius `R`), in every geometry.
  It is exact for the monopole term; see the FAQ entry "How do I choose a
  Robin far-field coefficient?".
- **Magnetics:** Dirichlet-at-a-distance is still the only supported far-field
  closure; the magnetic solvers reject Robin during setup.
- The inductance regression tests in `test/test_solvers.cpp`
  (`Magnetostatic loop inductance matches the analytic ring value` and its MQS
  counterpart) use `D/a = 40`, where the truncation error is negligible
  (`(a/D)^3 ~ 2e-5`), and assert 0.5%, which covers the discretization error
  with headroom.

## Possible future fixes

In increasing order of implementation cost.

### 1. Asymptotic / Robin far-field condition

Replace `u = 0` at `D` with a condition encoding the known decay rate of the
exterior solution, e.g. `du/dn + (k/r) u = 0` with `k` chosen for the leading
multipole. For an axisymmetric current loop the exterior `A_phi` is
dipole-like, so a correctly chosen `k` cancels the leading `(a/D)^3` error and
leaves that of the next multipole.

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

Whichever is chosen, its acceptance test should be a truncation sweep with
the source region's mesh held fixed, showing the error falling at the rate the
closure claims.
