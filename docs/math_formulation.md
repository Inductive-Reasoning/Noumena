# Mathematical Formulation

This document describes the mathematical formulation for the three physics types supported by MFEM-ElectroMag: Electrostatics, Magnetostatics, and Magnetoquasistatics, all in the axisymmetric coordinate system.

## Coordinate System

We use cylindrical coordinates `(r, z, φ)` where:
- `r` is the radial distance from the axis of symmetry
- `z` is the axial coordinate
- `φ` is the azimuthal angle

For axisymmetric problems, all field quantities are independent of `φ` (`∂/∂φ = 0`).

`r` and `z` are **metres**. The formulation is SI throughout: material
properties are per-metre (`ε₀` in F/m, `μ₀` in H/m) and the measure below uses
`r` as a physical length, so the unit of the mesh is part of the formulation
rather than a presentation choice. See
[Units](config_reference.md#units).

## Integration Measure and Output Units

The geometry modes differ only in the measure applied during assembly, and
that difference propagates directly into the units of every extracted quantity.
The full convention, including which `2*pi` factors are *not* part of the
measure, is given under "Integration Measure Convention" below.

- **Axisymmetric.** Revolving the meridional `(r, z)` domain through the full
  azimuthal angle gives the volume element `dV = 2*pi*r dr dz`, and the
  meridional boundary element `2*pi*r ds`. Every axisymmetric integrator applies
  this measure, so energies, charges, and flux linkages are absolute quantities
  for the complete revolved body. Coupling matrices are therefore in farads,
  henries, and ohms.

- **Planar.** The model is translationally invariant in the out-of-plane
  direction and represents an infinitely long structure. Assembly integrates
  over the cross-section only, which is equivalent to taking a unit out-of-plane
  depth. Extracted quantities are consequently **per unit length**: capacitance
  in F/m, inductance in H/m, and resistance in Ohm/m. To obtain absolute values
  for a structure of finite length `L`, multiply by `L` (valid only where end
  effects are negligible, which is the assumption the planar model already
  makes).

- **3D (`geometry_type: 3d`).** Assembly integrates over the true volume of a
  3D mesh, so extracted quantities are absolute (F, H, Ohm). Electrostatics
  keeps its scalar potential; the magnetic formulations above are
  scalar-potential 2D reductions, so 3D magnetics uses separate solvers in a
  vector (H(curl)) potential (the "3D Form" sections).

The measure, the matching scalar integrators, the required mesh dimension and
the output units are all defined in one place, `GeometryModel`
(`src/solvers/geometry_model.hpp`).

No extrusion length is configurable, so planar results are always reported per
unit length. Output labels reflect this: `PhysicsSolver::CouplingUnitLabel`
appends `/m` in planar mode only, so a written matrix is never ambiguous about which
convention produced it.

## 1. Electrostatics

### Strong Form

The electrostatic problem solves for the electric potential `V(r,z)`:

```
-∇ · (ε ∇V) = ρ
```

where:
- `V` is the electric potential [V]
- `ε = ε_r ε₀` is the permittivity [F/m]
- `ρ` is the space charge density [C/m³]
- `ε₀ = 8.854 × 10⁻¹² F/m` is the permittivity of free space

Note that the source term is `ρ`, not `ρ/ε₀`: the absolute permittivity `ε`
already appears inside the divergence, so dividing by `ε₀` as well would be
dimensionally inconsistent. (`ρ/ε₀` belongs to the homogeneous free-space form
`-∇²V = ρ/ε₀`, which is a different equation.)

### Weak Form

Find `V ∈ H¹` such that for all test functions `v`:

```
∫_Ω ε ∇V · ∇v dΩ = ∫_Ω ρ v dΩ + ∫_∂Ω g v dS
```

**Implementation status:** no domain charge integrator is assembled, so the
solver evaluates the source-free case `ρ = 0`, i.e. a Laplace problem driven
entirely by terminal and boundary data. The `∫_Ω ρ v dΩ` term is retained above
to document the complete formulation, not to describe present behavior.

### Axisymmetric Form

In cylindrical coordinates with axisymmetry:

```
∫₀^{z_max} ∫₀^{r_max} ε (∂V/∂r · ∂v/∂r + ∂V/∂z · ∂v/∂z) · 2πr dr dz
```

The factor `2πr` comes from the volume element in cylindrical coordinates when integrating over the full azimuthal angle. The implementation applies this measure
in full; see [Integration Measure Convention](#integration-measure-convention).

### Derived Quantities

After solving for `V`:

**Electric field:**
```
E⃗ = -∇V = -(∂V/∂r r̂ + ∂V/∂z ẑ)
```

**Energy density:**
```
u = ½ ε |E⃗|²
```

### Boundary Conditions

- **Dirichlet:** `V = V₀` on `∂Ω_D` (e.g., electrode surfaces)
- **Neumann:** `n̂ · (ε ∇V) = g` on `∂Ω_N`. The configured `value`
  is this outward natural flux and is added to the weak-form boundary RHS.
  A zero value is the implicit natural condition and requires no assembled term.
- **Robin:** `n̂ · (ε ∇V) + α V = g` on `∂Ω_R`, with `α = robin_coefficient`
  ≥ 0 and `g = value`. It adds `∫ α V v dS` to the operator (under the
  geometry's measure) and `∫ g v dS` to the RHS. Charge extraction uses the
  domain stiffness alone, so the Robin term never enters `Q = K₀ V`.

For axisymmetric problems, a nonzero Neumann load is integrated with the
meridional boundary measure `2πr ds`, the same full measure carried by the
stiffness operator.

## 2. Magnetostatics

### Strong Form

The magnetostatic problem solves for the magnetic vector potential `A(r,z)`. In 2D axisymmetry, only the azimuthal component `A_φ` is non-zero:

```
∇ × (ν ∇ × A⃗) = J⃗
```

where:
- `A⃗ = A_φ(r,z) φ̂` is the magnetic vector potential [Wb/m or T·m]
- `ν = 1/μ` is the reluctivity [H⁻¹/m]
- `μ = μ_r μ₀` is the permeability [H/m]
- `μ₀ = 4π × 10⁻⁷ H/m` is the permeability of free space
- `J⃗ = J_φ(r,z) φ̂` is the current density [A/m²]

### Weak Form

Find `A_φ ∈ H¹` such that for all test functions `v`:

```
∫_Ω ν (∇ × A⃗) · (∇ × v) dΩ = ∫_Ω J⃗ · v dΩ
```

### Axisymmetric Form

The curl of the azimuthal vector potential is:

```
∇ × A⃗ = ∇ × (A_φ φ̂) = -∂A_φ/∂z r̂ + (1/r ∂(rA_φ)/∂r) ẑ
                       = -∂A_φ/∂z r̂ + (A_φ/r + ∂A_φ/∂r) ẑ
```

The weak form becomes:

```
∫₀^{z_max} ∫₀^{r_max} ν [(∂A_φ/∂z)(∂v/∂z)
                         + (∂A_φ/∂r + A_φ/r)(∂v/∂r + v/r)] · 2πr dr dz
     = ∫₀^{z_max} ∫₀^{r_max} J_φ · v · 2πr dr dz
```

The second bracket must be kept as the full product. Expanding it gives

```
(∂A_φ/∂r)(∂v/∂r) + (A_φ/r)(v/r) + (∂A_φ/∂r)(v/r) + (A_φ/r)(∂v/∂r)
```

and the last two cross terms are **not** negligible. Dropping them is only
possible at the cost of a compensating boundary contribution, so the shortened
form `(∂A_φ/∂r)(∂v/∂r) + A_φ·v/r²` is not equivalent.
`AxisymmetricCurlCurlIntegrator` assembles the complete product.

### Behavior at r = 0

Two distinct questions are easy to conflate here, and the distinction determines
what the code may legitimately do.

**The constrained solution is regular.** Axis regularity forces `A_φ → 0`
linearly as `r → 0`, so for the physical solution

```
lim_{r→0} A_φ/r = ∂A_φ/∂r|_{r=0}
```

exists and is finite, and the axial flux density has the exact limit
`B_z → 2 ∂A_φ/∂r`.

**Individual basis functions are not.** A single Lagrange shape function need
not vanish at `r = 0`, so no such limit exists for it, and `∫ N_j N_k / r`
genuinely diverges logarithmically for the shape functions that do not vanish
there. There is nothing to substitute.

Assembly therefore does **not** clamp or substitute anything. Standard interior
quadrature keeps `r > 0` even on elements that touch the axis, so the `1/r` term
is integrated as written. Regularity is delivered instead by essential-BC
elimination of `A_φ = 0` on the axis, which removes exactly the divergent basis
directions. The `B_z → 2 ∂A_φ/∂r` limit is applied only during field recovery
(`ComputeElementFlux` and `MagneticFieldCoefficient`), which operates on the
constrained solution where the limit is valid.

Quadrature order for the `1/r` term is chosen from element geometry rather than
basis degree; see `AxisymmetricCurlCurlIntegrator::RadialExtraOrder` and finding
M5 in `docs/math_review_findings.md`.

### Derived Quantities

**Magnetic flux density:**
```
B⃗ = ∇ × A⃗ = [-∂A_φ/∂z, (A_φ/r + ∂A_φ/∂r), 0] in (r, z, φ) components
```

**Magnetic field:**
```
H⃗ = ν B⃗
```

**Energy density:**
```
u = ½ B⃗ · H⃗ = ½ ν |B⃗|²
```

### Boundary Conditions

- **Dirichlet:** `A_φ = A₀` on `∂Ω_D` (e.g., far-field boundaries, symmetry planes)
- **Neumann:** The configured `value` is the outward natural flux `g`; zero is
  implicit and nonzero data contributes to the boundary RHS. In the planar
  scalar formulation, `g = ν n̂ · ∇A_z`. In the axisymmetric curl-curl
  formulation,
  `g = ν[n_r(∂A_φ/∂r + A_φ/r) + n_z ∂A_φ/∂z]`. The `A_φ/r` contribution is
  part of the radial natural flux and cannot be replaced by `n̂ · (ν∇A_φ)`.
- **Robin:** Electrostatics only; the magnetic solvers reject it.

For an axisymmetric magnetic problem that reaches `r = 0`, regularity requires
`A_φ = 0` on the axis. The solver detects that boundary and applies this
essential constraint automatically; it is not a natural Neumann condition.

### 3D Form

`geometry_type: 3d` uses a separate formulation (`MagnetostaticSolver3D`),
because in 3D the potential is a full vector field:

```
∇ × (ν ∇ × A) = J,   A ∈ H(curl),   B = ∇ × A
```

`A` is discretized with Nédélec (edge) elements, so the tangential component of
`A`, and therefore the normal component of `B`, is continuous across every face.
Boundary conditions are homogeneous only: `dirichlet` 0 means `n × A = 0` (flux
tangent: `B·n = 0`), and `neumann` 0 or no entry means the natural `n × H = 0`
(flux normal). In an eddy-current solve an `n × A = 0` wall also acts as a
perfect electrical contact for any conductor touching it (see the 3D
magnetoquasistatic form).

The curl-curl operator is singular: it annihilates every gradient.

- `iterative` (MPI/HYPRE build): CG preconditioned by hypre's AMS solves the
  singular system directly. It is consistent because every load is projected
  (below). Afterwards the gradient part of `A` is removed,
  `A ← A − Gψ` with `(GᵀMG)ψ = GᵀMA`, which is the discrete Coulomb gauge.
- `direct`: the factorization needs a nonsingular matrix, so `β (A, w)` is
  added with `β = 10⁻⁶ ν_min / L²` (`L` the bounding-box diagonal). With a
  divergence-free source this selects the same Coulomb gauge and perturbs `B`
  by about 10⁻⁶ relative.

Every current terminal is a conductor with a direction field `w = −∇v`,
from a unit conduction potential `v` (analytic `w = φ̂ / (Θr)` for
`azimuthal`, `Θ` the conductor's angular extent: `2π` for a full ring, less
for a sector bounded by symmetry planes; solved on the conductor for `electrodes` and `cut`, with `σ`
weighting for a massive conductor). A stranded conductor carries
`J = (I / A_cs) w / |w|` with `A_cs = ∫|w| dV` (for azimuthal,
`∫ dV / (Θr)`, the meridional area); a massive one its DC distribution
`J = σ w I / G` with `G = ∫σ|w|² dV` its conductance. Each unit-current
load `b` is made discretely divergence-free before use: with `G` the discrete
gradient from the matching H1 space and `M_c` the Nédélec mass matrix over the
terminal's own conductor, `(GᵀM_cG) ψ = Gᵀ b` and `b' = b − M_c G ψ`, with `ψ`
free on the conductor's DOFs except where it touches an `n × A = 0` wall. Then
`Gᵀ b' = 0` for every gradient of the mesh (`b'` only sees a gradient through
the conductor's DOFs), and the correction `−∇ψ` stays inside the conductor with
no normal flux at its surface, so any imbalance is redistributed within it
rather than made up by current in the surroundings. It carries no current
along the path (`∫∇ψ·w = 0` for the harmonic path `w`). The fraction of the
current density removed is reported when above 2%.

The stranded direction `w/|w|` is not divergence-free in general, even though
`w` is: `∇·(w/|w|) = −(w/|w|)·∇ln|w|`, which vanishes only where `|w|` is
constant along each field line (an azimuthal path, a straight bar). On a
solved path through a cut or between electrodes `|w|` varies along the field
lines wherever the conductor turns at different radii across its section --
a racetrack's corners, where it grows like `1/r` inwards -- and the projection
then removes that part, so the source is the projected field rather than a
uniform winding current. In TEAM 7's racetrack coil this is 2.6% on any mesh
(see `examples/team7/README.md` for the measurements), 4.7% in TEAM 21a's.
Its effect on the results is about the square of that fraction: solving with
the exact winding current instead moves TEAM 7's fields by 0.12-0.17% and the
TEAM 21a losses by 0.3-0.4%. A construction that avoids it is an open issue. The flux linkage of
terminal `k` is `λ_k = ∫ A · J_k dV = b'_k · A` (with `J_k` its unit-current
density), and the inductance matrix `L = B'ᵀ K⁻¹ B'` is symmetric by
construction.

## 3. Magnetoquasistatics (Eddy Currents)

### Strong Form

The time-harmonic magnetoquasistatic problem includes eddy current effects:

```
∇ × (ν ∇ × A⃗) + jω σ A⃗ = J⃗_source
```

where:
- `ω = 2πf` is the angular frequency [rad/s]
- `σ` is the electrical conductivity [S/m]
- `J⃗_source` is the applied current density [A/m²]
- `j = √(-1)` is the imaginary unit

For time-harmonic fields (`e^{jωt}`), `A⃗` becomes complex: `A⃗ = A⃗_real + j A⃗_imag`.

### Weak Form

Find complex `A_φ ∈ H¹` such that for all test functions `v`:

```
∫_Ω ν (∇ × A⃗) · (∇ × v*) dΩ + jω ∫_Ω σ A⃗ · v* dΩ = ∫_Ω J⃗_source · v* dΩ
```

where `v*` is the complex conjugate of the test function.

### Axisymmetric Form

This becomes a complex-valued system:

```
[K + jωM] {A} = {F}
```

where:
- `K` is the stiffness matrix (curl-curl term)
- `M` is the mass matrix (conductivity term)
- `{F}` is the source term

In integral form:

```
∫_Ω ν (∇ × A⃗_real) · (∇ × v) · 2πr dr dz
    + jω ∫_Ω σ A⃗_real · v · 2πr dr dz (real part)

∫_Ω ν (∇ × A⃗_imag) · (∇ × v) · 2πr dr dz
    + jω ∫_Ω σ A⃗_imag · v · 2πr dr dz (imaginary part)
```

### Physical Interpretation

The real and imaginary parts of `A_φ` represent:
- **Real part:** Component in phase with the driving current
- **Imaginary part:** Component 90° out of phase (eddy current losses)

### Derived Quantities

**Complex magnetic flux density:**
```
B⃗ = B⃗_real + j B⃗_imag = ∇ × A⃗
```

**Magnitude (`B_Magnitude`):**
```
|B⃗| = √(|B⃗_real|² + |B⃗_imag|²)
```
With peak phasors this is √2 times the RMS of `B(t)`; for a field whose
direction rotates in time (elliptical polarization) it is not the time peak
of `|B(t)|` either.

**Time-domain fields:**
```
B⃗(t) = Re{B⃗ e^{jωt}} = B⃗_real cos(ωt) - B⃗_imag sin(ωt)
```

**Power loss density (Joule heating):**

The loss density is set by the total electric field in the conductor,
`P = ½ σ |E⃗|²`. That field is not `-jωA⃗` everywhere:

```
E⃗ = E⃗_drive - jωA⃗
```

where `E⃗_drive` is the scalar-potential or port-drive contribution. Hence

```
P = ½ σ |E⃗_drive - jωA⃗|²
```

The frequently quoted simplification

```
P = ½ σ ω² |A⃗|²
```

is the special case `E⃗_drive = 0`. It is valid in induction-only regions, but
**not** in massive or open-current regions, which are driven by a port and carry
a nonzero `E⃗_drive`. Using the simplified form there omits the drive term and
the cross term, and so misstates the loss.

**Implementation status:** implemented. `MqsLossDensityCoefficient`
(`src/mqs_loss_density_coefficient.hpp`) evaluates the general expression above
and is exported as the field `P_Loss`; `MagnetoquasistaticSolver::
ComputeRegionLosses()` integrates it per region and reports per-region and total
dissipation for field scenarios.

**Phasor convention:** peak (amplitude), matching the common convention in
commercial AC/DC and eddy-current tools. This is why the time-average factor `½`
appears explicitly above, and it is consistent with the coupling extraction
`R = Re(V/I)` and `L = Im(V/I)/ω`, which are amplitude ratios. Switching to RMS
would change every reported loss by exactly a factor of two; three independent
tests pin the convention.

**Scope:** every region with `σ > 0` is reported, not only those owning a port
unknown. The `jωσ` mass term induces eddy currents in any conductive material, so
a flux shield or a steel brace dissipates real power while appearing in no
coupling matrix. Regions are classified by the constraint they carry:

| Region kind | Port unknown | `E⃗_drive` |
|---|---|---|
| Massive terminal | yes | solved port voltage |
| `current_constraint: open` | yes (pins net current to zero) | solved voltage enforcing that constraint |
| No constraint, `σ > 0` | none | zero |

The per-attribute drive table is zero-initialised and written only where a port
exists, so unported conductors reduce to the `P = ½ σ ω² |A⃗|²` special case
through the general expression rather than through a separate code path. Note
that this makes the "frequently quoted simplification" above exactly correct for
unported conductors, and wrong only for driven ones.

Stranded terminals dissipate nothing here. A stranded conductor is a winding of
insulated strands in series: the connection fixes every strand's current and
none crosses between strands, so its current is the imposed source alone and
the `jωσA` term is omitted there, whatever the material's `σ` (which is the
wire conductivity, relevant to the winding's own resistance and in-strand
losses, neither of which is modelled).

**Verification:** the DC limit gives `P → I²/(2 G_dc)` for a single massive port,
and global power balance `Σ_regions P = ½ Re(Σ_p V_p I_p*)` holds to
discretization error (measured relative gap 3.0e-3, falling to 7.4e-4 under 2×
radial refinement — second-order convergence to exact balance). Unported and
open-current regions dissipate while contributing no net port current, so they
appear on the left of that identity and not the right.

### Boundary Conditions

Same essential/natural split as magnetostatics:
- **Dirichlet:** The configured real value constrains `A_φ`; the imaginary
  boundary value is currently zero.
- **Neumann:** The configured real outward natural flux is assembled into the
  real field RHS. A zero value remains implicit.
- **Robin:** Electrostatics only; the magnetic solvers reject it.

### 3D Form

`geometry_type: 3d` (`MagnetoquasistaticSolver3D`) solves for the complex
Nédélec potential `A` with the same conductors and boundary conditions as 3D
magnetostatics:

```
∇ × (ν ∇ × A) + jωσA = J_s + σ V w
```

- Stranded terminals are sources `J_s = I (w/|w|) / A_cs`, projected as in
  magnetostatics.
- A massive terminal is a port: in the conductor `E = V w − jωA`, and the
  voltage `V` is the unknown that makes its net current `∫σE·w dV = I`,
  i.e. `G V − jω cᵀA = I` with `c_i = ∫σ w·N_i dV`. The block system is the
  same `MqsMassivePortOperator` as in 2D.
- Every other conducting region carries `−jωσA`. `A` is the modified
  potential of the A-formulation: in a conductor its gradient part carries the
  electric scalar potential, so the eddy current is weakly divergence-free with
  no normal component at the conductor surface, with no extra unknown.

Both linear solvers solve a regularized system: in the nonconducting regions
curl-curl alone is singular. Tested with a gradient, the regularized equation is
`∫(β + jωσ) A·∇ψ = 0`, so `β` enters charge conservation in and at the surface
of every conductor, off by a relative `β/(ωσ)`. The static `β` is harmless for
good conductors but not for weak ones (it overstated the loss of a 1 S/m block
at 50 Hz by 68%), and confining it to the nonconducting regions does not help,
since the surface term remains. MQS therefore uses
`β = 10⁻⁶ min(ν_min/L², ω_min σ_min)` over all scenario frequencies and
conducting regions, which keeps `β/(ωσ) ≤ 10⁻⁶`, floored at `10⁻⁶` of the
static weight to stay above round-off (with a warning if the floor binds).

`direct` factors the packed real form once per frequency (sparse LU); `iterative` is GMRES with the
block-diagonal preconditioner `diag(P, P)`, `P ≈ (K + ωM_σ)⁻¹` by AMS, plus the
exact inverse of the port corner.

Coupling rows: a massive terminal's `Z = V/I` from its solved voltage, a
stranded terminal's `Z = jω λ` with `λ = b'·A`; `R = Re Z`, `L = Im Z / ω`.
The loss density is `½σ|V w − jωA|²`. For the discrete solution the total loss
equals the real input power `½ Re(Σ_k V_k I_k*)` (with `V_k = jωλ_k` for a
stranded terminal) by the Galerkin energy identity, but only as exactly as the
quadratures agree: `G`, the port columns `c` and the loss integral use their own
rules, which are exact on affine elements and approximate on curved ones, so on
a curved mesh the balance holds to quadrature error. Stranded regions do not
conduct in the field solve, so they add nothing to either side.

**Conductors touching an `n × A = 0` wall.** `n × A = 0` makes the tangential
`E = −jωA` vanish on the wall, so the wall behaves like a perfect electrical
contact: eddy current may flow from a conductor into it and return elsewhere
through it. That is right on a symmetry plane that current crosses normally,
but on an outer box it shorts the conductor's surface to the box. MQS warns
when a conductor touches such a wall outside its own electrodes.

## Finite Element Discretization

All three formulations use H¹-conforming (Lagrange) finite elements:

- **Basis functions:** `φ_i(r, z)` with `C⁰` continuity
- **Degrees of freedom:** Nodal values
- **Integration:** Gauss quadrature with elevated order. The `2*pi*r` weight is
  polynomial and raises the exact order by a fixed amount; the `1/r` term is not
  polynomial and its cost depends on element geometry (see M5).

### Special Considerations for Axisymmetry

1. **Axis handling:** Assembly applies no limit or clamp at `r = 0`. Interior
   quadrature keeps `r > 0`, and the `A_φ/r` limit is used only during field
   recovery on the constrained solution. See "Behavior at r = 0" above for why
   an assembly-time substitution would be invalid.
2. **Integration weight:** Include the full `2*pi*r` factor in all volume integrals
3. **Boundary conditions:** Electrostatic axis symmetry is natural. The magnetic
   `A_phi` formulation does **not** enforce `A_phi = 0` automatically; the
   condition is imposed as an essential BC on the detected axis boundary, and
   doing so is what makes the `1/r` term well posed.

### Integration Measure Convention

Revolving a meridional `(r, z)` domain through the full azimuthal angle gives the
volume element `2*pi*r dr dz` and the boundary element `2*pi*r ds`. **Every
axisymmetric integrator applies the full measure**, obtained from the single
definition `Axisymmetric::Measure(r)` in `src/axisymmetric_measure.hpp`:

| Term | File |
|---|---|
| ES stiffness | `axisymmetric_diffusion_integrator.hpp` |
| MS/MQS curl-curl | `axisymmetric_curl_curl_integrator.hpp` |
| MQS sigma mass | `axisymmetric_mass_integrator.hpp` |
| Domain source | `axisymmetric_lf_integrator.hpp` |
| Neumann load | `axisymmetric_boundary_lf_integrator.hpp` |

Nothing is factored out and restored later. Consequently every assembled matrix,
every right-hand side, and every derived quantity is in SI units and is directly
comparable to a hand calculation: a gathered terminal charge is coulombs, `G_dc`
is siemens, a flux linkage is webers. Adding a new derived quantity requires no
knowledge of any normalization convention.

Two `2*pi` factors in the code are **not** part of this measure and must not be
routed through the helper:

- `AxisymmetricConductanceCoeff`'s `1/(2*pi*r)`, which is physical - it comes
  from `E_phi = V/(2*pi*r)` for an azimuthal conductor.
- `omega = 2*pi*f`, a temporal frequency conversion.

The historical alternative - omitting the global `2*pi` from every integrator and
reintroducing it in each derived quantity - is mathematically equivalent but was
abandoned: it required six separate scale factors at call sites (some multiplying
by `2*pi`, some dividing), left intermediate quantities in non-physical units,
and was the direct cause of findings 1, 2, 3 and 5 in
`docs/math_review_findings.md`.

## Conventions and Recurring Pitfalls

Points that the math review found easy to get wrong, recorded so they are not
rediscovered. Each is enforced by a regression test.

**Keep the documented form and the assembled form identical.** Several findings
(D1-D4) were cases where the code was right and this document was wrong. A
formulation document that drifts from the implementation is worse than none: it
invites "fixing" correct code to match an incorrect description. When either
changes, change both.

**Do not describe unimplemented behavior in the present tense.** The `ρ` charge
source (D1) is documented above but not assembled, and carries an explicit
implementation-status note. Joule loss (D4) was in the same state and is now
implemented, so its note records what exists rather than what might. Documenting
a general form is useful; implying it runs is not.

**Report a physical quantity wherever it is physical, not only where the model
names it.** Joule loss was initially scoped to terminals and open-current
regions, which are the entities appearing in coupling matrices. That would have
omitted flux shields and structural steel: they are conductive, the `jωσ` term
already induces eddy currents in them, and they dissipate real power while owning
no port unknown. The correct scope followed from what the operator actually
assembles, not from how the model labels regions.

**Give each mesh attribute exactly one reporting owner.** A terminal and a region
routinely share an entity group, since a massive port's conductor is usually also
declared as a material region. Aggregating per-region results by both names
independently double-counts the shared attribute. The per-region values were
individually correct while the total was exactly twice the truth, which is the
kind of error that survives a casual review of the output.

**Distinguish the solution from the basis functions.** This is the single most
recurrent trap in the axisymmetric formulation. `A_φ/r` has a finite limit for
the *constrained solution* but not for an *individual shape function*. Anything
that holds only after essential-BC elimination belongs in post-processing, never
in assembly. D3 and finding M4 were both this confusion.

**Non-polynomial terms are not covered by a polynomial order rule.** The `1/r`
factor's quadrature cost is governed by the element's geometric ratio
`r_min/width`, not by basis degree. A degree-only heuristic is structurally
blind to the hard case rather than merely loosely tuned (M5).

**Carry the full physical measure everywhere.** Factoring a constant out of
assembly and restoring it downstream is mathematically equivalent but was the
direct cause of four separate findings. Intermediate quantities should be
readable in SI units at every stage.

**State the extrusion convention in any reported value.** Planar results are per
unit length and axisymmetric results are absolute; identical numbers mean
different things (M6). Units belong in the label, not in the reader's memory.

**A quantity that cannot be computed accurately should say so.** Where the
capped `1/r` quadrature rule cannot reach its accuracy target, the solver warns
at setup with the offending element rather than silently returning a degraded
result.
