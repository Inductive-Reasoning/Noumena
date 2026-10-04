# Example: Axisymmetric Solenoid

This example calculates the magnetostatic field of a current-carrying solenoid coil.

## Problem Description

**Geometry:** (all dimensions in metres -- the solver is SI and assumes a mesh
in metres; see [Units](../../docs/config_reference.md#units))
- Solenoid: inner radius r_in = 0.05 m, outer radius r_out = 0.08 m
- Length: L = 0.2 m
- Number of turns: N = 1000
- Current per turn: I = 1 A
- Axisymmetric about central axis

**Boundary Conditions:**
- Far field: A_φ = 0 (Dirichlet)
- Axis (r = 0): A_φ = 0 regularity condition, imposed automatically

**Materials:**
- Coil region: Current density J_φ = N × I / A_coil
- Air core: μᵣ = 1.0
- Surrounding air: μᵣ = 1.0

**Optional:** Add iron core (μᵣ = 1000) to demonstrate field concentration

## Analytical Solution

For a finite, uniformly filled rectangular winding, the free-space center
field is:

**Magnetic field inside solenoid:**
```
B_center = mu0 N I / (2 (r_out - r_in))
           * (asinh(2 r_out/L) - asinh(2 r_in/L))
         = 5.27 mT
```

The infinite-solenoid estimate `mu0 N I/L = 6.28 mT` is not a 2% reference
for this aspect ratio.

**Current density in coil:**

The input file specifies the terminal excitation as a **total current** in
amperes (here the full ampere-turns, N I = 1000 A); the solver divides by the
coil cross-section to obtain the source current density:

```
J = N I / A_coil
```

where `A_coil = (r_out - r_in) × L = 0.03 × 0.2 = 0.006 m²`

```
J = 1000 × 1 / 0.006 = 1.67 × 10⁵ A/m²
```

## Running the Example

```bash
# From project root
cmake -S . -B build
cmake --build build --config Release

# Run simulation
./build/mfem-electromag examples/solenoid/config.json
```

This config does **not** enable file output. Add a top-level `output` block:

```json
{
  "output": {
    "paraview": {},
    "gmsh": {},
    "hdf5": {}
  }
}
```

The scenario `energized` produces a collection under
`results/paraview/scenario_000000_energized/`, a Gmsh file at
`results/gmsh/scenario_000000_energized.msh`, and a `results/results.h5` archive.
These paths are relative to the config directory unless `output.directory` is set.

## Expected Results

The simulation should produce:
1. **Vector potential A_φ:** Increases inside coil, drops to zero at far field
2. **Magnetic flux density B:** Approximately 5.27 mT at the center in free space
3. **Field lines:** Closed loops through coil and return path in air
4. **Fringing:** Field spreads near ends of solenoid

## Visualization

```bash
# Open in ParaView
paraview results/paraview/scenario_000000_energized/scenario_000000_energized.pvd
```

**Suggested visualizations:**
- Vector field of B (magnetic flux density)
- Streamlines showing field lines
- Contour plot of |B| magnitude
- Line plot along axis to show field uniformity

## Mesh Generation

The mesh should include:
- Coil region (source of current)
- Air core inside coil
- Air region outside coil
- Far-field boundary (large enough to avoid boundary effects)

Recommended mesh:
- Element size in coil: ~5 mm
- Element size in air: graded from 5 mm to 50 mm
- Far-field radius: at least 5 × coil radius

## Validation

Compare a mesh and boundary-converged center field to the finite-winding
formula above. A 2% comparison is a suggested study target, not a reproduced
result for this mesh:

**Inside solenoid (center):**
```
|B_z,FEM - B_analytical| / B_analytical < 2%
```

**End effects:** The field decreases near the ends; the half-center estimate
is only the long, thin solenoid limit.

## With Iron Core

Add iron core (μᵣ = 1000) inside solenoid:

```json
{
  "entity_groups": [
    {"name": "Core", "dim": 2, "attribute_ids": [3]}
  ],
  "materials": [
    {"name": "Iron", "properties": {"mu_r": 1000.0}}
  ],
  "regions": [
    {"name": "IronCore", "entity_group": "Core", "material": "Iron"}
  ]
}
```

Merge these entries into the corresponding arrays only after meshing a distinct
core domain with attribute 3; do not assign an existing air domain twice.

**Expected changes:**
- Demagnetizing fields limit the increase; it is not generally a factor of mu_r.
  Saturation is not represented by the present constant-permeability model.
- Field concentration along axis
- Reduced fringing outside coil

## Variations

Try:
1. **Current:** Vary I (0.1 A to 10 A) - field scales linearly
2. **Turns:** Change N - field scales with N
3. **Core material:** Compare air (μᵣ=1) vs iron (μᵣ=1000)
4. **Geometry:** Vary length-to-diameter ratio
5. **Mesh refinement:** Increase order or mesh density

## Advanced: Inductance Calculation

Calculate inductance from stored magnetic energy:

```
L = 2 U / I²
```

where `U = ∫ (B²/2μ) dV` is the magnetic energy.

Extract from simulation:
```
U = ∫ (B·H/2) dV
L ≈ N Φ / I
```

Use the physical turn current I in the energy formula, not the configured
ampere-turn excitation N I. For an ampere-turn-normalized coupling matrix,
physical winding coefficients are `L_winding(i,j) = N_i N_j L_solver(i,j)`.
Do not apply this conversion a second time if turn factors are already part
of the terminal source definition.
