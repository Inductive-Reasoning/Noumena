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

## Analytical Solution

On the axis of a finite coil of uniform current density J = NI/((r_out − r_in) L),
centred on z = 0, the free-space field is

```
B_z(z) = (μ₀ J / 2) [f(z + L/2) − f(z − L/2)],
f(u) = u ln[(r_out + √(r_out² + u²)) / (r_in + √(r_in² + u²))]
```

which gives

```
B_z(0)   = 5.27 mT   (centre)
B_z(L/2) = 2.99 mT   (end of the coil, 57% of the centre value)
```

The long-solenoid value μ₀NI/L = 6.28 mT does not apply: this coil is only
1.25 times as long as its outer diameter, and the long-solenoid formula is
16% above its centre field.

**Current density in coil:**

The terminal is `stranded`, a winding of many turns that carries a uniform
current density (a `massive` conductor would carry its DC conduction
distribution instead, which around the axis falls off as 1/r). The terminal
has `"turns": 1000` and its excitation is the current in each turn, I = 1 A;
the solver divides the ampere-turns N I by the coil cross-section to obtain
the source current density:

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
2. **Magnetic flux density B:** about 5.2 mT axial field at the centre
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

Sample B on the axis with a probe (see `output.probes` in
[the configuration reference](../../docs/config_reference.md)), for example
at (r, z) = (0, 0) and (0, 0.1). The shipped mesh gives

| Point | Computed | Analytical | Difference |
|---|---|---|---|
| Centre, z = 0 | 5.232 mT | 5.266 mT | -0.65% |
| End, z = L/2 | 2.954 mT | 2.986 mT | -1.1% |

Both are a little low because the far boundary (A_φ = 0 at r = 0.4 m,
|z| = 0.5 m) truncates the return flux; see
[Open-boundary truncation](../../docs/open_boundary.md).

## With a Magnetic Core

The shipped mesh has no core region; to try one, add a separate surface for
the bore in `solenoid.geo` and give it a material with `mu_r > 1`. Do not
expect the field to rise by μᵣ: a core no longer than this coil is strongly
demagnetized, so the gain is set mostly by its shape (and, for real iron, by
saturation, which this linear solver does not model).

## Variations

Try:
1. **Current:** Vary I (0.1 A to 10 A) - field scales linearly
2. **Turns:** Change N - field scales with N
3. **Core material:** Add a core region and compare μᵣ = 1 with μᵣ > 1
4. **Geometry:** Vary length-to-diameter ratio
5. **Mesh refinement:** Increase order or mesh density

## Advanced: Inductance Calculation

Set `"analysis_type": "coupling_matrix"` to get the inductance from the
stored magnetic energy, `L = 2U / I²`. With the terminal's `turns` set, I is
the current in each turn and the reported value is the inductance of the
whole winding. (Without `turns`, the excitation is the ampere-turns and the
reported value is per ampere-turn squared, 1/N² of the winding's.)
