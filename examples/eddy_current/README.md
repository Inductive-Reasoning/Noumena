# Example: Eddy Current in Conducting Cylinder

This example demonstrates time-harmonic eddy current analysis for a conducting cylinder in an alternating magnetic field.

## Problem Description

**Geometry:** (all dimensions in meters -- the solver is SI and assumes a mesh
in meters; see [Units](../../docs/config_reference.md#units))
- Conducting cylinder: radius R = 0.05 m, length L = 0.1 m
- Surrounding coil: creates time-varying magnetic field
- Axisymmetric configuration

**Physics:**
- Two scenarios: `PowerFrequency` at f = 60 Hz, and `FrequencySweep` from
  10 Hz to 1 kHz (five logarithmically spaced points)
- The 60 Hz analytical calculations below correspond directly to the
  `PowerFrequency` scenario
- Angular frequency at 60 Hz: ω = 2πf = 377 rad/s
- Conductor: Aluminum (σ = 3.5 × 10⁷ S/m, μᵣ = 1.0)
- Coil: stranded winding driven with a total current of 240 A, which over the
  0.0024 m² coil cross-section gives J_source = 1 × 10⁵ A/m²

**Boundary Conditions:**
- Far field: A_φ = 0 (Dirichlet)

**Skin Depth:**
```
δ = √(2 / (ω μ σ))
  = √(2 / (377 × 4π×10⁻⁷ × 3.5×10⁷))
  = 10.98 mm
```

At 60 Hz the skin depth is about a fifth of the cylinder's 50 mm radius, so
the skin effect is moderate: the field reaches well into the aluminum.

## Physical Phenomena

### 1. Skin Effect
- Induced currents flow primarily near surface
- Exponential decay with depth: `e^(-x/δ)`
- Current density maximum at surface

### 2. Eddy Current Losses
- Power dissipation: `P = ∫ (σ/2) |E|² dV`
- Joule heating in conductor
- Phase lag between applied field and induced current

### 3. Shielding Effect
- Eddy currents generate opposing magnetic field
- Reduced field penetration into conductor
- Effective permeability appears < μ_r at high frequency

## Analytical Approximation

For a conductor much thicker than the skin depth (the half-space limit,
δ ≪ R; at 60 Hz this cylinder is only roughly in it):

**Surface resistance:**
```
R_s = √(ω μ / (2 σ)) = 1 / (σ δ)
    = √(377 × 4π×10⁻⁷ / (2 × 3.5×10⁷))
    = 2.60 × 10⁻⁶ Ω/square
```

**Power loss per unit area**, with `H_t` the peak tangential surface field:
```
P/A = ½ R_s |H_t|²
```

## Running the Example

```bash
# From project root
cmake -S . -B build
cmake --build build --config Release

# Run simulation
./build/noumena examples/eddy_current/config.json
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

This config defines two scenarios. `PowerFrequency` uses a scalar frequency.
`FrequencySweep` expands to one solve *per frequency point*, with names
`<scenario>_f<n>_<frequency>Hz`, with `.` replaced by `p`:

| Scenario | Frequency | Scenario label |
|----------|-----------|-------------|
| `PowerFrequency` | 60 Hz | `PowerFrequency` |
| `FrequencySweep` | 10 Hz | `FrequencySweep_f1_10Hz` |
| `FrequencySweep` | 31.6 Hz | `FrequencySweep_f2_31p6227766017Hz` |
| `FrequencySweep` | 100 Hz | `FrequencySweep_f3_100Hz` |
| `FrequencySweep` | 316 Hz | `FrequencySweep_f4_316p227766017Hz` |
| `FrequencySweep` | 1000 Hz | `FrequencySweep_f5_1000Hz` |

Visualization names prefix these labels with a stable index, such as
`scenario_000000_PowerFrequency`. ParaView collections and Gmsh files go under
`results/paraview/` and `results/gmsh/`, relative to the config directory.
HDF5 stores every frequency's fields and metadata in one `results/results.h5`
archive. Set `output.directory` to change the common root.

## Expected Results

The simulation produces complex-valued fields:

1. **A_real, A_imag:** In-phase and quadrature components of vector potential
2. **B_real, B_imag:** Real and imaginary parts of magnetic flux density
3. **B_Magnitude:** √(|B_real|² + |B_imag|²), a peak-phasor magnitude:
   √2 times the RMS value for a field of fixed direction
4. **Power loss:** Concentrated near conductor surface

## Visualization

```bash
# Open the 60 Hz scenario in ParaView
paraview results/paraview/scenario_000000_PowerFrequency/scenario_000000_PowerFrequency.pvd
```

**Key visualizations:**
- B_Magnitude: Shows field concentration and shielding
- A_Real vs A_Imag: Phase relationship
- Line plot through conductor: Demonstrates skin effect
- Time animation: Use Temporal Interpolator filter

### Time-Domain Animation

To animate the time-harmonic solution:

1. In ParaView, add "Calculator" filter:
   ```
   B_time = B_Real*cos(2*pi*60*t) - B_Imag*sin(2*pi*60*t)
   ```

2. Add "Programmable Source" to generate time steps

3. Animate to see oscillating field

## Mesh Considerations

**Critical:** Mesh must resolve skin depth!

- **In conductor:** element size about δ/3 near the surface at the highest
  frequency solved. The shipped mesh uses 0.9 mm, a third of the 2.7 mm skin
  depth at 1 kHz (see `eddy_current.geo`).
- **Boundary layer:** use a mesh graded from the surface inward
- **Air region:** coarser mesh acceptable (15-40 mm in the shipped mesh)

## Validation

The analytical figures above are for 60 Hz, so validate against the
`PowerFrequency` scenario rather than a sweep point (the sweep brackets 60 Hz
between its 31.6 Hz and 100 Hz points but does not land on it).

### 1. Skin Depth Check
Plot |B| vs depth into conductor. Where δ is small against the radius (the
upper sweep points; δ = 2.7 mm at 1 kHz), the decay follows the half-space
law

```
|B(x)| / |B(0)| ≈ e^(-x/δ)
```

and at depth x = δ the field drops to ~37% (1/e) of its surface value. At
60 Hz, with δ ≈ R/5, the cylinder's curvature changes the profile.

### 2. Power Loss
Compare computed losses to analytical for simple geometry. The solver reports
per-region time-averaged Joule loss at the end of each scenario; for
`PowerFrequency` the conductor loss is ~1.03e-01 W with the shipped mesh and
the peak-phasor excitation convention.

### 3. Phase Relationship
Inside conductor:
- In the half-space limit the surface current density lags the surface
  field by 45°; in a finite cylinder with δ comparable to its size the angle
  differs and varies over the surface
- Phase increases with depth

## Frequency Sweep

Each MQS scenario requires either a positive scalar frequency or an inclusive
linear/logarithmic range. This example uses both forms. A scalar solves a
single point and keeps the scenario name unchanged:

```json
"frequency": 60.0
```

A range expands to one solve per point:

```json
"frequency": {
  "scale": "log",
  "start": 10.0,
  "stop": 1000.0,
  "points": 5
}
```

Use `"scale": "linear"` for uniform spacing. Both endpoints are included;
`"points": 1` solves only `start`. The excitation list is copied to every
expanded frequency point, and output scenario names include the point and
frequency.

For a coupling-matrix sweep, set `"analysis_type": "coupling_matrix"` and
provide one or more frequency scenarios. Terminal excitations in those
scenarios are ignored because the solver synthesizes each unit-current column;
each unique frequency contributes a resistance and inductance matrix to the
archive enabled by `output.hdf5`. Coupling fields are omitted unless
`output.export_fields_for_coupling_matrix` is true. See [the HDF5 schema](../../docs/coupling_hdf5.md)
for the shared frequency axis and quantity groups.

**Expected trends** (for this fixed source current):
- Higher f → smaller δ (stronger skin effect)
- Higher f → greater power loss (growing like f² while δ is large against
  the conductor, like √f once it is small)
- Higher f → better shielding

**At 1 kHz:**
```
δ = 10.98 mm / √(1000/60) = 2.69 mm
```

## Material Variations

Compare different conductors:

| Material | σ (S/m) | μᵣ | δ @ 60 Hz | Application |
|----------|---------|----|-----------|-------------|
| Aluminum | 3.5×10⁷ | 1 | 11.0 mm | Lightweight |
| Copper   | 5.8×10⁷ | 1 | 8.5 mm | High conductivity |
| Steel    | 1.0×10⁶ | 100 | 6.5 mm | Structural, magnetic (δ ∝ 1/√μᵣ; 65 mm at μᵣ = 1) |
| Carbon   | 1.0×10⁴ | 1 | 650 mm | Composite materials |

## Applications

This example models:
- **Induction heating:** Controlled power deposition in conductors
- **Eddy current testing (NDT):** Defect detection via field distortion
- **Transformer cores:** Laminations reduce eddy current losses
- **Electromagnetic braking:** Drag force from induced currents
- **Shielding:** Protection from EMI

## Advanced Analysis

### 1. Loss Calculation

Integrate power loss density:

For a passive conductor, where E = −jωA:

```
P_total = ∫_conductor (σ ω² / 2) |A|² dV
```

Extract from simulation output.

### 2. Force Calculation

Time-averaged Lorentz force on the conductor, from the peak phasors:

```
⟨F⟩ = ½ Re ∫ J × B* dV
```

Its sign depends on the phase between the induced current and the field.

### 3. Impedance

For coil + conductor system:

```
Z = R + jωL
```

where R includes eddy current losses, L is effective inductance.
