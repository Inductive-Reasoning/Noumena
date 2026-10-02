# MFEM-ElectroMag Examples

This directory contains example problems demonstrating electromagnetic field simulations
using MFEM-ElectroMag. All use second order (`"order": 2`) elements; all but `team7/` use
the axisymmetric (r-z) formulation.

**Units: SI, with mesh coordinates in metres.** Every `.geo` here is written in metres,
and every config value is SI (`sigma` in S/m, `frequency` in Hz, excitations in V or A).
The solver has no unit or scale key, and nothing validates the mesh extent, so a geometry
authored in millimetres will solve cleanly and report absolute quantities that are wrong
by powers of 1000. See [Units](../docs/config_reference.md#units).

## Examples

| Directory | Physics | Mesh | Notes |
|-----------|---------|------|-------|
| `simple_capacitor/` | Electrostatics | `capacitor_from_sl.mesh` | Parallel plate capacitor, fringing fields, capacitance |
| `solenoid/` | Magnetostatics | `solenoid.mesh` | Axisymmetric solenoid, inductance |
| `current_loop/` | Magnetostatics + MQS | `loop.mesh` | Single current loop with an analytical hand calculation notebook, plus a low-frequency MQS cross-check |
| `eddy_current/` | Magnetoquasistatics | `eddy_current.mesh` | Conducting cylinder in an AC field, skin effect, losses |
| `team7/` | Magnetoquasistatics (3D) | `team7.msh` | TEAM benchmark 7: plate with a hole under a racetrack coil, compared with measurements; needs the MPI build |

Each directory contains a `config.json` and its `.geo` source geometry. The generated
`.mesh` files are committed, so the examples run without installing Gmsh.
`simple_capacitor/`, `solenoid/`, `eddy_current/` and `team7/` also include a `README.md` with the
detailed problem description; `current_loop/` includes `hand_calc.ipynb` instead.

`current_loop/` additionally ships `config_mqs_lowfreq.json`, which solves the same mesh
with the magnetoquasistatic solver at 0.1 Hz. Because `physics_type` is a per-file
setting, the cross-check has to be a separate config rather than an extra scenario. See
[Cross-checking magnetostatics against MQS](#cross-checking-magnetostatics-against-mqs).

## Quick Start

### 1. Build the solver

```bash
cmake -S . -B build
cmake --build build --config Release
```

On Windows the repository also ships CMake presets (`x64-debug`, `x64-release`) that
Visual Studio picks up automatically:

```pwsh
cmake --preset x64-release
cmake --build --preset x64-release
```

MFEM is fetched and built by CMake, so the first configure takes a while.

### 2. Run an example

The config path is the only required argument. Relative paths inside the config
(mesh file, `output.directory`) are resolved against the config file's directory, so the
examples can be run from anywhere:

```bash
./build/mfem-electromag examples/simple_capacitor/config.json
./build/mfem-electromag examples/solenoid/config.json
./build/mfem-electromag examples/current_loop/config.json
./build/mfem-electromag examples/eddy_current/config.json
```

Useful options:

| Option | Purpose |
|--------|---------|
| `--output-directory <dir>` | Override `output.directory` (relative paths resolve against the CWD) |
| `--verbosity <0\|1\|2>` | `0` status/timing, `1` solver output, `2` diagnostics |
| `--machine-readable` | Emit JSON Lines progress on stdout for tooling |
| `--version` | Print version/build information |
| `-h`, `--help` | Show usage |

### 3. Output

Output is off by default and enabled per format in a top-level `output` block:

```json
{
  "output": {
  "directory": "results",
  "export_fields_for_coupling_matrix": false,
  "paraview": {},
  "gmsh": {},
  "hdf5": {"file": "results.h5"}
  }
}
```

- `paraview` writes collections under `results/paraview/scenario_<id>_<label>/`.
- `gmsh` writes `results/gmsh/scenario_<id>_<label>.msh` (Gmsh MSH 2.2 ASCII). The mesh is
  emitted as native high-order Lagrange elements at the solution order, and the file
  includes an `$InterpolationScheme` block so a consumer can reconstruct fields at
  arbitrary points using the same basis MFEM used.
- `hdf5` writes one archive with the mesh, scenario fields, and coupling matrices.
- Format destinations resolve beneath `output.directory`, which defaults to
  `results` beside the config file. Absolute destinations are preserved.
- Coupling fields require `export_fields_for_coupling_matrix: true`; otherwise coupling analyses
  write only the HDF5 mesh/matrices and console tables.

`simple_capacitor` enables ParaView; `two_loops` enables Gmsh, HDF5, and coupling
fields. Other configs require adding format objects to write files.

Visualize with:

```bash
paraview examples/simple_capacitor/results/paraview/scenario_*/*.pvd
gmsh results/gmsh/scenario_000000_energized.msh
```

## Cross-checking magnetostatics against MQS

As the frequency goes to zero the magnetoquasistatic curl-curl system loses its
`j*omega*sigma` term and reduces to the magnetostatic one, so the two solvers should
agree on the same mesh. `current_loop/` exercises this:

```bash
./build/mfem-electromag examples/current_loop/config.json              # magnetostatic
./build/mfem-electromag examples/current_loop/config_mqs_lowfreq.json  # MQS at 0.1 Hz
```

The two configs are identical apart from `physics_type`, the added `frequency`, and the
conductivity the MQS run needs. Add `"output": {"gmsh": {}}` on both and compare the
magnetostatic `A` against the MQS `A_Real` node-for-node. With the shipped mesh:

| Quantity | Result |
|----------|--------|
| `max abs(A)` (magnetostatic) | 1.007e-06 |
| `max abs(A - A_Real)` | 5.9e-10 (5.9e-04 relative) |
| `max abs(A_Imag)` | 4.0e-13 (4.0e-07 relative) |

The real parts agree to ~0.06% and the quadrature component is seven orders of magnitude
below the field, which is the expected low-frequency limit.

> **Make the check meaningful.** The loop must be a *conducting massive* terminal
> (`"sigma": 5.8e7`, `"conductor_type": "massive"`). With `sigma: 0` everywhere the MQS
> system has no imaginary part at all, so it reproduces the magnetostatic answer to
> round-off at **any** frequency — the comparison passes perfectly while testing nothing.
> Verify the run reports `massive port assembly (1 ports)` and a nonzero Joule loss.

## Validating inductance against the hand calculation

The cross-check above only proves the two solvers agree with *each other*. To pin the
absolute scale, `current_loop/hand_calc.ipynb` derives a closed-form self-inductance and
the solver is compared against it.

The notebook's `A_phi(r, z)` is a *filamentary* loop expression, so it diverges as
`r -> a` and cannot yield a self-inductance on its own — that is why the sampling cell
uses `r = 0.0999` rather than `0.1`. For a conductor of finite cross-section the
singularity is removed by substituting the geometric mean distance of the section. For a
square of side `w`, `r_eq = 0.2235 * (w + h)`, which already carries the internal
inductance, so Grover's thin-ring formula applies in its `-2` form:

```
L = mu_0 * a * (ln(8a / r_eq) - 2)
```

For the shipped geometry (`a = 0.1 m`, `w = 0.002 m`) this gives `r_eq = 8.94e-4 m` and
`L = 6.028e-07 H`. Neither example config computes an inductance by default — both run
`analysis_type: "field"`. To get the matrix, set `"analysis_type": "coupling_matrix"`
and add `"output": {"hdf5": {"file": "inductance.h5"}}`. Use separate output roots
or filenames for the two runs to retain both archives.
Read `/coupling/Inductance/values`; MQS indexes its matrix series by the shared
`/coupling/frequency_hz` array. See [the HDF5 schema](../docs/coupling_hdf5.md).

| Solve | L [H] | Error vs. analytic |
|-------|-------|--------------------|
| Analytic (GMD ring) | 6.0277e-07 | — |
| Magnetostatic | 6.0236e-07 | -0.07% |
| MQS @ 0.1 Hz | 6.0236e-07 | -0.07% |

The residual is **almost entirely far-field truncation**, not discretization and not the
GMD approximation. Imposing `A_phi = 0` at a finite radius removes field energy that
physically extends past the boundary, which biases the inductance low. Sweeping the
domain size shows a clean `1/D` decay:

| Domain `D` | `D/a` | Error |
|---|---|---|
| 0.5 m | 5 | -0.49% |
| 1.0 m | 10 | -0.20% |
| 4.0 m | 40 | -0.06% |
| 16.0 m | 160 | -0.03% |

Richardson-extrapolating to `D -> infinity` gives `6.0265e-07 H`, within -0.013% of the
closed form — so the solver reproduces Grover's formula to about a part in 10^4 once the
boundary is moved out. The test meshes therefore use `D/a = 40` and assert 0.5%. See
[Open-Boundary Truncation](../docs/open_boundary.md) for the full study and the options
for doing better than Dirichlet-at-a-distance.

This comparison is enforced as a regression in `test/test_solvers.cpp` by
`Magnetostatic loop inductance matches the analytic ring value` and
`Magnetoquasistatic loop inductance matches the analytic ring value at low frequency`.
Both are kept because they exercise genuinely different assembly paths: the magnetostatic
case is a real stiffness solve with a prescribed current density, while the MQS case is a
complex block system with a massive-port constraint. A scale error in only one of them —
a missing `2*pi` from the axisymmetric measure, say — passes the existing reciprocity
tests, which constrain only symmetry and sign.


Raising the frequency is a useful control: at 1 kHz the same comparison gives
`max abs(A_Imag)` of 4.0e-09 (4.0e-03 relative), four orders of magnitude larger,
confirming the term being neglected is genuinely frequency-dependent.

## Regenerating Meshes

The committed `.mesh` files are sufficient to run the examples. Regenerate only if you
change a `.geo` file.

`generate_meshes.sh` (bash; requires Gmsh on `PATH`) regenerates the eddy current,
capacitor, and solenoid meshes:

```bash
cd examples
./generate_meshes.sh
```

It runs `gmsh -2 -format msh2` on each `.geo`, then uses MFEM's `convert-mesh` utility if
available, otherwise simply renames the `.msh` to `.mesh` (MFEM reads Gmsh MSH2 directly).

Note: the script writes `simple_capacitor/capacitor.mesh`, but that example's config uses
`capacitor_from_sl.mesh`. `current_loop/loop.mesh` is not covered by the script.

To regenerate a single mesh manually:

```bash
cd examples/solenoid
gmsh -2 -format msh2 solenoid.geo -o solenoid.mesh
```

Use MSH2 format; MFEM does not read all MSH4 element types.

### Mesh refinement

Each `.geo` sets mesh size fields for appropriate refinement:

- **Eddy current**: fine mesh near the conductor surface to resolve skin depth
- **Capacitor**: refined near plate edges to capture fringing fields
- **Solenoid**: uniform in the coil region, graded toward the far field

Adjust the `lc_*` parameters in the `.geo` files to change mesh density: increase them for
a fast coarse test, decrease them for higher accuracy.

## Modifying Examples

### Geometry

Edit the `.geo` parameters, then regenerate the mesh:

```c
// In eddy_current.geo
cyl_radius = 0.05;      // Change cylinder size
coil_r_inner = 0.06;    // Change coil position
```

### Materials / physics

Edit `config.json`:

```json
{
  "materials": [
	{
	  "name": "Conductor",
	  "properties": { "sigma": 5.8e7 }
	}
  ]
}
```

Material attribute names must match the physical groups in the mesh. The solver validates
the config up front and reports unknown or mismatched fields by JSON path.

## Troubleshooting

**`gmsh: command not found`** — install Gmsh (`sudo apt-get install gmsh`,
`brew install gmsh`, or download from https://gmsh.info/).

**"Unknown element type"** — regenerate with `-format msh2`.

**No output files**: add format objects under the top-level `output` block.
For coupling visualization also set `output.export_fields_for_coupling_matrix` to `true`.

**Results look wrong** — check mesh quality in Gmsh, verify material attributes match the
mesh physical groups, confirm the boundary conditions, and refine the mesh if the solution
looks under-resolved.

## Physics Summary

| Example | Type | Key physics | Typical runtime |
|---------|------|-------------|-----------------|
| simple_capacitor | Electrostatics | Fringing fields, capacitance | < 1 min |
| solenoid | Magnetostatics | Magnetic field, inductance | < 1 min |
| current_loop | Magnetostatics | Loop field vs. analytical result | < 1 min |
| eddy_current | Magnetoquasistatics | Skin effect, eddy losses | 2-5 min |

## Further Information

See the individual example README files for detailed problem descriptions, analytical
solutions for validation, expected results, and parameter studies.
