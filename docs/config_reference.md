# Configuration File Reference

Exhaustive listing of every key the solver reads, its type, whether it is
required, and its default. This document is the normative reference; for the
model behind these keys and the reasoning that shaped them, see
[input_file_format.md](input_file_format.md).

Defaults shown are the literal values applied in code. A key listed as
**required** is enforced by `ConfigValidator`; a key with a default may be
omitted entirely.

**Units are SI throughout, and mesh coordinates must be in metres.** Every
numeric value in this document is SI: lengths in m, conductivity in S/m,
frequency in Hz, voltage in V, current in A. There is no unit or scale key in
the schema. See [Units](#units).

---

## Contents

- [Units](#units)
- [`simulation`](#simulation)
- [`simulation.amr`](#simulationamr)
- [`entity_groups`](#entity_groups)
- [`materials`](#materials)
- [`regions`](#regions)
- [`boundary_conditions`](#boundary_conditions)
- [`terminals`](#terminals)
- [`scenarios`](#scenarios)
- [Validation rules](#validation-rules)
- [Superseded names](#superseded-names)

---

## Units

SI, with no exceptions and no configurable scale factor.

| Quantity | Unit |
|----------|------|
| Mesh coordinates | metres (m) |
| `properties.sigma` | S/m |
| `properties.epsilon_r`, `properties.mu_r` | dimensionless (relative) |
| `frequency` | Hz |
| Voltage excitation | V |
| Current excitation | A |
| Charge density | C/m^3 |

The mesh unit is the sharp edge. `MU_0` and `EPSILON_0` in
`src/core/constants.hpp` are per-metre, and the axisymmetric measure `2*pi*r`
uses the radial coordinate as a physical length, so a mesh authored in
millimetres solves without complaint and returns absolute quantities --
energy, capacitance, inductance, loss -- that are wrong by powers of 1000.

> The convention is **documented but not enforced.** Nothing inspects the mesh
> extent, because a plausible bounding box cannot distinguish a small model in
> metres from a large one in millimetres. Convert at mesh generation time.

Field values written to output files carry the same SI units as the
formulation that produced them (V, V/m, T, Wb/m, W/m^3).

---

## `simulation`

Object. **Required** -- the only required section.

| Key | Type | Default | Range / values |
|-----|------|---------|----------------|
| `physics_type` | string | **required** | `electrostatics`, `magnetostatics`, `magnetoquasistatics` |
| `mesh` | string | **required** | Path; relative resolves against the config file's directory. Coordinates must be in **metres** |
| `geometry_type` | string | `planar` | `planar`, `axisymmetric`, `3d` |
| `analysis_type` | string | `field` | `field`, `coupling_matrix` |
| `order` | integer | `1` | 1-10 |
| `linear_solver` | string | `direct` (2D), `iterative` (`3d`) | `direct`, `iterative` |
| `solver_tolerance` | number | `1e-12` | (0, 1); relative residual of the iterative solve |
| `solver_max_iter` | integer | `1000` | >= 1 |
| `solver_print_level` | integer | `1` | -- |
| `amr` | object | absent = disabled | See below |

`geometry_type` must match the mesh: `planar` and `axisymmetric` require a 2D
mesh, `3d` a 3D mesh. A mismatch is rejected, including the `planar` default on
a 3D mesh. `planar` coupling quantities are per unit length (F/m, H/m); the
`axisymmetric` and `3d` ones are absolute (F, H, Ohm). `3d` supports every
physics type and every output format. 3D magnetics (magnetostatics and MQS)
supports only homogeneous boundary conditions (`dirichlet` 0 for
`n × A = 0`, `neumann` 0 or no entry for `n × H = 0`) and has no AMR. In 3D
MQS an `n × A = 0` wall is also a perfect electrical contact: a conductor
touching it can pass eddy current into the wall, so keep conductors off such
walls unless the wall is a symmetry plane (a warning is printed when one
touches). The 3D magnetic `iterative` solvers (preconditioned by hypre's AMS) need the MPI/HYPRE
build; a serial build defaults 3D magnetics to `direct`, which is practical
only for small meshes. Gmsh output supports
tetrahedra (orders 1-10) and hexahedra (orders 1-9) in 3D.

`direct` is the default linear solver for the 2D models because a
coupling-matrix run amortizes one factorization over every terminal's
right-hand side, and its accuracy does not depend on a residual tolerance.

For `3d` the default is `iterative`. The direct solver's fill-in grows far
faster in 3D (a P2 Laplacian took 346 s and 1.5 GB at 118k unknowns), and a
warning is printed before a 3D direct factorization above 50k unknowns.

`iterative` means, for electrostatics and magnetostatics, conjugate gradients
preconditioned by algebraic multigrid (AMGCL: smoothed aggregation with
Chebyshev smoothing, which stays stable at every element order; threaded with
OpenMP, set `OMP_NUM_THREADS` to control it). Its iteration count grows slowly,
if at all, as the mesh is refined (13-14 for order-3 tetrahedra up to 389k
unknowns, 23-63 for order-2 hexahedra up to 913k), and the multigrid
hierarchy is built once per mesh and reused for every scenario. The MQS solver
uses unpreconditioned GMRES. 3D magnetostatics uses CG preconditioned by
hypre's AMS (MPI/HYPRE build only), whose iteration count also stays roughly
constant under refinement; 3D MQS uses GMRES preconditioned block-diagonally
by AMS on `K + ωM_σ` (MPI/HYPRE build only). AMS smooths with hybrid
Gauss-Seidel on one thread and with Chebyshev on several, where Gauss-Seidel
loses strength; the iteration counts differ accordingly. The 3D MQS `direct` solver is a
sparse LU of the complex system, refactored per frequency; no MUMPS or other
parallel direct solver is included. In every solver `solver_tolerance` is the relative
residual `||b - Ax|| / ||b||`; a run that does not reach it within
`solver_max_iter` iterations prints a warning.

`frequency` is **not** valid here. It belongs on each scenario; see
[`scenarios`](#scenarios).

Output is configured in the separate top-level `output` object below.

## `output`

Optional object. Each format is enabled by including its object, including an
empty `{}`. Omit a format to disable it. With no formats, no result files are
written; coupling matrices are still printed to the console.

```json
"output": {
  "directory": "results",
  "export_fields_for_coupling_matrix": false,
  "paraview": { "directory": "paraview" },
  "gmsh": { "directory": "gmsh", "version": "2.2" },
  "hdf5": { "file": "results.h5" },
  "probes": [
    { "name": "axis", "line": { "from": [0, 0, 0], "to": [0, 0, 0.1], "count": 11 } },
    { "name": "surface", "points": [[0.02, 0, 0.019]], "entity_group": "Plate" }
  ]
}
```

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `directory` | string | `results` | Root directory, relative to the config file |
| `export_fields_for_coupling_matrix` | bool | `false` | Export fields for unit-excitation coupling solves in all enabled formats |
| `paraview` | object | omitted | Enable one ParaView collection per exported scenario |
| `paraview.directory` | string | `paraview` | Collection directory beneath the root |
| `gmsh` | object | omitted | Enable one high-order Gmsh file per exported scenario |
| `gmsh.directory` | string | `gmsh` | Gmsh directory beneath the root |
| `gmsh.version` | string | `2.2` | `2.2` or `4.1` |
| `hdf5` | object | omitted | Enable one archive for the entire run |
| `hdf5.file` | string | `results.h5` | Archive filename/path beneath the root |
| `probes` | array | omitted | Points at which every exported field is sampled; see below |

Absolute paths are preserved. Relative format destinations resolve beneath
`output.directory`, not beside the config file. Paths must be nonempty; use
`.` to select the parent directory. Unknown output settings are rejected.
The CLI `--output-directory <dir>` overrides the root, resolving relative to
the current working directory. It does not enable formats or redirect absolute
per-format destinations.

`export_fields_for_coupling_matrix` has no effect on field analyses. For coupling analyses it
defaults to false to avoid storing one field set per terminal and frequency.
HDF5 still contains the mesh and coupling matrices when this switch is false.
ParaView and Gmsh have no matrix output, so write nothing in that case.

Visualization artifacts use `scenario_000000_<label>` names in solve order,
with a sanitized, bounded scenario/terminal label. HDF5 uses the stable
`scenario_000000` ID and preserves full names and frequencies as metadata.
Each run replaces its archive and same-named visualization artifacts; use a
new root for independent runs. Files from older runs with different names are
not removed automatically. AMR replaces results on each mesh pass; a successful
run leaves only the final mesh and its scenario data in the archive.

See [the HDF5 schema](coupling_hdf5.md) for mesh reconstruction, fields, matrix
units, frequency indexing, and C# dictionary assembly.

### `output.probes`

Each probe is a named set of points at which every field the solver exports
(the same fields the other formats write: potentials, B, E, J, loss density,
...) is evaluated exactly, for each scenario whose fields are written.

| Key | Type | Required | Meaning |
|-----|------|----------|---------|
| `name` | string | yes | Unique; letters, digits, `_` and `-` |
| `points` | array | one of | Coordinate arrays, of the model's space dimension (2, or 3 for `3d`) |
| `line` | object | one of | `from` and `to` (coordinates) and `count` >= 2: points evenly spaced, both ends included |
| `entity_group` | string | no | A domain group; points are located only in its elements |

Fields jump across material interfaces (E across a permittivity change, J at
a conductor's surface), so a point on an interface takes whichever adjacent
element is found first. Give the probe the `entity_group` of the side it
should be sampled from; a point outside that group is an error, as is a point
outside the mesh.

Each probe writes `probes/<artifact>_<probe>.csv` beneath `output.directory`
(`<artifact>` as for the visualization files): a header row, then per point
its coordinates and every field's components, named `<field>_x`, `_y`, `_z`
(`_r`, `_z` in an axisymmetric model). With `hdf5` enabled the same samples
are stored under each scenario's `probes` group. Probes follow the other
formats in coupling analyses: they are written only with
`export_fields_for_coupling_matrix`.

## `simulation.amr`

Object. Optional; absent or `enabled: false` means a single solve.

| Key | Type | Default | Range / values |
|-----|------|---------|----------------|
| `enabled` | bool | `false` | Master switch |
| `max_iterations` | integer | `5` | >= 0; refine/re-solve passes |
| `max_dofs` | integer | `2000000` | <= 0 disables the cap |
| `error_fraction` | number | `0.7` | (0, 1]; bulk (Dorfler) marking fraction |
| `error_tolerance` | number | `0.0` | >= 0; <= 0 ignores the threshold |
| `conforming` | bool | `true` | Require conforming output |

Unknown keys in this block are ignored, so a producer and the solver can evolve
independently.

---

## `entity_groups`

Array of objects. The only section that names raw mesh attribute ids.

| Key | Type | Required | Meaning |
|-----|------|----------|---------|
| `name` | string | yes | Unique group name |
| `dim` | integer | yes | `1` = curve, `2` = surface, `3` = volume |
| `attribute_ids` | array of integer | yes | Positive mesh attribute ids |

Boundary versus domain is **derived** from `dim`, never prescribed:

| Relation to mesh dimension | Role |
|---|---|
| `dim == mesh dimension` | domain -- carries a material and the PDE |
| `dim == mesh dimension - 1` | boundary -- bounds the solved region |

Any other value is rejected. `dim` is stated rather than inferred because it
selects which numbering space the ids live in: Gmsh numbers physical groups
independently per dimension, so id `1` may name both a curve and a surface.

---

## `materials`

Array of objects. A material is a property bundle with no location.

| Key | Type | Required | Default | Meaning |
|-----|------|----------|---------|---------|
| `name` | string | yes | -- | Unique material name |
| `properties.sigma` | number | no | `0.0` | Conductivity [S/m] (assumes a mesh in metres) |
| `properties.epsilon_r` | number | no | `1.0` | Relative permittivity |
| `properties.mu_r` | number | no | `1.0` | Relative permeability |

Defaults are vacuum. Only the properties the physics needs are read:
electrostatics uses `epsilon_r`, magnetostatics `mu_r`, MQS `mu_r` and `sigma`.

---

## `regions`

Array of objects binding materials to domain entity groups.

| Key | Type | Required | Default | Meaning |
|-----|------|----------|---------|---------|
| `entity_group` | string | yes | -- | Must be a domain group |
| `material` | string | yes | -- | A defined material name |
| `current_constraint` | string | no | `none` | `none`, `open` |

Every domain attribute in the mesh must be claimed by exactly one region.
`current_constraint: open` marks a conductor carrying no net current -- an
induced-current-only body.

---

## `boundary_conditions`

Array of objects.

| Key | Type | Required | Meaning |
|-----|------|----------|---------|
| `entity_group` | string | yes | Must be a boundary group |
| `type` | string | yes | `dirichlet`, `neumann`, `robin` |
| `value` | number | yes | Prescribed value or flux |
| `name` | string | no | Documentation only |
| `robin_coefficient` | number | Robin only | `alpha` below; rejected on non-Robin entries |

- **dirichlet** prescribes the solution: potential `V` (electrostatics) or
  `A_phi` (magnetics).
- **neumann** prescribes the outward natural flux. Value `0` is the implicit
  natural condition.
- **robin** prescribes `n . (material flux) + alpha u = value` (outward normal,
  the same flux as Neumann). Implemented for **electrostatics** only, where
  `alpha = robin_coefficient` must be non-negative; the magnetic solvers
  reject it. As a far-field closure on a sphere of radius `R` around compact
  sources, `alpha = eps/R` with `value = 0` is exact for the monopole term
  (see [Open-boundary truncation](open_boundary.md)). Coupling-matrix runs keep
  the `alpha` term in the operator but omit `value`, like all boundary data.

Boundaries with no entry are homogeneous Neumann. Axis regularity on `r = 0` in
axisymmetric magnetic runs is imposed automatically and must **not** be
prescribed here.

---

## `terminals`

Array of objects naming drive/measurement sites.

| Key | Type | Required | Default | Meaning |
|-----|------|----------|---------|---------|
| `name` | string | yes | -- | Unique terminal name |
| `quantity` | string | yes | none | `voltage`, `current` |
| `entity_group` | string | yes | -- | Role depends on `quantity` |
| `conductor_type` | string | no | `massive` | `massive`, `stranded` |
| `direction` | object | 3D magnetic current terminals | -- | Current path of a 3D conductor; see below |

| `quantity` | Required group role | Realization |
|------------|---------------------|-------------|
| `voltage` | boundary | Essential constraint on the boundary |
| `current` | domain | Source term over the conductor domain |

`quantity` has no default: a wrong guess would silently change the physics while
still solving.

`conductor_type` applies to magnetic current terminals. In MQS, `stranded`
imposes uniform current density (a winding of fine insulated strands in
series); `massive` solves for the true current distribution including skin
and proximity effects. A `stranded` conductor carries no eddy current: its
material's `sigma` is taken as the wire's conductivity and does not enter the
field solve, so it neither screens the field nor dissipates. The winding's own
resistance is not included in `R`. 2D magnetostatics treats both as uniform `I / area`. 3D
magnetostatics gives a `massive` conductor its DC distribution `σ E` (which
differs from uniform where the path length varies, e.g. `J ∝ 1/r` in a ring),
so a `massive` 3D conductor needs a material with positive `sigma`.

`direction` is required on every current terminal of a `3d` magnetic model and
rejected everywhere else (a 2D model's current direction is fixed by its
geometry). It says where the conductor's current flows:

```json
"direction": {"type": "azimuthal", "origin": [0, 0, 0], "axis": [0, 0, 1]}
"direction": {"type": "cut", "cut": "LoopCut", "normal": [0, 1, 0]}
"direction": {"type": "electrodes", "input": "LeadIn", "output": "LeadOut"}
```

- `azimuthal`: around the axis through `origin` (default `[0, 0, 0]`) along
  `axis` (right-hand rule: positive current makes flux along `+axis` inside
  the loop). For conductors of revolution; needs no extra mesh features. The
  conductor must not reach its own axis, and should be a body of
  revolution about it (see the balance check below). In a symmetry model the
  conductor may be a sector whose two ends lie on `dirichlet` (n x A = 0)
  planes through the axis: its angular extent is read from those ends, the
  terminal current is the current through its cross-section, and the reported
  inductances and resistances are the sector's (a quarter model's are a
  quarter of the full ring's).
- `cut`: a closed loop of any shape. `cut` names a boundary group (dim 2) of
  internal faces forming a single surface that crosses the conductor once and
  severs it completely (its rim must lie on the conductor surface). The cut
  need not be planar: which side of it each element lies on follows from the
  mesh's connectivity. Its faces may also extend beyond the conductor; only
  those inside it count. `normal` says which way the current crosses the cut:
  it must cross the cut squarely somewhere (within 60° of some face's normal)
  and agree with the cut's orientation on balance, so a normal that runs
  along the cut is rejected.
- `electrodes`: an open conductor (a bus bar, a lead). Current enters through
  the `input` boundary group and leaves through `output`. Both must lie on a
  `dirichlet` (`n × A = 0`) boundary, the only place current can enter or
  leave the model consistently. Both must lie on one connected piece of it:
  otherwise a closed loop on the remaining `n × H = 0` boundary runs around
  the conductor between the pieces, and by Ampère's law (tangential H is zero
  along it) no net current could pass through it. Such a model has no
  solution and is rejected. Only part of the walls need be `dirichlet`, as
  long as that part joins the electrodes.

Whatever the type, a terminal's current must balance in its conductor: be
divergence-free, and leave it only through its electrodes. Setup measures how
much of the current density (in L2 norm) the divergence-free projection has to
remove and warns above 2%; the effect on energies and inductances is about
that fraction squared. Correct directions lose only discretization noise. The
warning catches an `azimuthal` direction about the wrong axis (about 1% per mm
of offset on a 5 cm coil) or on a shape that is not revolved, a coarsely
faceted round conductor (a few percent at 16 straight segments), and a
`stranded` current in a conductor whose cross-section varies along its path or
that has a dead-end branch (a stranded current is uniform along its path; a
`massive` one follows the conduction current and always balances). A
`stranded` current on a `cut` or `electrodes` path also loses a little where
the conductor turns at different radii across its section, because the
uniform current along a solved path is not exactly divergence-free there:
2.6% in TEAM 7's racetrack coil, whatever the mesh (see
[the formulation](math_formulation.md) and `examples/team7/README.md`).

For `cut` and `electrodes` the direction comes from a unit conduction
potential solved on the conductor, weighted by its `sigma` if it is
`massive`. A `stranded` conductor carries a uniform current density
`I / A_cs` along its path, as a coil of many equal fine strands does. `A_cs`
is its cross-section, so the terminal's `value` is the total current through
it, exactly as in 2D. For an azimuthal conductor `A_cs` is the meridional
area, so a 3D coil and its axisymmetric model carry the same current density.
A `massive` conductor carries `σ E`: at DC its conduction current (conductance
`G = ∫σ|∇v|²`), in MQS the solved eddy-current distribution driven by a port
voltage, with `R → 1/G` as the frequency goes to zero.

In a 3D MQS model every conducting region without a terminal (a shield, a
tank wall) carries the eddy currents the field induces in it. Region
`current_constraint` (`open`) is a 2D device for conductors whose ends the
planar model cannot represent, and is rejected for `3d`.

---

## `scenarios`

Array of objects, one per field solve. Static `coupling_matrix` analyses ignore
these scenarios and synthesize unit-drive scenarios. MQS coupling uses their
unique frequencies, in ascending order, and synthesizes unit-current terminal
excitations at each frequency; prescribed excitation values are ignored.

| Key | Type | Required | Meaning |
|-----|------|----------|---------|
| `name` | string | yes | Scenario name; labels output |
| `excitations` | array | no | Per-terminal values |
| `frequency` | number / array / object | MQS only | See below |

Each entry of `excitations`:

| Key | Type | Required | Meaning |
|-----|------|----------|---------|
| `terminal` | string | yes | A defined terminal name |
| `value` | number | yes | Volts or amps, per that terminal's `quantity` |

A terminal omitted from `excitations` defaults to zero of its quantity:
grounded for voltage, open for current. Omission is meaningful, not an error.

> **Excitation values are PEAK (amplitude) phasors in time-harmonic runs.**
> There is no rms/peak selector and no conversion. See
> [faq.md](faq.md#are-excitations-peak-or-rms).

### `frequency` (MQS only)

Three accepted forms, all expanding to one solve per point:

```json
"frequency": 60.0                                              // single
"frequency": [10.0, 100.0, 1000.0]                             // explicit list
"frequency": { "scale": "log", "start": 10.0,
			   "stop": 1000.0, "points": 5 }                   // sweep
```

Sweep object keys:

| Key | Type | Required | Values |
|-----|------|----------|--------|
| `scale` | string | yes | `log`, `linear` |
| `start` | number | yes | > 0 |
| `stop` | number | yes | > 0 |
| `points` | integer | yes | >= 1; includes both endpoints |

With `points: 1` only `start` is solved. Swept scenarios are named
`<name>_f<n>_<frequency>Hz`. Frequency is required and positive for MQS and
ignored by the static solvers.

---

## Validation rules

All checks run before solving, and every failure is reported together rather
than one at a time.

**Structural**
- Root is an object; `simulation` is present.
- Every section has the expected JSON type.
- Superseded names are rejected with the replacement named (see below).

**Names and references**
- Entity group names are unique.
- Every `entity_group` reference resolves to a defined group.
- Every group's `dim` is the mesh dimension (domain) or one below it
  (boundary); anything else is rejected, naming both acceptable values.
- Every referenced group has the role its use requires.
- Every `material` reference resolves to a defined material.
- Every `terminal` reference in an excitation resolves to a defined terminal.

**Coverage and conflicts**
- Every mesh domain attribute is claimed by exactly one region.
- Attribute ids are positive integers.
- No DOF is pinned to two different values by two boundary conditions.
- No boundary entity carries two boundary conditions.
- No DOF belongs to two terminals, or to a terminal and a Dirichlet boundary
  condition at once. Because terminal values vary per scenario, any shared DOF
  is guaranteed to conflict in some scenario.

**Ranges**
- `order` in [1, 10]; `solver_tolerance` in (0, 1); `solver_max_iter` >= 1;
  `amr.error_fraction` in (0, 1];
  `amr.max_iterations` and `amr.error_tolerance` non-negative.

**Physics-specific**
- MQS requires a positive frequency on every scenario.
- `simulation.frequency` is rejected for MQS; frequency belongs on scenarios.
- Robin boundary conditions are electrostatics-only and need a non-negative
  `robin_coefficient`.
- Region `current_constraint` is rejected for `geometry_type` `3d`.

---

## Superseded names

Rejected with a message naming the replacement:

| Superseded | Use instead |
|------------|-------------|
| `simulation.physics` | `simulation.physics_type` |
| `simulation.type` | `simulation.physics_type` |
| `simulation.geometry` | `simulation.geometry_type` |
| `simulation.output_paraview`, `simulation.output_gmsh`, `simulation.output_hdf5` | Top-level `output` format objects |
| `simulation.results_file`, `simulation.results_filename` | `output.hdf5.file` |
| `simulation.results_path` | `output.directory` |
| `simulation.gmsh_format` | `output.gmsh.version` |
| `boundaries` | `boundary_conditions` |
| entity group `kind` | `dim` |
| terminal `excitation_type` | `quantity` |
| terminal `excitation` | `quantity` |

A capitalized boundary condition type (for example `"Dirichlet"`) is reported
with its lowercase spelling rather than a generic rejection.
