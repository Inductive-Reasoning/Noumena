# Input File Format

The solver reads one JSON configuration file. This document explains the model
the file describes, defines every term, and shows a worked example.

Companion documents:

- [config_reference.md](config_reference.md) -- the normative key-by-key
  reference: every key, type, default, accepted value, and validation rule.
  Consult it when you need to know *what is legal*.
- [boundary_and_terminal_model.md](boundary_and_terminal_model.md) -- the
  boundary/terminal/coupling vocabulary in depth, and why those terms were
  chosen.

This document is the guide; the reference is the schema.

---

## 1. The model in one page

A configuration answers five questions, in this order. The order matters,
because each layer only refers to names defined by the layer above it.

| # | Question | Section | Refers to |
|---|----------|---------|-----------|
| 1 | How is the run configured? | `simulation` | - |
| 2 | Which parts of the mesh have names? | `entity_groups` | mesh attribute ids |
| 3 | What are those parts made of? | `materials`, `regions` | entity groups |
| 4 | How is the model constrained and driven? | `boundary_conditions`, `terminals` | entity groups |
| 5 | What is actually solved? | `scenarios` | terminals |

The mesh is the substrate. Everything else is a layer of naming and physics
placed on top of it, and nothing in the config refers to raw mesh numbers except
`entity_groups`.

```
mesh attribute ids
		|
		v
  entity_groups ------------------+--------------------+
		|                         |                    |
		v                         v                    v
	regions --> materials  boundary_conditions     terminals
														|
														v
												  scenarios
												 (excitations)
```

---

## 2. Glossary

These are the load-bearing terms. Each is defined once here and used
consistently everywhere else.

| Term | Definition |
|------|------------|
| **Attribute id** | An integer tag Gmsh/MFEM attaches to mesh elements. Domain attributes tag area/volume elements; boundary attributes tag curve/surface elements. The two numbering spaces are independent -- attribute `1` as a domain is unrelated to attribute `1` as a boundary. |
| **Entity group** | A *name* bound to a set of attribute ids of one entity dimension. The sole bridge between mesh numbering and physics. Every other section refers to groups by name, never to raw ids. |
| **Domain** | A full-dimensional part of the model (an area in 2D). Carries a material and the PDE. |
| **Boundary** | A codimension-one part (a curve in 2D) bounding the solved region. |
| **Material** | A named set of physical properties: `sigma`, `epsilon_r`, `mu_r`. Purely a property bundle; it has no location. |
| **Region** | The assignment of a material to a domain entity group. This is what gives a material a location. |
| **Boundary condition** | A constraint applied on a boundary entity group: either Dirichlet (prescribed value) or Neumann (prescribed flux). Prescribed once; fixed for the entire run. |
| **Terminal** | A named connection through which the model is driven or measured. Its value is *not* fixed here -- scenarios supply it. |
| **Excitation** | One scenario's setting of one terminal. Volts for a voltage terminal, amps for a current terminal. |
| **Scenario** | One solve: a set of excitations plus, for MQS, a frequency. |
| **Coupling matrix** | The terminal-by-terminal result extracted by driving each terminal in turn (capacitance, inductance, or R/L). |

### Distinctions that matter

**Material vs. region.** A material is *what something is made of*; a region is
*where that material lives*. Two regions may share one material. This split is
why materials can be reused across a model without duplication.

**Boundary condition vs. terminal.** Both can pin DOFs, so the distinction is
not "essential vs. not". It is **who owns the value**:

- a boundary condition's value is prescribed in the file and fixed for the run;
- a terminal's value is supplied per scenario.

A grounded far-field boundary is a boundary condition. A driven electrode is a
terminal. See the companion document for the full argument.

**Terminal vs. excitation.** The terminal is the *site* and persists across the
whole run; the excitation is a *value at that site* in one scenario.

---

## 3. Sections

This section explains what each part of the file is *for*. For the complete
key/type/default tables, see [config_reference.md](config_reference.md).

### 3.1 `simulation` (required)

Run-level settings: which physics to solve, which mesh to read, what geometry
the mesh represents, whether to extract a coupling matrix, and how to solve and
export. Only `physics_type` and `mesh` are required; everything else has a
default. Full listing: [`simulation`](config_reference.md#simulation).

`direct` is the default solver because a coupling-matrix run amortizes one
factorization over every terminal's right-hand side, and its accuracy does not
depend on a residual tolerance.

#### `simulation.amr`

Optional adaptive mesh refinement block. When AMR is absent or disabled the
solver performs a single solve; when enabled it repeats a refine/re-solve cycle
until an iteration, DOF, or error limit is hit. Full listing:
[`simulation.amr`](config_reference.md#simulationamr).

### 3.2 `entity_groups`

Array of named attribute-id sets. The only section that mentions raw mesh
numbers. Each group binds a `name` to a list of `attribute_ids` of one entity
dimension `dim`. Full listing:
[`entity_groups`](config_reference.md#entity_groups).

```json
"entity_groups": [
  { "name": "Air",         "dim": 2, "attribute_ids": [1] },
  { "name": "Dielectric",  "dim": 2, "attribute_ids": [2] },
  { "name": "TopPlate",    "dim": 1, "attribute_ids": [1] },
  { "name": "FarField",    "dim": 1, "attribute_ids": [3] }
]
```

`dim` is the topological dimension of the entities the ids name. It is stated,
not inferred, because it selects *which numbering space the ids live in*. Gmsh
numbers physical groups independently per dimension, so one model may use id `1`
for both a curve and a surface; MFEM preserves that split as `bdr_attributes`
versus `attributes`. In the example above `Air` and `TopPlate` both use id `1`
and refer to entirely different entities. Without `dim` the ids are ambiguous.

Boundary versus domain is **derived** from `dim`, not prescribed:

| Relation to mesh dimension | Role |
|---|---|
| `dim == mesh dimension` | domain -- carries a material and the PDE |
| `dim == mesh dimension - 1` | boundary -- bounds the solved region |

So in a 2D mesh `dim: 2` is a domain and `dim: 1` is a boundary; in 3D those
become `3` and `2`. Any other value names entities the solver cannot use and is
rejected, naming both acceptable values. The derived role is then checked
against use: a boundary condition must reference a boundary group, a region must
reference a domain group.

### 3.3 `materials`

Array of named property bundles. A material has no location -- it is *what
something is made of*, and `regions` decides where it lives. Properties are
`sigma`, `epsilon_r`, and `mu_r`; omitted ones default to vacuum. Full listing:
[`materials`](config_reference.md#materials).

Only the properties the physics needs are read:
electrostatics uses `epsilon_r`, magnetostatics `mu_r`, MQS `mu_r` and `sigma`.

Properties are SI: `sigma` in S/m, with `epsilon_r` and `mu_r` dimensionless
relative values. Because these are per-metre quantities, they only compose
correctly with a mesh whose coordinates are in **metres**. See
[Units](config_reference.md#units).

### 3.4 `regions`

Array binding materials to domain entity groups. This is what gives a material
a location. Full listing: [`regions`](config_reference.md#regions).

**Every domain attribute in the mesh must be claimed by exactly one region.**
Unclaimed attributes and doubly-claimed attributes are both errors. This is
strict on purpose: an unclaimed domain silently receives default (vacuum)
properties and produces a plausible but wrong answer.

`current_constraint: open` marks a conductor carrying no net current -- an
induced-current-only body.

### 3.5 `boundary_conditions`

Array of boundary conditions, each applying a `dirichlet`, `neumann`, or
`robin` constraint with a fixed `value` to a boundary entity group. Full
listing: [`boundary_conditions`](config_reference.md#boundary_conditions).

- **dirichlet** prescribes the solution: potential `V` (electrostatics) or
  `A_phi` (magnetics).
- **neumann** prescribes the outward natural flux. Value `0` is the implicit
  natural condition, so a boundary with no entry behaves as homogeneous Neumann.
- **robin** is parsed and reserved but **not implemented**; the solver rejects it.
  `robin_coefficient` is required for Robin entries and rejected on all others.

Entries omitted entirely are homogeneous Neumann. Axis regularity on `r = 0` in
axisymmetric magnetic runs is imposed automatically and must **not** be prescribed.

### 3.6 `terminals`

Array of named drive/measurement sites. A terminal has a `name`, a `quantity`
(`voltage` or `current`), and an `entity_group`. Full listing:
[`terminals`](config_reference.md#terminals).

`quantity` decides which role of group is required, and how the terminal is
realized:

| `quantity` | Group role | Realization |
|------------|------------|-------------|
| `voltage` | boundary | Essential constraint on the boundary |
| `current` | domain | Source term over the conductor domain |

`quantity` has no default. A wrong guess here would silently change the physics
while still solving, so it must be stated.

`conductor_type` applies to current terminals in MQS: `stranded` imposes uniform
current density (a winding of insulated strands; its material `sigma` does not
enter the field solve, so no eddy current flows in it); `massive` solves for
the true current distribution including skin and proximity effects.

### 3.7 `scenarios`

Array of solves. Ignored when `analysis_type` is `coupling_matrix`, which
synthesizes its own unit-drive scenarios. Each scenario has a `name`, an
optional list of `excitations` (a `terminal` and a `value`), and for MQS a
`frequency`. Full listing: [`scenarios`](config_reference.md#scenarios).

**A terminal omitted from `excitations` defaults to zero** of its quantity:
grounded for voltage, open for current. Omission is meaningful, not an error.

**Excitation values are peak (amplitude) phasors** in time-harmonic runs, not
rms. There is no selector and no conversion; see
[faq.md](faq.md#are-excitations-peak-or-rms).

#### Frequency (MQS only)

Frequency is a property of the scenario, not of the run, so a single file can
sweep. It takes three forms -- a single number, an explicit list, or a sweep
object with `scale`, `start`, `stop`, and `points` -- all expanding to one solve
per point:

```json
"frequency": 60.0                                              // single
"frequency": [10.0, 100.0, 1000.0]                             // explicit list
"frequency": { "scale": "log", "start": 10.0,
			   "stop": 1000.0, "points": 5 }                   // sweep
```

Swept scenarios are named `<name>_f<n>_<frequency>Hz`. Frequency is required and
positive for MQS and ignored by the static solvers. Full listing:
[`frequency`](config_reference.md#frequency-mqs-only).

---

## 4. Analysis types

### `field`

Solves the prescribed scenarios and writes the resulting fields.

### `coupling_matrix`

Ignores prescribed scenarios. Drives each terminal in turn to unit value with all
others held at zero, and assembles the terminal-by-terminal matrix:

| Physics | Matrix | Unit |
|---------|--------|------|
| Electrostatics | Capacitance | F |
| Magnetostatics | Inductance | H |
| MQS | Resistance + inductance, per frequency point | Ohm, H |

Rows and columns follow the terminal definition order. For MQS, each scenario
supplies a frequency point and a separate labeled R/L pair is written per point.

---

## 5. Validation rules

Every configuration is checked before solving, and all failures are reported
together rather than one at a time. The checks fall into five families:

- **Structural** -- correct JSON types, `simulation` present, superseded names
  rejected with their replacement named.
- **Names and references** -- unique group names, and every reference to a
  group, material, or terminal resolves to something defined.
- **Roles** -- every referenced group has the role its use requires: boundary
  conditions need boundary groups, regions need domain groups.
- **Coverage and conflicts** -- every mesh domain attribute claimed exactly
  once, and no DOF pinned by two competing sources.
- **Ranges** -- numeric bounds on solver and AMR settings.

The complete enumerated rule list, including the exact numeric ranges and the
superseded-name table, is in
[config_reference.md](config_reference.md#validation-rules).

The strictness around coverage is deliberate. An unclaimed domain attribute
would silently receive vacuum properties and produce a plausible but wrong
answer, so it is an error rather than a warning. Likewise, a DOF shared between
a terminal and a Dirichlet condition is rejected outright: because terminal
values vary per scenario, such a DOF is guaranteed to conflict in some scenario
even if it happens to agree in the one you tested.

---

## 6. Worked example

Axisymmetric parallel-plate capacitor with a dielectric, in air, truncated by a
grounded far-field boundary.

```json
{
  "simulation": {
	"physics_type": "electrostatics",
	"geometry_type": "axisymmetric",
	"mesh": "capacitor.mesh",
	"order": 2,
	"analysis_type": "coupling_matrix"
  },
  "output": {
	"directory": "results",
	"export_fields_for_coupling_matrix": true,
	"paraview": {},
	"hdf5": {}
  },

  "entity_groups": [
	{ "name": "Air",         "dim": 2, "attribute_ids": [1] },
	{ "name": "Dielectric",  "dim": 2, "attribute_ids": [2] },
	{ "name": "TopPlate",    "dim": 1, "attribute_ids": [1] },
	{ "name": "BottomPlate", "dim": 1, "attribute_ids": [2] },
	{ "name": "FarField",    "dim": 1, "attribute_ids": [3] }
  ],

  "materials": [
	{ "name": "Air",        "properties": { "epsilon_r": 1.0 } },
	{ "name": "Dielectric", "properties": { "epsilon_r": 2.5 } }
  ],

  "regions": [
	{ "entity_group": "Air",        "material": "Air" },
	{ "entity_group": "Dielectric", "material": "Dielectric" }
  ],

  "boundary_conditions": [
	{ "entity_group": "FarField", "type": "dirichlet", "value": 0.0 }
  ],

  "terminals": [
	{ "name": "Top",    "quantity": "voltage", "entity_group": "TopPlate" },
	{ "name": "Bottom", "quantity": "voltage", "entity_group": "BottomPlate" }
  ]
}
```

Note what is *absent*: no `scenarios`, because `coupling_matrix` synthesizes
them. Switching to `"analysis_type": "field"` would require adding:

```json
"scenarios": [
  {
	"name": "Charged",
	"excitations": [ { "terminal": "Top", "value": 1.0 } ]
  }
]
```

`Bottom` is omitted and therefore grounded.

---

## 7. Terminology review

Terms were assessed against one test: **does the name say what the thing is?**

### Kept

| Term | Why |
|------|-----|
| **entity group** | Role-neutral, so one concept covers both boundary and domain naming. `physical_group` (Gmsh) would tie the schema to one mesher; `attribute_set` (MFEM) leaks the FE library into the user's file. |
| **region** | Standard for a material-assigned subdomain, and correctly distinct from the material itself. |
| **material** | Universal. |
| **terminal** | Matches COMSOL. Deliberately not `port`, which means a modal/S-parameter boundary and would mislead in a static/quasi-static code. |
| **excitation** | Matches Ansys, where prescribed drives live under Excitations rather than boundary conditions. Correctly implies a scenario-supplied value. |
| **scenario** | Neutral and accurate: one parameterized solve. `case` and `step` both carry unwanted baggage (`step` implies time-stepping). |
| **coupling matrix** | Names the content -- terminal-to-terminal coupling -- rather than merely the index set. |

### Corrected

Four names disagreed with the model they described. Each was renamed rather than
kept, since the internal model already used the better name in every case and the
JSON surface was the only thing out of step. All four superseded spellings are
now rejected with a message naming the replacement.

| Was | Now | Why |
|-----|-----|-----|
| `boundaries` | `boundary_conditions` | The section defines *conditions*, not geometry -- and the geometry it references is already named by entity groups. The internal type was already `BoundaryCondition`. |
| `excitation_type` | `quantity` | The value is `voltage` or `current` -- a quantity, not a type of excitation. Matches the internal `Quantity` enum. |
| `"Dirichlet"` | `"dirichlet"` | Every other enum value in the schema is lowercase. A capitalized value now reports its lowercase spelling rather than a bare rejection. |

One internal name moved with them: `Terminal::ExcitationType` ->
`Terminal::DriveQuantity` (`Quantity` alone would collide with the enum type).

`entity_groups[].dim` was briefly renamed to `kind: "boundary" | "domain"` and
then reverted. The role is derived from the dimension by comparing it to the
mesh dimension, so prescribing it separately stored a fact the mesh already
determines and allowed the two to disagree. More importantly, `dim` names the
mesher's own concept -- Gmsh and MFEM both key their attribute namespaces on
entity dimension -- and so keeps its meaning across tools and in 3D, where a
boundary is a surface rather than a curve. `EntityKind` was likewise reverted to
a plain `int Dim` with derived `IsDomain()`/`IsBoundary()` helpers.

### Deliberate, not an inconsistency

**`terminals[].name` vs. `excitations[].terminal`.** Definition sites use `name`;
reference sites use the referenced type as the key. This holds throughout the
schema and is noted only to confirm it is intentional.

### Rejected

| Proposed | Rejected because |
|----------|------------------|
| `TerminalMatrix` for coupling matrix | Names the index set, not the content |
| Separate exterior/interior boundary kinds | Nothing branches on it; entity group names already convey it to the reader |
| `port` for terminal | Reserved for modal/S-parameter use |
| Folding terminals into boundary conditions | Only voltage terminals are boundary conditions; current terminals are domain sources |
