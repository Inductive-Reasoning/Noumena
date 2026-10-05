# Boundary, Terminal, and Constraint Model

How this project names and models the things that constrain a solution: boundary
conditions, terminals, and the essential-DOF sets they produce.

This document is **prescriptive**. It describes the model the code implements,
and records the reasoning so the conclusions can be re-examined rather than
re-litigated. It was written after a terminology review in which an earlier
refactor took a wrong turn; the wrong turn is documented at the end, because the
reasoning that produced it is easy to repeat.

---

## Summary table

| # | Decision | Status |
|---|----------|--------|
| 1 | Drop "closure"; the word is topologically backwards | **Done** |
| 2 | A terminal is not a kind of boundary condition | **Done** |
holds only prescribed boundary conditions
| 4 | `ess_bdr` is an explicit union of several independent sources | **Done** |
| 5 | No exterior/internal boundary distinction in the solver | **Decided - do not add** |
| 6 | Keep "terminal"; avoid "port" | **Decided - keep** |
| 7 | Predicate methods on `MarkedBoundaryCondition` | **Done** |
| 8 | `BuildEssentialBoundaryMarker()` hook owns the ess_bdr union | **Done** |
| 9 | Distinct names per essential-DOF index space in MQS | **Done** |
| 10 | Keep "coupling matrix"; reject `TerminalMatrix` | **Decided - keep** |

---

## 1. Three layers, kept separate

The vocabulary splits into three layers that answer different questions. Most
naming confusion in this area comes from collapsing them.

### Layer 1 - Geometry: where does the entity live?

| Term | Meaning |
|------|---------|
| **Domain** | A full-dimensional region carrying a material and a PDE. |
| **Boundary** | A codimension-one entity bounding exactly one modeled domain. Part of the mathematical boundary of the solved region. |
| **Interface** | A codimension-one entity shared by two modeled domains. Not part of the outer boundary of the solved region. |

Whether a given surface is a boundary or an interface is **not** a property of
the geometry. It depends on what was meshed. The surface of an electrode is:

- a **boundary**, if the conductor interior is excluded from the mesh (the field
  problem lives only in the surrounding insulator, and the electrode is a hole
  punched out of it);
- an **interface**, if the conductor is meshed as its own domain.

Both cases occur in this project. Electrostatic models typically exclude
conductor interiors. The MQS solver meshes its conductors -- `ComputePortConductance()`
integrates conductivity over the conductor cross-section, which requires them to
be meshed domains.

A boundary component may be geometrically enclosed by another boundary component
and is still a boundary. The far-field ring and each electrode surface of a
capacitor model are all components of the same boundary; that boundary simply
is not connected.

### Layer 2 - Formulation: what condition is imposed?

| Term | Meaning |
|------|---------|
| **Boundary condition** | Data imposed on a boundary. Dirichlet, Neumann, or Robin. |
| **Essential condition** | Restricts the trial space by prescribing DOF values. Dirichlet, in the formulations implemented here. |
| **Natural condition** | Enters weakly through the load vector. Neumann. |
| **Interface condition** | A continuity or jump relation across an interface. |
| **Constraint** | The general term for any restriction on solution unknowns, including ones that are not boundary conditions. |

`Dirichlet`/`Neumann` classifies the PDE; `essential`/`natural` classifies the
FEM treatment. They coincide in this codebase, but `essential` is the better
word inside the solver because it states *why the code cares*: these are the
DOFs that get eliminated.

Every essential boundary condition is a constraint. Not every constraint is a
boundary condition -- a gauge condition or a floating-conductor equation is a
constraint with no boundary attached.

### Layer 3 - Coupling: how does the model meet the outside world?

| Term | Meaning |
|------|---------|
| **Terminal** | A named connection through which the model is driven or measured. |
| **Excitation** | The value a scenario assigns to a terminal. |
| **Coupling matrix** | The terminal-by-terminal matrix extracted by driving each terminal in turn. |

A terminal names *intent* ("this is where 1 V is applied"), not *mechanism*. The
mechanism is chosen by the formulation.

---

## 2. Why a terminal is not a boundary condition

This is the central decision, and it is settled by the existing code rather than
by principle. The four terminal kinds already implemented resolve four different
ways:

| Terminal | Realization | Boundary condition? |
|----------|-------------|---------------------|
| Electrostatic voltage | Essential BC on a boundary | Yes |
| Magnetostatic current | Domain source (`BuildTerminalCurrentDensity`) | No |
| MQS stranded current | Domain source | No |
| MQS massive current | Port-block coupling with an added global unknown | No |

Three of four are not boundary conditions. `Terminal::EntityGroupName` already
refers to a **boundary** group for voltage terminals and a **domain** group for
current terminals -- the config type is explicitly dual, and `problem_config.hpp`
documents this.

Modeling a terminal as a boundary condition therefore describes one of four
cases and misdescribes the rest. Any such abstraction has to fabricate a
`BoundaryCondition` for terminals whose value is meaningless, which is a
reliable sign the type is wrong.

Anticipated future terminal kinds resolve to still other mechanisms -- a floating
conductor becomes an added unknown plus an integral charge constraint; a
circuit-coupled terminal becomes an external network equation. The separation
is not speculative future-proofing; it is already required.

### Correct relationship

> A terminal is a model-level connection. Each solver **realizes** its terminals
> into formulation-specific contributions: an essential constraint, a natural
> load, a domain source, an integral constraint, or a coupled unknown. Whatever
> subset of those produces essential DOFs is then unioned into `ess_bdr`.

The union is real and necessary. It just happens at the level of *markers*,
not at the level of *boundary condition objects*.

---

## 3. Why not "closure"

"Closure" was used to mean "a prescribed boundary condition, as opposed to a
terminal." It should be removed for two reasons.

**It inverts the topology.** The closure of a domain is the domain *together
with* its boundary. Using "closure" to mean "the boundary" names the larger set
while intending the smaller one, in a codebase whose whole subject is domains
and boundaries.

**It names the wrong axis.** The intended contrast was never geometric. It was
about *who owns the value*: fixed for the whole run, versus supplied per
scenario. But once terminals leave `BoundaryConditionSet` entirely, the contrast
has nothing left to distinguish -- every entry is a prescribed boundary condition.
The word disappears rather than needing a replacement.

There is a legitimate physics sense of "closure" (a closure relation that makes
an underdetermined system determinate). That sense is unrelated to what the code
meant, and its existence is another reason to avoid the word here.

---

## 4. Why no exterior/internal distinction

It is tempting to distinguish the far-field truncation boundary from an enclosed
electrode boundary. **Do not add this.** Three reasons:

1. **Nothing would branch on it.** Dirichlet on the outer ring and Dirichlet on
   an electrode are the same operation on the same kind of entity. Assembly,
   essential-DOF elimination, and projection are all indifferent.
2. **It is not robustly computable.** "Is this boundary component the outer one?"
   needs winding numbers or containment tests, and is genuinely ambiguous for
   multiply-connected or truncated-unbounded domains.
3. **Entity group names already carry it.** `FarField`, `TopPlate`, and `Shield`
   communicate the distinction to the human reading the config, which is the
   only consumer that needs it.

The distinction the code *does* act on -- fixed value versus scenario-supplied
value -- is orthogonal. A grounded internal shield is fixed; a driven outer
enclosure would be scenario-supplied.

If a future feature genuinely needs the geometric classification, add it as an
independent property of the entity group, not as a kind of boundary condition.

---

## 5. Why "terminal" and not "port"

**Terminal** matches COMSOL's usage: a named boundary feature carrying voltage,
current, charge, or a circuit connection.

**Port** in electromagnetics means a *modal* boundary where a waveguide mode is
launched and S-parameters extracted (HFSS wave ports, lumped ports). This project
solves statics and quasi-statics with no modal decomposition, so "port" would be
actively misleading. It should stay reserved in case a wave formulation is ever
added.

Ansys is a partial counterexample worth knowing: Maxwell files prescribed
voltages under **Excitations** rather than boundary conditions, which supports
keeping terminals separate even when they realize as essential BCs. Q3D uses
**source**/**sink**/**net** for matrix extraction -- more specialized than the
generic terminal here, and a reasonable direction if connection topology ever
matters.

Magnetic specializations (**coil**, **winding**) sit *above* the terminal
abstraction rather than replacing it.

---

## 5a. Why "coupling matrix"

`AnalysisType::CouplingMatrix` drives each terminal in turn and assembles the
terminal-by-terminal result: a capacitance matrix (electrostatic), an inductance
matrix (magnetostatic), or a resistance/inductance pair per frequency (MQS). The
individual names are all standard; the generic umbrella is ours, because the
field does not supply a single accepted word for the family.

**The term is kept.** "Coupling" names what the matrix *contains* -- the
terminal-to-terminal coupling of the model. That is the property being measured,
and it is what makes one generic name legitimate across three formulations whose
units differ.

**`TerminalMatrix` was considered and rejected.** It names the index set but not
the content: a matrix indexed by terminals could hold anything. The same test
retired "closure" -- a name should say what a thing *is*, not merely what it is
adjacent to.

### The collision to be aware of

In multiphysics FEM, "coupling matrix" usually means the *off-diagonal block*
coupling two fields in a monolithic system (piezoelectric, fluid-structure).
That reading is available to a newcomer here, and MQS does assemble a genuine
block system with port corner blocks, so the misreading is plausible rather than
far-fetched. It is accepted as a known cost, not overlooked:

> In this codebase, a coupling matrix is always a terminal-by-terminal
> extraction result. It is never an inter-field block of the system operator.

For reference, no vendor uses the phrase for this operation. Ansys Q3D/Maxwell
say **Matrix** (matrix setup/extraction); COMSOL performs a **terminal sweep**
producing a **capacitance matrix**; the industry-wide phrase for the activity is
**parameter extraction**. Any of these would be defensible; none is clearly
better than what is already in use and already documented in the config schema.

---

## 6. Target model

### `BoundaryCondition` (config)

Unchanged. Prescribed conditions only, one per boundary entity group.

### `MarkedBoundaryCondition`

A `BoundaryCondition` paired with its resolved mesh marker. It should carry no
role tag and no terminal name, and every instance should correspond to a real
prescribed condition.

```cpp
struct MarkedBoundaryCondition {
	mfem::Array<int> Marker;
	BoundaryCondition Condition;

	bool IsDirichlet() const;
	bool IsNeumann() const;
	bool IsRobin() const;
	bool IsNonzeroDirichlet() const;
};
```

`IsEssential()` is deliberately absent: with terminals removed it is a synonym
for `IsDirichlet()`, and a synonym implies a distinction that no longer exists.
`BuildBoundaryConditions()` rejects Robin for any solver whose
`SupportsRobin()` is false (the magnetic ones); electrostatics assembles it.

### `BoundaryConditionSet`

Holds prescribed boundary conditions and the folds over them.
no `TerminalMarkers()`, no `Closures()` -- with terminals gone, `Closures()`
would return everything.

```cpp
class BoundaryConditionSet {
public:
	void Add(mfem::Array<int> marker, const BoundaryCondition& condition);
	mfem::Array<int> DirichletMarker(int max_bdr_attr) const;
	// iteration
};
```

Named `DirichletMarker` rather than `EssentialMarker`: it is the contribution of
*this set*, not the finished essential marker, which also draws on terminals and
axis regularity.

### Terminal resolution

Terminal markers stay with the solver that needs them, in a plain
`unordered_map<string, mfem::Array<int>>`. A dedicated `ResolvedTerminal` type is
**not** recommended at present -- only electrostatics needs per-name lookup, and a
wrapper serving one consumer adds a layer without removing one. Revisit if a
second solver needs the same lookup.

### Building `ess_bdr`

The union is owned by a virtual hook, `PhysicsSolver::BuildEssentialBoundaryMarker()`.
The base supplies the prescribed Dirichlet conditions; each formulation overrides,
calls the base, and merges its own essential sources:

```cpp
// PhysicsSolver - the default
ess_bdr = boundary_conditions.DirichletMarker(mesh.bdr_attributes.Max());

// ElectrostaticSolver - voltage terminals also pin DOFs
PhysicsSolver::BuildEssentialBoundaryMarker();
for (const auto& [name, marker] : terminal_markers) MergeMarker(ess_bdr, marker);

// MagneticSolver - axis regularity, A_phi = 0 on r = 0
PhysicsSolver::BuildEssentialBoundaryMarker();
MergeMarker(ess_bdr, axis_boundary);
```

A hook rather than an open-coded union at each call site, because the sources
must all be merged *before* `BuildOperators()` reads `ess_bdr`. Previously that
ordering was a comment on `ImposeAxisRegularity()` telling the caller when to
invoke it; now it is structural, and "what pins DOFs in this formulation?" has
exactly one place to look per solver. The three solvers also gain a shared
extension point, which is what keeps their high-level structure parallel.

`MergeMarker(target, source)` is the shared helper the overrides are built from.

### Essential DOF lists are not interchangeable

`ess_bdr` is a marker over boundary *attributes*. `ess_tdof_list` is a list of
*true DOF indices* derived from it, and is valid only for the mesh and space it
was built against.

MQS does not solve on the FE space at all: it solves a packed
`[Re | Im] x [mesh | port]` block system, so its essential list indexes a larger
space. It therefore keeps two separately named lists and does **not** use the
inherited `ess_tdof_list`:

| Member | Index space |
|--------|-------------|
| `ess_mesh_tdofs` | `[0, N_DOFs)` - the FE space |
| `ess_packed_tdofs` | `[0, N_DOFs + N_Ports)` - the packed block system |

Reusing one name for both index spaces is a silent-wrong-answer hazard: the
lists differ in length and meaning, and mixing them eliminates the wrong rows
without any type error. Name them apart.

---

## 7. Naming reference

| Avoid | Use | Why |
|-------|-----|-----|
| closure | boundary condition | Closure means domain *plus* boundary |
| closure BC | prescribed boundary condition | |
| `closure_bcs` | `boundary_conditions` | |
| `BuildClosureBcs()` | `BuildBoundaryConditions()` | |
| `IsDrivenClosure()` | `IsNonzeroDirichlet()` | |
| `BoundaryRole` | (remove) | Conflates geometry with value ownership |
| port (for a DC/AC connection) | terminal | Port implies modal/S-parameter |
| terminal matrix | coupling matrix | Names the index set, not the content |
| boundary condition (off the boundary) | constraint | Not every constraint is a BC |

Keep: **domain**, **boundary**, **interface**, **essential**, **natural**,
**Dirichlet**, **Neumann**, **terminal**, **excitation**, **region**,
**entity group**.

---

## 8. Recorded misstep: terminals folded into `BoundaryConditionSet`

An earlier refactor moved voltage terminals *into* `BoundaryConditionSet` via an
`AddTerminal()` method, with a `BoundaryRole { Closure, Terminal }` tag to keep
them distinguishable. Tests passed and behavior was unchanged. It was still
wrong, and the reasoning is worth preserving.

**The stated motivation was overstated.** The claim was that essential BCs had
"two answers in two containers." In fact the electrostatic solver gathered
markers from two legitimately different sources and unioned them correctly. It
was verbose -- an intermediate `vector<Array<int>>` that existed only to be folded
-- but it was honest. A readability issue was inflated into a correctness-flavored
one, which justified a larger change than the evidence supported.

**The tell was in the code.** `AddTerminal()` had to construct a
`BoundaryCondition` with `Value = 0.0` and a comment explaining that the value
was meaningless. Fabricating a value to satisfy a type signature means the type
is wrong. Everything downstream -- the `IsEssential()`/`IsDrivenClosure()` split,
the `Closures()` filter to undo the merge -- existed only to work around that
initial fabrication.

**A green test suite did not catch it.** The refactor was behavior-preserving; it
degraded the *model*, not the results. Tests cannot detect that, which is why an
abstraction that requires a lie to construct should be treated as a defect even
when everything passes.

**What survived.** The predicate methods on `MarkedBoundaryCondition` were a
genuine improvement -- they removed six open-coded
`Condition.Type == BoundaryConditionType::Dirichlet` checks -- and are unaffected
by reverting the terminal merge.

**General lesson.** When a refactor requires inventing data that the domain does
not have, stop and check whether the types are being forced together. Here the
question "are terminals boundary conditions?" was answerable from the existing
code -- three of four realizations are not -- and was never asked.
