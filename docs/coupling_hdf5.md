# Run results in HDF5

Include `"output": {"hdf5": {}}` to enable one archive for the entire run,
at `results/results.h5` relative to the config file by default. Set
`output.directory` and `output.hdf5.file` to choose the destination. Missing
directories are created; each run replaces the archive, rather than appending.

Field analyses store every scenario. Coupling analyses store matrices, and
store unit-excitation scenario fields only when `output.export_fields_for_coupling_matrix` is
true. This switch applies to ParaView and Gmsh too, and defaults to false.
Omitting `output.hdf5` disables archive output but not console matrix tables.

## Schema version 2

The root has integer `schema_version = 2` and string attributes `physics_type`,
`geometry_type`, and `analysis_type`, with the same values as the configuration.
All archived fields reference the final solved mesh. AMR intermediate results
are replaced on each mesh pass, not retained as extra scenarios.

### Mesh

`/mesh/mfem` is a scalar string containing the complete MFEM mesh serialization
at round-trip floating-point precision. It includes curved/high-order geometry
when present. Read this into `mfem::Mesh` to reconstruct the native FE spaces.
The original input mesh is not required, including after AMR.

For non-MFEM readers, `/mesh/vertices` is a double array `[vertex, component]`.
`/mesh/elements` and `/mesh/boundary` each contain one-dimensional integer arrays:

- `offsets`: start offsets into connectivity, with a final sentinel.
- `vertices`: concatenated zero-based corner vertex indices.
- `attributes`: material/domain or boundary attribute per element.
- `geometry`: MFEM geometry codes (point 0, segment 1, triangle 2, square 3,
  tetrahedron 4, cube 5, prism 6, pyramid 7).

Mesh attributes are `dimension`, `space_dimension`, and `coordinate_units = "m"`.
Corner connectivity alone is not a high-order geometry description. A curved
mesh (for example one read from a second-order Gmsh file) also has
`/mesh/nodes`, laid out like a field group: `kind = "geometry"` and a `values`
dataset holding the node coordinates as a `space_dimension`-component field,
with the same attributes as field values (including `space`, below). The
element's geometry is then `x(xi) = sum_k nodes[dof_k] * phi_k(xi)`, the same
expansion used for fields. `/mesh/nodes` is absent for straight-sided meshes,
whose elements are the affine maps of their corner vertices.

### Scenario fields

`/scenarios/scenario_000000`, etc., are stable IDs in solve order. Scenario and
terminal names are metadata, not path components, so punctuation and repeated
display names cannot collide. Each scenario group has:

- `name`: original scenario name.
- `mesh`: `/mesh`.
- `frequency_hz`: numeric frequency for MQS scenarios.
- `driven_terminal`: the unit-excited terminal for coupling scenarios.
- `excitations/terminal_names` and `excitations/values`: aligned drive arrays.
- `fields/<field>/values`: a one-dimensional double array of FE coefficients.

Each field group has `kind` (`primary`, `scalar`, or `vector`). Each `values`
dataset has `finite_element_collection`, `vector_dimension`, and `ordering`
attributes. Ordering is MFEM's `byNODES = 0` or `byVDIM = 1`. Rebuild the named
collection on `/mesh/mfem`, create its space with that dimension and ordering,
and load the array into a `GridFunction`. These are native DOFs, not values on
the corner-vertex array. Derived fields are projected into L2 order
`max(0, solution_order - 1)`; primary fields retain their native space.

MQS stores `A_Real`, `A_Imag`, `B_Real`, `B_Imag`, `B_Magnitude`, and `P_Loss`.
Phasors use the peak-amplitude convention; `P_Loss` is time-averaged loss density.

### Finite element spaces

Each field's `values` dataset also has a string `space` attribute naming a
group under `/spaces`, one per finite element collection (for example
`/spaces/H1_2D_P2`). Spaces depend only on the collection and the mesh, so
every scenario shares them. They let readers without MFEM evaluate a field
inside any element, at any order:

- `element_dof_offsets`: one-dimensional integer array, one start offset per
  mesh element plus a final sentinel, in `/mesh/elements` order.
- `element_dofs`: the concatenated scalar DOF indices of each element, in the
  element's local DOF order. A negative entry `d` means DOF `-1 - d` with its
  sign flipped (MFEM's convention; H1 and L2 never use it). For a vector field,
  component `c` of DOF `d` is at `c * ndofs + d` when `ordering` is `byNODES`,
  or at `d * vector_dimension + c` when it is `byVDIM`.
- `ndofs` attribute: the number of scalar DOFs.
- `reference_nodes/<geometry>`: double array `[local DOF, dimension]` giving
  where each local DOF sits on the reference element, keyed by MFEM geometry
  code (triangle `2`), with an integer `order` attribute.

The reference triangle has vertices `(0,0)`, `(1,0)`, `(0,1)`, matching the
element's vertex order in `/mesh/elements`. Reference nodes are written only
for nodal, value-mapped elements (MFEM's default H1 and L2 bases): a DOF is the
field's value at its node, so the element's shape functions are the Lagrange
polynomials on those nodes, and the field at reference point `x` is
`sum_k sign_k * values[dof_k] * phi_k(x)`. The nodes are exported rather than
named because they depend on the basis type (Gauss-Lobatto for H1,
Gauss-Legendre for L2 by default).

### Coupling matrices

| Physics | Matrix groups under `/coupling` |
|---------|--------------------------------|
| Electrostatics | `Capacitance` |
| Magnetostatics | `Inductance` |
| Magnetoquasistatics | `Inductance`, `Resistance` |

The `/coupling` group has string attributes `physics_type` and `geometry_type`,
using the same values as the configuration.

`/coupling/terminal_names` is a one-dimensional array of variable-length UTF-8
strings. Its order is the solver's name-sorted terminal order and labels **both**
matrix axes. Rows are measured terminals; columns are driven terminals.
Terminal names are data, not HDF5 paths, so punctuation in names is preserved.

Each matrix group contains a `values` dataset of 64-bit floating-point numbers.
For static solvers its shape is `[N, N]`, where `N` is the terminal count:

- `/coupling/Capacitance/values`
- `/coupling/Inductance/values`

The `values` dataset has a string `units` attribute. Units are `F`, `H`, or `Ohm`
for axisymmetric models and `F/m`, `H/m`, or `Ohm/m` for planar models. Values
are stored without rounding to console precision.

## MQS frequency axis

`/coupling/frequency_hz` is a one-dimensional array of 64-bit floating-point
frequencies, with `units = "Hz"`. It contains unique, positive frequencies in
ascending numerical order. Duplicate frequencies across scenarios or within a
sweep are solved and stored once. Equality is exact numeric equality, not a
formatted frequency-string comparison. Scenario names are not dictionary keys.

Both `/coupling/Inductance/values` and `/coupling/Resistance/values` have shape
`[K, N, N]`, where `K` is the frequency count. The first dimension always exists,
even for a single-frequency solve. Slice `i` in each dataset belongs to
`frequency_hz[i]`. These are the extracted L and R matrices; an impedance matrix
can be constructed as `Z(f) = R(f) + j * 2 * pi * f * L(f)`.

## Reading into a C# dictionary

Use an HDF5 library to read `frequency_hz` as `double[]`, `terminal_names` as
`string[]`, and each `values` dataset as either a rectangular `double[,,]` or a
flat `double[]`, depending on the library. Check `schema_version` and the dataset
shapes before consuming the arrays.

For flat arrays, HDF5's last dimension varies fastest. The element for frequency
index `i`, row `r`, and column `c` is at `(i * N + r) * N + c`. Do not transpose
based on MFEM's internal column-major storage: the writer has already converted
it to the documented order.

After reading flat arrays named `frequencies`, `inductanceValues`, and
`resistanceValues`, and string array `terminalNames`, the dictionary assembly is
independent of the HDF5 library:

```csharp
var matrices = new Dictionary<double, (double[,] Inductance, double[,] Resistance)>();
int n = terminalNames.Length;
for (int i = 0; i < frequencies.Length; ++i)
{
	var inductance = new double[n, n];
	var resistance = new double[n, n];
	for (int r = 0; r < n; ++r)
	{
		for (int c = 0; c < n; ++c)
		{
			int offset = (i * n + r) * n + c;
			inductance[r, c] = inductanceValues[offset];
			resistance[r, c] = resistanceValues[offset];
		}
	}
	matrices.Add(frequencies[i], (inductance, resistance));
}
```

Use the stored frequency values as keys, rather than reconstructing them from
rounded display labels. Iterate the frequency array when sorted order matters;
do not rely on dictionary enumeration order.
