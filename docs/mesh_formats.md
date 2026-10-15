# Mesh formats

Which mesh files Noumena reads, where the names a configuration can refer to
come from, and how to produce meshes with Netgen from an application that links
it.

## Formats

Noumena reads every format MFEM reads, plus Netgen's native `.vol`, and it
reads names that MFEM's own Gmsh reader leaves out:

| Format | Written by | Names | Notes |
|---|---|---|---|
| Gmsh `.msh` 2.2 / 4.1 | Gmsh | `$PhysicalNames` | Names are read by Noumena; MFEM's reader skips them |
| Netgen `.vol` | Netgen `Mesh.Save()` | `materials`, `bcnames` | Read by Noumena (`src/io/netgen_vol.hpp`); 2D and 3D |
| MFEM v1.3+ | MFEM | `attribute_sets` block | |
| MFEM v1.0, `areamesh2`, Netgen neutral, VTK | various | none | Attributes only |

The format is recognized from the file's content, not its extension.

## Entity groups from the mesh file

An entity group is a named set of mesh attributes (see
[`entity_groups`](config_reference.md#entity_groups)). The names a mesh file
carries are entity groups too: domain names become domain groups, boundary
names boundary groups. A configuration can refer to them directly, and its own
`entity_groups` can combine them:

```json
"entity_groups": [
  {"name": "FarField", "dim": 1, "entity_groups": ["outer", "symmetry"]}
],
"boundary_conditions": [
  {"name": "Ground", "entity_group": "outer", "type": "dirichlet", "value": 0.0}
]
```

Names are resolved when the mesh is loaded, and every group becomes a list of
attribute numbers before the solvers see it. One name may cover several
attributes: Netgen numbers each face or edge of the geometry separately, so
five faces named `outer` are five attributes in one group. A name defined
nowhere is an error that lists the groups that exist, which keeps a
configuration portable between formats: on a mesh without names it fails
loudly instead of selecting nothing.

## Element orientation

Noumena repairs element orientation the way MFEM's own mesh constructors do: an
element whose vertices are listed with the opposite handedness is reordered,
which is exact. MFEM can reorder triangles, quadrilaterals, tetrahedra and
pyramids. An inverted hexahedron or prism it cannot repair, and the mesh is
rejected.

Netgen lists the vertices of every volume element with the opposite handedness
to MFEM, for prisms as well as tetrahedra. The `.vol` reader converts Netgen's
order to MFEM's itself, so Netgen's prism boundary layers load too.

## Netgen from an application

Netgen is LGPL, so an application may link it dynamically, mesh in-process, and
hand the result to Noumena as a file. Noumena itself does not link Netgen.

1. **Name every domain and every boundary** in the geometry: `mat()` on
   solids, `.name` on faces with the OpenCASCADE geometry; `SetMaterial()` and
   `bc=` in 2D. Netgen calls an unnamed face or edge `"default"`, which Noumena
   does not treat as a name, so an unnamed boundary cannot be referred to by
   name.
2. **Save in Netgen's native format:** `mesh.Save("model.vol")` (or
   `Ng_SaveMesh` from nglib). This is the only Netgen output that keeps both the
   boundary elements and the names. nglib's C interface (`nglib.h`) can save a
   mesh but has no calls for naming, so the names have to be set through
   Netgen's C++ or Python geometry classes.
3. **Refer to the names in the configuration.** The application never needs
   Netgen's numbers, and needs no `entity_groups` section unless it combines
   names.

```python
from netgen.occ import Box, Cylinder, Pnt, Z, X, OCCGeometry, Glue

air = Box(Pnt(-1, -1, -1), Pnt(1, 1, 1)); air.mat("air")
air.faces.name = "outer"
air.faces.Min(X).name = "symmetry"
bar = Cylinder(Pnt(0, 0, -0.2), Z, r=0.1, h=0.4); bar.mat("bar")
bar.faces.name = "bar_side"
bar.faces.Min(Z).name = "terminal_in"
bar.faces.Max(Z).name = "terminal_out"

mesh = OCCGeometry(Glue([air - bar, bar])).GenerateMesh(maxh=0.3)
mesh.Save("bar.vol")
```

```json
"regions": [
  {"entity_group": "air", "material": "air"},
  {"entity_group": "bar", "material": "bar"}
],
"terminals": [
  {"name": "in",  "quantity": "voltage", "entity_group": "terminal_in"},
  {"name": "out", "quantity": "voltage", "entity_group": "terminal_out"}
],
"boundary_conditions": [
  {"name": "outer", "entity_group": "outer", "type": "dirichlet", "value": 0.0}
]
```

Coordinates are taken as they are written, so the geometry must be in meters.

### Where the `.vol` attributes come from

Following Netgen's own reader (`Mesh::Load`):

- **3D:** a volume element's material number is its attribute, named by
  `materials`. A boundary face's boundary condition number is its attribute,
  named by `bcnames`.
- **2D:** the domain elements are Netgen's "surface elements"; each belongs to
  the face descriptor matching its region, and that descriptor's 1-based index
  is its attribute, named by `materials`. The boundary is the `edgesegmentsgi3`
  section; each segment's 0-based edge index plus one is its attribute, named by
  `bcnames`.

Interior faces and edges (between two domains) are boundary elements too, as in
Gmsh meshes, so a terminal or interface can be named and selected.

### Netgen's other outputs

Measured with Netgen 6.2; none of them is a substitute for `.vol`:

- **Gmsh2 export** writes no `$PhysicalNames`, drops the boundary elements of a
  2D mesh, and writes a version line (`2.000000`) MFEM rejects.
- **Neutral export** (3D) carries no names and no format line, so MFEM does not
  recognize it without one being added.
- **`areamesh2`** (MFEM's 2D Netgen format) is not written by Netgen at all.

### Not supported

- Second-order (curved) Netgen elements; the reader rejects them with a
  message. Mesh with first-order elements.
- 2D `.vol` files from Netgen older than 6.2 (`edgesegmentsgi2` boundary
  sections); save them again with 6.2 or newer.

## Licensing

Gmsh is GPL and Noumena MIT, so Gmsh is used only as a separately installed
executable (`examples/generate_meshes.sh`), never linked. Netgen is LGPL, which
permits an application to link it dynamically with the relink freedom
preserved; static linking does not satisfy the LGPL.

Permissively licensed meshers surveyed as alternatives:

| Mesher | License | Fit |
|---|---|---|
| TetGen | AGPL | no |
| CGAL mesh generation | GPL / commercial | no |
| Triangle (Shewchuk) | non-commercial use only | no |
| MMG | LGPL | same category as Netgen |
| Salome SMESH | LGPL | same category as Netgen |
| fTetWild | MPL-2.0 | permissive, but 3D tetrahedra only |
| poly2tri | BSD | 2D constrained Delaunay only; no sizing or grading |

No permissively licensed general-purpose mesher covers the 2D axisymmetric and
3D cases this project needs.

## Related

- `docs/open_boundary.md` - far-field truncation in the same current-loop model.
