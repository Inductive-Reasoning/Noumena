# Example: TEAM Workshop Problem 15 (rectangular slot in a thick plate)

An eddy-current nondestructive-evaluation benchmark: a circular air-cored coil
is scanned along a surface-breaking slot in a thick aluminium-alloy plate, and
the quantity of interest is the change in the coil's impedance that the slot
causes, dZ = dR + j omega dL, against the coil's position. Problem 1 is at
900 Hz (skin depth 3.04 mm, comparable to the slot's 5 mm depth), problem 2
at 7 kHz with a larger coil (1.09 mm, near the thin-skin limit).

## Problem

Dimensions in millimetres here; the mesh and configs are in metres. The plate's
top face is z = 0, the slot is centred on the origin along x, and the coil's
axis is vertical through (x, 0), x being the scanned position.

- **Plate:** sigma = 3.06e7 S/m, mu_r = 1, 12.22 thick (modelled 200 x 200 in
  the plane, far beyond the coil's reach).
- **Slot:** 12.6 long (|x| <= 6.3), 5 deep, 0.28 wide.
- **Coil (problem 1):** inner radius 6.15, outer 12.4, length 6.15, 3790
  turns, lift-off 0.88; isolated inductance 221.8 mH.
- **Coil (problem 2):** inner radius 9.34, outer 18.4, length 9.0, 408 turns,
  lift-off 2.03; isolated inductance 3.96 mH.
- **Boundary:** n x A = 0 on an air box 150 from the origin.

The model is the y >= 0 half, with the plane of symmetry y = 0 part of the
n x A = 0 boundary: the coil is an `azimuthal` stranded conductor whose half
ends on that plane (see "azimuthal" in `docs/config_reference.md`). The slot
is a volume of its own, so the flawed and unflawed plates are one mesh with the
slot's material changed. dZ, a few tenths of a percent of the coil's
impedance, is the difference of two solves on that mesh, free of the meshing
noise two separately meshed plates would add.

## Running

```bash
cd examples/team15
./scan.py 1 --step 1     # problem 1, every mm from 0 to 22 mm
./compare.py 1
```

`scan.py` meshes the coil at each position with Gmsh (the meshes are not
committed), solves `config-<n>.json` with the slot filled and empty, and
appends the whole coil's impedance change to `results-<n>/dz.csv`; without
`--step` it takes the measured positions. It resumes an interrupted scan.
`compare.py` prints the computed and measured dL, dR, |dZ| and arg dZ.
Both read the solver's HDF5 output, which needs h5py (`pip install h5py`).

The on-demand test `mfem_tests "[team15]"` solves three positions of
problem 1 and checks them against the measurements.

## Results
