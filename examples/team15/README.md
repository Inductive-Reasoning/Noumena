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
problem 1 (0, 9 and 17 mm) and checks them against the measurements to
100 uH in dL and 0.25 ohm in dR.

## Results

Problem 1, scanned every millimetre from 0 to 22 mm with `--step 1`. The run
has about 0.9 million complex unknowns per position at order 2 and takes about
8.5 minutes per position (two solves) on four threads with the iterative
solver of the MPI build, about 3.3 hours for the whole scan.

| x [mm] | dL computed [uH] | dL measured [uH] | dR computed [ohm] | dR measured [ohm] | arg dZ computed | arg dZ measured |
|---|---|---|---|---|---|---|
| 0 | 437 | 390 | 0.22 | 0.2 | 85.0 | 84.8 |
| 2 | 678 | 630 | -0.10 | -0.1 | 91.5 | 91.6 |
| 4 | 1264 | 1220 | -0.87 | -0.8 | 96.9 | 96.6 |
| 6 | 1893 | 1840 | -1.65 | -1.5 | 98.8 | 98.2 |
| 8 | 2284 | 2220 | -2.00 | -1.85 | 98.8 | 98.4 |
| 9 | 2343 | 2270 | -1.96 | -1.85 | 98.4 | 98.2 |
| 10 | 2304 | 2230 | -1.79 | -1.65 | 97.8 | 97.5 |
| 12 | 1963 | 1880 | -1.10 | -1.1 | 95.6 | 95.9 |
| 14 | 1388 | 1310 | -0.17 | -0.2 | 91.2 | 91.5 |
| 16 | 797 | 720 | 0.52 | 0.35 | 83.5 | 85.1 |
| 18 | 370 | 310 | 0.69 | 0.55 | 71.9 | 72.6 |
| 20 | 145 | 110 | 0.50 | 0.3 | 58.6 | 64.3 |
| 22 | 53 | 30 | 0.28 | 0.1 | 47.0 | 59.5 |

Over all 23 positions the RMS difference is 2.7% of the largest measured dL
and 6.5% of the largest measured dR. The shape of the scan, the peak of dL at
9 mm, the sign changes of dR and the phase of dZ (within 1 degree to
14 mm and 2 degrees to 19 mm) all follow the measurements. dL is high
throughout, by 23 to 80 uH (up to 3.5% of its 2270 uH peak; the measurements'
error is +-40 uH), and in the tail beyond 15 mm, where the coil has passed
the slot's end and dZ is small, dR is high by 0.13-0.2 ohm (the
measurements' error is +-0.1 ohm).

The isolated coil (plate and slot made air) has 225.9 mH against the
problem's measured 221.8 mH, 1.9% higher: the model's coil, with its nominal
dimensions and a uniform current density over its section, couples a little
more strongly than the real winding, which is consistent with a dL a few
percent high. Over the unflawed plate the coil has 178.63 mH and 120.67 ohm.

The solution is converged: at x = 18 mm, refining the slot to 0.15 mm
(1.35 million unknowns) or the plate to 0.7 mm (2.0 million) changes dL by at
most 0.2% and dR by 0.1%. The unflawed impedance differs between the
positions' meshes by at most 9 uH in L (5e-5 of it), the meshing noise that
taking dZ on one mesh avoids. Unlike TEAM 7 and 21a, the coil's current is
divergence-free as given (an azimuthal current in a coil of revolution), so
the divergence-free projection removes no more than discretization noise.

Problem 2 (7 kHz) is set up (`./scan.py 2`) but has not been run: its finer
plate mesh for the 1.09 mm skin depth makes it several times larger.
