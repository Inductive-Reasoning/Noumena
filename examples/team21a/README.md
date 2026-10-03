# Example: TEAM Workshop Problem 21a (stray-field loss in a non-magnetic plate)

The linear members of the Compumag TEAM Problem 21 family (V.2009): a 10 mm
non-magnetic steel plate beside two racetrack coils carrying opposite 50 Hz
currents, as in a transformer's tank wall or tie-plate, with 0 to 3 slits
cut in the plate to break up its eddy currents (P21a-0 ... P21a-3). The
benchmark is the plate's eddy-current loss, and for P21a-2 the flux density
beside the plate.

## Problem

Dimensions in millimetres here; the mesh and configs are in metres. The plate
is centred on x = 0 and z = 0 is the mid-plane between the coils.

- **Plate:** non-magnetic steel (20Mn23Al), sigma = 1.3889e6 S/m, mu_r = 1;
  10 thick (|x| <= 5), 360 wide (y), 820 tall (z).
- **Slits:** 10 wide, 660 long (|z| <= 330), through the plate, centred at
  y = 0 (P21a-1), +-60 (P21a-2) or -90, 0, +90 (P21a-3).
- **Coils:** two square racetracks around the z axis, 270 x 270 outside (R45
  corners) and 200 x 200 inside (R10, concentric), each 217 tall, 24 apart
  (12 <= |z| <= 229), their near face 12 from the plate (x = 17 ... 287);
  300 turns each, 10 A rms (4243 ampere-turns peak), in opposite directions.
  Each is a stranded conductor driven through a cut (Cut1, Cut2) across its
  near side.
- **Boundary:** n x A = 0 on an air box 600 beyond the plate and coils.

The problem does not say which coil carries which direction; the configs
choose the one whose Bx has the tables' sign.

## Running

The meshes are not committed; generate them with Gmsh (one per variant):

```bash
cd examples/team21a
for n in 0 1 2 3; do gmsh -3 -format msh2 -setnumber slits $n team21a.geo -o team21a-$n.msh; done
for n in 0 1 2 3; do ../../build-mpi/mfem-electromag config-$n.json; done
pip install h5py   # compare.py reads the losses from the HDF5 results
python3 compare.py
```

Each variant has about 0.95 million complex unknowns at order 2 and takes
about 3.5 minutes on four threads with the iterative solver of the MPI
build. The plate loss is printed after each solve and stored in
`results-<n>/results.h5`; the Bx probes are written to
`results-<n>/probes/`. `measured.csv` holds the problem's measured values and
its authors' own (Ar-V-Ar) calculation.

The on-demand test `mfem_tests "[team21a]"` generates the meshes with Gmsh
(found by CMake; the test skips without it), solves all four variants and
checks them against the bounds below.

## Results

Plate loss at the rated 10 A rms:

| Model | Computed [W] | Measured [W] | Difference | Problem's calculation [W] |
|---|---|---|---|---|
| P21a-0 | 9.39 | 9.17 | +2.4% | 9.31 |
| P21a-1 | 3.37 | 3.40 | -0.9% | 3.34 |
| P21a-2 | 1.70 | 1.68 | +1.4% | 1.66 |
| P21a-3 | 1.01 | 1.25 | -19% | 1.14 |

Bx of P21a-2 along z (14 points, rms) agrees with the measurements to 1.6%
on the coil side (x = +5.76 mm) and 1.3% on the far side (x = -5.76 mm),
as RMS differences relative to the largest measured value, and with the
problem's calculation equally well.

The losses are converged: refining the plate to 7 mm and the coils to 15 mm
(1.7-1.9 million unknowns) changes them by at most 0.2%. P21a-3 is the
exception to the agreement, and not for want of resolution: the refined mesh
gives the same 1.01 W, and so does refining the plate alone to 5 mm (two
elements through its thickness, 16 across each strip; 1.012 W against
1.013 W).

The problem's authors solved P21a-3 with two formulations and got 1.14 W
(Ar-V-Ar) and 1.15 W (T-Omega-Omega) (Cheng et al., IEEE Trans. Magn.
40(2), 2004, Table II). Their other three models agree with this solver to
within 1-2%, but this one is 11% below theirs, and the reason is not known.
The measurement is above both calculations: by 9% over theirs and 19% over
this one, where the other three agree to 2.4%. It also breaks the trend of
the others: going from two slits to three, the measured loss falls to 0.74
of its value, against 0.69 in their calculation and 0.59 here (from one slit
to two, measurement and both calculations give 0.49-0.51).

The three-slit loss is sensitive to the slit geometry. Moving the outer slits
5 mm outward (to y = +-95, which is how the drawing's "90 90 90" chain would
read if it were dimensioned to slit edges) raises it by 7%, to 1.09 W.
Shortening every slit by 30 mm at each end raises it by 14%, to 1.15 W. The
slit positions here are those of the dimensioned drawing.

As in TEAM 7, the solver reports that the divergence-free projection removed
4.7% of each coil's current density: the uniform stranded current along a
solved path is not exactly current-conserving in a racetrack's corners (see
`docs/math_formulation.md`), an open issue of the stranded-path model. Its
effect is small: with the exact winding current (straight sides and circular
corners, as an experiment) the four losses rise by 0.3-0.4%, the P21a-3 gap
included.
