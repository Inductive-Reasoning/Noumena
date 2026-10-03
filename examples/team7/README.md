# Example: TEAM Workshop Problem 7 (asymmetrical conductor with a hole)

The Compumag TEAM benchmark 7: a thick aluminium plate with an eccentric hole,
under a racetrack coil carrying a sinusoidal current, solved with 3D
magnetoquasistatics and compared with the problem's measurements of the flux
density above the plate and the eddy currents on its surfaces.

## Problem

All dimensions in millimetres here; the mesh and config are in metres.

- **Plate:** aluminium, sigma = 3.526e7 S/m, 294 x 294 x 19 (0 <= x, y <= 294,
  0 <= z <= 19), with a through hole 18 <= x, y <= 126.
- **Coil:** a racetrack 25 wide and 100 tall (49 <= z <= 149), outline 200 x 200
  at 94 <= x <= 294, 0 <= y <= 200, corner radii 50 outside and 25 inside;
  2742 ampere-turns, peaking at omega t = 0.
- **Frequencies:** 50 and 200 Hz (skin depths 12 and 6 mm).
- **Boundary:** B . n = 0 on the problem's air box, -1353 <= x, y <= 1647,
  -300 <= z <= 449: a `dirichlet` (n x A = 0) wall.

The coil is a stranded conductor driven through a `cut` (the group `Cut`, a
cross-section of its straight side along y = 0, crossed in +x). It is
modelled as one turn of 2742 A, the same field as 2742 turns of 1 A. The hole
needs no special treatment: the plate is simply a conductor with a hole.

## Running

```bash
./build-mpi/mfem-electromag examples/team7/config.json   # MPI build: iterative solver
python3 examples/team7/compare.py                         # computed vs measured
```

The model has 0.9 million complex unknowns at order 2; it needs the iterative
(AMS-preconditioned) solver of the MPI build and takes about 15 minutes and a
few GB. The probes write `results/probes/*.csv`: Bz on A1-B1 and A2-B2
(z = 34 mm, y = 72 and 144 mm) and Jy on the plate's top (z = 19 mm) and
bottom (z = 0) surfaces along y = 72 mm, sampled from inside the plate.
`measured.csv` holds the problem's measured tables.

The on-demand test `mfem_tests "[team7]"` runs the same model and checks
every column of the comparison against the bounds below.

Regenerate the mesh with `gmsh -3 -format msh2 team7.geo -o team7.msh`; the
sizes in the plate and coil are `h_plate` (6 mm) and `h_coil` (12 mm), and
`-setnumber h_plate 0.004` and so on refines them.

## Results

RMS difference over each column, as a fraction of the column's largest
measured magnitude, at omega t = 0 / 90 degrees:

| Table | 50 Hz | 200 Hz |
|---|---|---|
| A1-B1 Bz | 1.4% / 2.2% | 1.5% / 41% |
| A2-B2 Bz | 1.6% / 2.6% | 1.6% / 59% |
| Jy, top surface (table "A4-B4") | 13% / 17% | 14% / 22% |
| Jy, bottom surface (table "A3-B3") | 56% / 51% | 13% / 16% |

The solution is converged: with the plate at 4 mm and the coil at 8 mm (2.3
million unknowns) no probe value moves by more than 2% of its column's peak.
What remains is the comparison with the measurements:

- **Bz** agrees to 1.4-2.6% except the small quadrature part at 200 Hz, which
  is computed about 2.5 G above the measurements over the coil, where the
  in-phase field is about 50 G. One phase rotation of the measurements by 2.9
  degrees (none is needed at 50 Hz) halves the complex error at 200 Hz, from
  3.2% to 1.6% of the peak: the difference looks like a phase-reference error
  of the measurement at 200 Hz.
- **Jy:** the tables labelled A3-B3 (z = 19 mm, the surface facing the coil)
  and A4-B4 (z = 0) are compared with the opposite surfaces. The table
  labelled z = 0 carries the larger currents, as the coil-facing surface must,
  and it matches the computed top surface to 13-22%; as labelled, the
  differences are 30-77%. NGSolve's TEAM 7 solution makes the same exchange
  ([TEAM-problems](https://github.com/NGSolve/TEAM-problems), `TEAM-7/team7.ipynb`).
  The bottom surface at 50 Hz still disagrees by about half its peak.
- The solver warns that the divergence-free projection removed 2.6% of the
  coil's current density; see the next section.

## The coil's current (an open issue)

The coil is `stranded`: its current density has magnitude `I / A_cs`
everywhere, along the direction `w / |w|` of the path `w = -grad v` solved on
the coil through the cut. `w` is divergence-free, but its direction is not
in general:

    div(w / |w|) = -(w / |w|) . grad ln|w|,

which vanishes only where `|w|` is constant along each field line, as on an
azimuthal path or a straight bar. In a racetrack the solved path is
potential flow: `|w|` grows like `1/r` towards the inside of each corner and
is uniform across the straight sides, so it changes along the field lines
where the two meet. The winding's actual current -- uniform, straight along
the sides and on circular arcs around the corners -- is divergence-free but
is not the direction of any such potential. The projection then removes the
non-conserving part, and the source used is the projected field, not the
winding's.

Two measurements separate this from the geometry, with the projection
`ProjectWithin` of the order-2 load and the removed fraction in the L2 norm:

| Coil current | straight tetrahedra | second-order (curved) tetrahedra |
|---|---|---|
| The solver's: uniform along the solved path | 2.63% | 2.67% |
| The winding's: uniform along straights and arcs (exact) | 0.73% | 0.024% |

The second-order mesh (`gmsh -order 2`) is loaded and used throughout: its
coil volume integrates to the exact rounded-racetrack volume within 3.5e-6
(-3.1e-4 straight-sided), its corner faces lie on the arcs within 0.6 um
(0.48 mm), and with the exact winding current the projection's removal falls
from 0.73% to 0.024%. With the solver's own direction it stays at 2.65%, as
it does on refinement (2.64% at 4 mm), so the 2.6% is the stranded path
model's, not the mesh's. On the curved mesh no probe value moves by more than
0.2% of its column's peak, so the example keeps the straight-sided mesh.

The effect on the results is measured by solving with the exact winding
current in place of the solver's (an experiment, not an option of the
solver): every probe value moves by 0.12-0.17% of its column's peak, almost
uniformly, and the plate loss by 0.3%. The projection mostly reduces the
current's amplitude a little rather than misplacing it, and none of the
differences from the measurements above comes from it. The projected source
is still not the winding's current. A better construction of
stranded currents on solved paths, one that stays uniform and conserves
current without a winding-specific direction, is still to be found.
