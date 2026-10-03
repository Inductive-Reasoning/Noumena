#!/usr/bin/env python3
"""Scan the TEAM 15 coil along the slot and record the change in its impedance.

Usage: scan.py <problem 1|2> [--step MM] [--solver PATH] [--gmsh PATH]

For each coil-centre position x (those of the measurements, or every --step
millimetres over their range) this meshes team15.geo with the coil at x and
solves config-<problem>.json on it twice, with the slot empty (flawed) and
filled with the plate's aluminium (unflawed). The two solves share the mesh, so
their difference, the impedance change the problem asks for, carries no
meshing noise. Each row of results-<problem>/dz.csv is one position:

  x_mm, L0_H, R0_ohm (unflawed), dL_uH, dR_ohm (flawed - unflawed)

for the whole coil: the model is a half (y >= 0) driven with one ampere-turn,
so its inductance and resistance are doubled and scaled by the number of
turns squared. Positions already in dz.csv are skipped, so an interrupted scan
resumes. Reading the results needs h5py (pip install h5py).
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TURNS = {1: 3790, 2: 408}


def measured_positions(problem):
    with open(HERE / "measured.csv") as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        return [float(r["x_mm"]) for r in rows if int(r["problem"]) == problem]


def coil_impedance(results):
    """(L, R) of the half model per ampere-turn squared."""
    import h5py
    with h5py.File(results / "results.h5", "r") as f:
        return float(f["coupling/Inductance/values"][0, 0, 0]), float(f["coupling/Resistance/values"][0, 0, 0])


def solve(args, work, x, slot_material):
    config = json.loads((HERE / f"config-{args.problem}.json").read_text())
    config["simulation"]["mesh"] = "team15.msh"
    config["output"] = {"directory": f"results-{slot_material}", "hdf5": {}}
    config["terminals"][0]["direction"]["origin"] = [x, 0.0, 0.0]
    next(r for r in config["regions"] if r["name"] == "Slot")["material"] = slot_material
    path = work / f"config-{slot_material}.json"
    path.write_text(json.dumps(config, indent=2))
    with open(work / f"solve-{slot_material}.log", "w") as log:
        subprocess.run([args.solver, str(path)], cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
    return coil_impedance(work / f"results-{slot_material}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("problem", type=int, choices=(1, 2))
    parser.add_argument("--step", type=float, help="position step in mm (default: the measured positions)")
    parser.add_argument("--solver", default=str(HERE.parent.parent / "build-mpi" / "mfem-electromag"))
    parser.add_argument("--gmsh", default="gmsh")
    args = parser.parse_args()

    positions = measured_positions(args.problem)
    if args.step:
        n = round(positions[-1] / args.step)
        positions = [i * args.step for i in range(n + 1)]
    out = HERE / f"results-{args.problem}"
    out.mkdir(exist_ok=True)
    table = out / "dz.csv"
    done = set()
    if table.exists():
        with open(table) as f:
            done = {float(r["x_mm"]) for r in csv.DictReader(f)}
    else:
        table.write_text("x_mm,L0_H,R0_ohm,dL_uH,dR_ohm\n")
    scale = 2 * TURNS[args.problem] ** 2

    for x_mm in positions:
        if x_mm in done:
            continue
        work = out / f"x{x_mm:g}"
        work.mkdir(exist_ok=True)
        x = x_mm / 1000
        subprocess.run([args.gmsh, "-3", "-format", "msh2", "-setnumber", "problem", str(args.problem),
                        "-setnumber", "X", repr(x), str(HERE / "team15.geo"), "-o", str(work / "team15.msh")],
                       stdout=subprocess.DEVNULL, check=True)
        L0, R0 = solve(args, work, x, "Aluminium")
        L1, R1 = solve(args, work, x, "Air")
        row = [x_mm, scale * L0, scale * R0, 1e6 * scale * (L1 - L0), scale * (R1 - R0)]
        with open(table, "a") as f:
            f.write(",".join(f"{v:.9g}" for v in row) + "\n")
        print(f"x = {x_mm:5.1f} mm: dL = {row[3]:8.2f} uH, dR = {row[4]:7.3f} ohm", flush=True)
        (work / "team15.msh").unlink()


if __name__ == "__main__":
    sys.exit(main())
