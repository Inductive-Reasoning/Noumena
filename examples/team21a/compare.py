#!/usr/bin/env python3
"""Compare TEAM 21a runs with the measurements in measured.csv.

Usage: compare.py [results_root]   (default: this directory)

Reads, for each P21a-n with a results-n directory under results_root, the
plate loss from results-n/results.h5 (needs h5py: pip install h5py) and, for
P21a-2, Bx along z from the probe CSVs. The tables give rms values at the
rated 10 A rms; the solver's phasors are peak values of the same excitation
(4243 ampere-turns peak), so computed fields are divided by sqrt(2). Losses
are time averages and compare directly.
"""

import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def measured():
    with open(HERE / "measured.csv") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def plate_loss(results):
    import h5py
    with h5py.File(results / "results.h5", "r") as f:
        group = f["scenarios/scenario_000000/losses"]
        names = [n.decode() if isinstance(n, bytes) else n for n in group["region_names"][()]]
        return float(group["power_w"][()][names.index("Plate")])


def bx_rms(results, probe):
    path = results / "probes" / f"scenario_000000_50Hz_{probe}.csv"
    with open(path) as f:
        return {round(1000 * float(r["z"]), 3): 1e4 * float(r["B_Real_x"]) / math.sqrt(2)
                for r in csv.DictReader(f)}


def compare(root):
    """{key: relative error}, and the printable report."""
    rows, errors, report = measured(), {}, []
    report.append("Plate loss [W]          computed   measured   (published calc.)")
    for row in (r for r in rows if r["quantity"] == "loss"):
        n = row["model"]
        results = root / f"results-{n}"
        if not results.exists():
            continue
        got, want = plate_loss(results), float(row["measured_coil_side"])
        errors[f"loss {n}"] = got / want - 1
        report.append(f"  P21a-{n}            {got:9.3f}  {want:9.3f}   ({float(row['calculated_coil_side']):.3f})"
                      f"   {100 * (got / want - 1):+5.1f}%")
    results = root / "results-2"
    if results.exists():
        near, far = bx_rms(results, "Bx_coil_side"), bx_rms(results, "Bx_far_side")
        report.append("\nP21a-2 Bx [1e-4 T rms]  coil side (x = +5.76 mm)    far side (x = -5.76 mm)")
        report.append("   z [mm]               computed  measured        computed  measured")
        sums = {"coil": [0.0, 0.0], "far": [0.0, 0.0]}
        for row in (r for r in rows if r["quantity"] == "Bx"):
            z = float(row["z_mm"])
            m1, m2 = float(row["measured_coil_side"]), float(row["measured_far_side"])
            report.append(f"  {z:6.0f}              {near[z]:9.2f} {m1:9.2f}       {far[z]:9.2f} {m2:9.2f}")
            for side, got, want in (("coil", near[z], m1), ("far", far[z], m2)):
                sums[side][0] += (got - want) ** 2
                sums[side][1] = max(sums[side][1], abs(want))
        count = sum(1 for r in rows if r["quantity"] == "Bx")
        for side, (sq, peak) in sums.items():
            errors[f"Bx {side}"] = math.sqrt(sq / count) / peak
        report.append(f"  RMS difference / max |measured|: coil side {100 * errors['Bx coil']:.1f}%,"
                      f" far side {100 * errors['Bx far']:.1f}%")
    return errors, "\n".join(report)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    print(compare(root)[1])


if __name__ == "__main__":
    main()
