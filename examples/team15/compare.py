#!/usr/bin/env python3
"""Compare TEAM 15 scans (scan.py) with the measurements in measured.csv.

Usage: compare.py [problem ...] [--root DIR]   (default: both problems, this directory)

Prints, for each problem with a results-<n>/dz.csv, the computed and measured
change in inductance and resistance at the measured positions the scan
covers, and the RMS difference of each relative to the largest measured
magnitude of that quantity. Also printed, as the problem asks, are the
magnitude and phase of dZ = dR + j omega dL.
"""

import argparse
import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
FREQUENCY = {1: 900.0, 2: 7000.0}


def read(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def compare(problem, root):
    """({quantity: relative RMS difference}, printable report), or None without a scan."""
    table = root / f"results-{problem}" / "dz.csv"
    if not table.exists():
        return None
    computed = {round(float(r["x_mm"]), 3): (float(r["dL_uH"]), float(r["dR_ohm"])) for r in read(table)}
    measured = [(float(r["x_mm"]), float(r["dL_uH"]), float(r["dR_ohm"]))
                for r in read(HERE / "measured.csv") if int(r["problem"]) == problem]
    omega = 2 * math.pi * FREQUENCY[problem]
    report = [f"Problem {problem} ({FREQUENCY[problem]:g} Hz)",
              "  x [mm]     dL [uH]           dR [ohm]          |dZ| [ohm]        arg dZ [deg]",
              "           computed measured  computed measured  computed measured  computed measured"]
    sums = {"dL": [0.0, 0.0], "dR": [0.0, 0.0]}
    count = 0
    for x, dl, dr in measured:
        if round(x, 3) not in computed:
            continue
        cl, cr = computed[round(x, 3)]
        cz, mz = complex(cr, omega * cl * 1e-6), complex(dr, omega * dl * 1e-6)
        report.append(f"  {x:5.1f}  {cl:9.2f} {dl:8.2f}  {cr:8.3f} {dr:8.3f}  {abs(cz):8.3f} {abs(mz):8.3f}"
                      f"  {math.degrees(math.atan2(cz.imag, cz.real)):8.1f} {math.degrees(math.atan2(mz.imag, mz.real)):8.1f}")
        for key, got, want in (("dL", cl, dl), ("dR", cr, dr)):
            sums[key][0] += (got - want) ** 2
            sums[key][1] = max(sums[key][1], abs(want))
        count += 1
    if count == 0:
        return {}, "\n".join(report)
    errors = {key: math.sqrt(sq / count) / peak for key, (sq, peak) in sums.items()}
    report.append(f"  RMS difference / max |measured| over {count} positions: "
                  f"dL {100 * errors['dL']:.1f}%, dR {100 * errors['dR']:.1f}%")
    return errors, "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("problems", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--root", type=Path, default=HERE)
    args = parser.parse_args()
    for problem in args.problems:
        result = compare(problem, args.root)
        if result is not None:
            print(result[1] + "\n")


if __name__ == "__main__":
    main()
