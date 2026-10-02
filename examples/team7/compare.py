#!/usr/bin/env python3
"""Compare a TEAM 7 run's probes with the measurements in measured.csv.

Usage: compare.py [results_directory]   (default: results, beside this script)

The solver's phasors X are peak values with time dependence Re(X e^{j w t}),
so the instantaneous value at w t = 0 is Re X and at 90 degrees -Im X. For
each line and column this prints computed against measured, and the RMS
difference as a fraction of the column's largest measured magnitude.
"""

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# line -> (field component in the probe CSV, factor to the table's unit)
QUANTITIES = {
    "A1B1": ("B", "z", 1e4),   # T -> gauss
    "A2B2": ("B", "z", 1e4),
    "A3B3": ("J", "y", 1e-6),  # A/m^2 -> 1e6 A/m^2
    "A4B4": ("J", "y", 1e-6),
}
SCENARIOS = {"f50": "scenario_000000_50Hz", "f200": "scenario_000001_200Hz"}


def measured():
    rows = {}
    with open(HERE / "measured.csv") as f:
        for row in csv.DictReader(line for line in f if not line.startswith("#")):
            rows.setdefault(row["line"], []).append(row)
    return rows


def computed(results, scenario, line):
    field, component, factor = QUANTITIES[line]
    path = results / "probes" / f"{SCENARIOS[scenario]}_{line}.csv"
    with open(path) as f:
        samples = list(csv.DictReader(f))
    real = [factor * float(s[f"{field}_Real_{component}"]) for s in samples]
    imag = [factor * float(s[f"{field}_Imag_{component}"]) for s in samples]
    x_mm = [round(1000 * float(s["x"]), 3) for s in samples]
    return {x: (re, -im) for x, re, im in zip(x_mm, real, imag)}


def main():
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    worst = 0.0
    for line, rows in measured().items():
        field, component, _ = QUANTITIES[line]
        print(f"\n{line}: {field}{component}")
        header = "  x [mm]"
        for scenario in SCENARIOS:
            for phase in ("0", "90"):
                header += f"  {scenario[1:]} Hz {phase:>2}: computed / measured"
        print(header)
        values = {s: computed(results, s, line) for s in SCENARIOS}
        errors = {}
        for row in rows:
            x = float(row["x_mm"])
            text = f"  {x:6.0f}"
            for scenario in SCENARIOS:
                for index, phase in enumerate(("wt0", "wt90")):
                    got = values[scenario][x][index]
                    want = float(row[f"{scenario}_{phase}"])
                    errors.setdefault((scenario, phase), []).append((got - want, want))
                    text += f"  {got:16.3f} / {want:8.3f}"
            print(text)
        summary = "  RMS difference / max |measured|:"
        for (scenario, phase), pairs in errors.items():
            rms = (sum(d * d for d, _ in pairs) / len(pairs)) ** 0.5
            scale = max(abs(w) for _, w in pairs)
            worst = max(worst, rms / scale)
            summary += f"  {scenario[1:]} Hz {phase[2:]}: {100 * rms / scale:5.1f}%"
        print(summary)
    print(f"\nWorst column: {100 * worst:.1f}%")


if __name__ == "__main__":
    main()
