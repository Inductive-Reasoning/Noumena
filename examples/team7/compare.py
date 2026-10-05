#!/usr/bin/env python3
"""Compare a TEAM 7 run's probes with the measurements in measured.csv.

Usage: compare.py [results_directory]   (default: results, beside this script)

The solver's phasors X are peak values with time dependence Re(X e^{j w t}),
so the instantaneous value at w t = 0 is Re X and at 90 degrees -Im X. For
each table this prints computed against measured, and the RMS difference of
each column as a fraction of that column's largest measured magnitude.

The surface current tables are compared with the opposite surface to the one
their labels name: the table labelled A4-B4 (z = 0) with the coil-facing
surface z = 19 mm, and A3-B3 with z = 0 (see README.md).
"""

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# table -> (probe, field, component, factor to the table's unit)
TABLES = {
    "A1B1": ("A1B1", "B", "z", 1e4),     # T -> gauss
    "A2B2": ("A2B2", "B", "z", 1e4),
    "A3B3": ("Bottom", "J", "y", 1e-6),  # A/m^2 -> 1e6 A/m^2
    "A4B4": ("Top", "J", "y", 1e-6),
}
SCENARIOS = {"f50": "scenario_000000_50Hz", "f200": "scenario_000001_200Hz"}
PHASES = ("wt0", "wt90")


def measured():
    rows = {}
    with open(HERE / "measured.csv") as f:
        for row in csv.DictReader(line for line in f if not line.startswith("#")):
            rows.setdefault(row["line"], []).append(row)
    return rows


def computed(results, scenario, table):
    probe, field, component, factor = TABLES[table]
    path = results / "probes" / f"{SCENARIOS[scenario]}_{probe}.csv"
    with open(path) as f:
        samples = list(csv.DictReader(f))
    values = {}
    for s in samples:
        x_mm = round(1000 * float(s["x"]), 3)
        values[x_mm] = (factor * float(s[f"{field}_Real_{component}"]),
                        -factor * float(s[f"{field}_Imag_{component}"]))
    return values


def compare(results):
    """{(table, scenario, phase): RMS difference / max |measured|}, and the
    printable report."""
    errors, report = {}, []
    for table, rows in measured().items():
        probe, field, component, _ = TABLES[table]
        report.append(f"\n{table} ({field}{component}, probe {probe})")
        header = "  x [mm]"
        for scenario in SCENARIOS:
            for phase in PHASES:
                header += f"  {scenario[1:]:>3} Hz {phase[2:]:>2}: computed / measured"
        report.append(header)
        values = {s: computed(results, s, table) for s in SCENARIOS}
        pairs = {}
        for row in rows:
            x = float(row["x_mm"])
            text = f"  {x:6.0f}"
            for scenario in SCENARIOS:
                for index, phase in enumerate(PHASES):
                    got, want = values[scenario][x][index], float(row[f"{scenario}_{phase}"])
                    pairs.setdefault((scenario, phase), []).append((got, want))
                    text += f"  {got:17.3f} / {want:8.3f}"
            report.append(text)
        summary = "  RMS difference / max |measured|:"
        for (scenario, phase), column in pairs.items():
            rms = (sum((g - w) ** 2 for g, w in column) / len(column)) ** 0.5
            scale = max(abs(w) for _, w in column)
            errors[(table, scenario, phase)] = rms / scale
            summary += f"  {scenario[1:]} Hz {phase[2:]}: {100 * rms / scale:5.1f}%"
        report.append(summary)
    return errors, "\n".join(report)


def main():
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    _, report = compare(results)
    print(report)


if __name__ == "__main__":
    main()
