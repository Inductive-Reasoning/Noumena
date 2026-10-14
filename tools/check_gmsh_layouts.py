#!/usr/bin/env python3
# Copyright (c) 2026 T. C. Raymond
# SPDX-License-Identifier: MIT
"""Cross-check gmsh_results_writer's Lagrange node layouts against Gmsh itself.

The writer builds each element's high-order node order recursively (see
GetHoLayout in src/io/gmsh_results_writer.hpp). This script compares every
layout, node by node, with gmsh.model.mesh.getElementProperties(), the
authoritative definition, for triangles and quadrilaterals (orders 1-10),
tetrahedra (1-10) and hexahedra (1-9).

Usage (requires `pip install gmsh`):
    g++ -std=c++17 -Isrc <mfem include flags> tools/dump_gmsh_layouts.cpp \\
        <libmfem.a> -o dump_gmsh_layouts
    ./dump_gmsh_layouts | python3 tools/check_gmsh_layouts.py

Gmsh's quadrilateral/hexahedron reference domain is [-1, 1]^d while MFEM's
(and the writer's) is [0, 1]^d; coordinates are mapped before comparing.
Simplices share the same reference domain.
"""
import sys

import gmsh

TOL = 1e-9


def main():
    lines = sys.stdin.read().split("\n")
    gmsh.initialize()
    failures = checked = 0
    i = 0
    while i < len(lines) and lines[i].strip():
        family, order, gtype, count = lines[i].split()
        order, gtype, count = int(order), int(gtype), int(count)
        ours = [tuple(map(float, lines[i + 1 + k].split())) for k in range(count)]
        i += 1 + count

        expected_type = gmsh.model.mesh.getElementType(family, order)
        _, dim, _, n, coords, _ = gmsh.model.mesh.getElementProperties(expected_type)
        ref = [tuple(coords[dim * k + c] if c < dim else 0.0 for c in range(3))
               for k in range(n)]
        if family in ("Quadrangle", "Hexahedron"):
            ref = [tuple((x + 1.0) / 2.0 if c < dim else 0.0
                         for c, x in enumerate(p)) for p in ref]

        problem = None
        if gtype != expected_type:
            problem = f"type {gtype} != gmsh {expected_type}"
        elif count != n:
            problem = f"{count} nodes != gmsh {n}"
        else:
            for k, (a, b) in enumerate(zip(ours, ref)):
                if max(abs(x - y) for x, y in zip(a, b)) > TOL:
                    problem = f"node {k}: {a} != gmsh {b}"
                    break
        checked += 1
        if problem:
            failures += 1
            print(f"FAIL {family} order {order}: {problem}")
    gmsh.finalize()
    print(f"{checked - failures}/{checked} layouts match Gmsh")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
