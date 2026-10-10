# Copyright (c) 2026 T. C. Raymond
# SPDX-License-Identifier: MIT
"""Generate the current-loop mesh from a Netgen GEOMETRY definition.

STATUS: one-off feasibility probe, not a supported tool. Nothing in the build
or the test suite runs this. It is kept only as the reproducible artifact behind
the measurements in docs/mesh_formats.md, and it requires a Netgen install that
is not a project dependency. If the Netgen question is ever closed -- either way
-- this script can be deleted.

This is the second half of the Netgen feasibility study (see
docs/mesh_formats.md). The first half only proved MFEM can READ an areamesh2
file; this script proves we can DRIVE Netgen from a geometry description and
get a mesh whose attribute ids we control well enough to hand to a config.

The geometry mirrors test/test_solvers.cpp: an axisymmetric (r, z) half-plane
containing a square-section conducting loop of radius kLoopRadius, surrounded
by air out to a far boundary at kLoopDomain.

Attribute contract (must match examples/current_loop/config.json):
    domain   1 = air, 2 = loop conductor
    boundary 1 = axis (r=0), 2 = outer, 3 = top/bottom

Run:
    $env:PATH="C:\\netgen\\bin;$env:PATH"
    $env:PYTHONPATH="C:\\netgen\\lib\\site-packages"
    python tools/netgen_current_loop.py loop_netgen.mesh
"""

import sys

from netgen.geom2d import SplineGeometry

LOOP_RADIUS = 0.1     # loop radius a [m]
LOOP_SIDE = 0.002     # conductor cross-section side [m]
LOOP_DOMAIN = 4.0     # far-field boundary distance [m]

# Netgen numbers domains by the order in which materials are SET.
#
# Boundary numbering is the sharper trap. Elements1D().index is the SPLINE
# DECLARATION INDEX (1..n), not the bc= value: passing bc=3 does NOT make the
# segment come back as index 3. The first attempt here assumed it did and got
# 'updown' numbered 1, then found 8 distinct ids where 4 were expected.
#
# So the mapping is built from Append()'s RETURN VALUE, which is the spline
# index, and collapsed onto our attribute contract explicitly. This is the
# id-provenance risk from docs/mesh_formats.md, made mechanical rather than
# assumed: the geometry declares the role, the code derives the number.
MAT_TO_ATTR = {"air": 1, "loop": 2}

# Exterior boundary roles, plus the air/conductor interface. Netgen emits
# interior segments as boundary elements too, so MFEM will see the interface
# as a bdr attribute even though no boundary condition references it.
ROLE_TO_ATTR = {"axis": 1, "outer": 2, "updown": 3, "interface": 4}


def build_geometry():
    """Return (geometry, spline_index -> boundary attribute)."""
    geo = SplineGeometry()
    spline_attr = {}

    def edge(a, b, role, leftdomain, rightdomain):
        # Append() returns a 0-BASED spline index, while Elements1D().index is
        # 1-BASED. Normalize here so the lookup below cannot be off by one.
        idx = geo.Append(["line", a, b], leftdomain=leftdomain,
                         rightdomain=rightdomain)
        spline_attr[idx + 1] = ROLE_TO_ATTR[role]
        return idx

    r_lo = LOOP_RADIUS - 0.5 * LOOP_SIDE
    r_hi = LOOP_RADIUS + 0.5 * LOOP_SIDE
    z_lo = -0.5 * LOOP_SIDE
    z_hi = 0.5 * LOOP_SIDE

    # Outer air box. The r=0 edge is the symmetry axis and gets its own bc so
    # the axisymmetric solver can find it; the rest is the truncation boundary.
    air = geo.AppendPoint(0.0, -LOOP_DOMAIN), \
        geo.AppendPoint(LOOP_DOMAIN, -LOOP_DOMAIN), \
        geo.AppendPoint(LOOP_DOMAIN, LOOP_DOMAIN), \
        geo.AppendPoint(0.0, LOOP_DOMAIN)

    edge(air[0], air[1], "updown", 1, 0)
    edge(air[1], air[2], "outer", 1, 0)
    edge(air[2], air[3], "updown", 1, 0)
    edge(air[3], air[0], "axis", 1, 0)

    # Conductor, embedded in the air region: domain 2 on the inside, 1 outside.
    loop = geo.AppendPoint(r_lo, z_lo), \
        geo.AppendPoint(r_hi, z_lo), \
        geo.AppendPoint(r_hi, z_hi), \
        geo.AppendPoint(r_lo, z_hi)
    for a, b in zip(loop, loop[1:] + loop[:1]):
        edge(a, b, "interface", 2, 1)

    geo.SetMaterial(1, "air")
    geo.SetMaterial(2, "loop")

    # Resolve the conductor: its section is 2 mm across in a 8 m domain, so
    # without a local size the mesher will not see it at all.
    geo.SetDomainMaxH(2, LOOP_SIDE / 4.0)
    return geo, spline_attr


def write_areamesh2(mesh, spline_attr, path):
    """Write a Netgen 2D mesh in the areamesh2 format MFEM reads.

    Netgen's own Export() has no areamesh2 writer, so we emit it. Indices are
    1-based and boundary elements precede elements (the reverse of the MFEM
    format). Segment indices are translated from Netgen's spline numbering to
    our attribute contract on the way out.
    """
    points = list(mesh.Points())
    els = list(mesh.Elements2D())
    segs = list(mesh.Elements1D())

    with open(path, "w") as f:
        f.write("areamesh2\n\n")

        f.write(f"{len(segs)}\n")
        for s in segs:
            attr = spline_attr[s.index]
            v = list(s.vertices)
            f.write(f"{attr} {v[0].nr} {v[1].nr}\n")
        f.write("\n")

        f.write(f"{len(els)}\n")
        for e in els:
            v = list(e.vertices)
            f.write(f"{e.index} {len(v)} " + " ".join(str(x.nr) for x in v) + "\n")
        f.write("\n")

        f.write(f"{len(points)}\n")
        for p in points:
            x, y, _ = p.p
            f.write(f"{x!r} {y!r}\n")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "loop_netgen.mesh"

    geo, spline_attr = build_geometry()
    mesh = geo.GenerateMesh(maxh=LOOP_DOMAIN / 8.0, grading=0.2)

    # Verify the ids Netgen actually assigned against the contract above.
    for idx in sorted({e.index for e in mesh.Elements2D()}):
        name = mesh.GetMaterial(idx)
        expected = MAT_TO_ATTR.get(name)
        print(f"domain   {idx} -> material {name!r} (expected attr {expected})")
        assert expected == idx, f"domain id drift: {name!r} got {idx}, want {expected}"

    produced = sorted({spline_attr[s.index] for s in mesh.Elements1D()})
    attr_to_role = {v: k for k, v in ROLE_TO_ATTR.items()}
    for attr in produced:
        n = sum(1 for s in mesh.Elements1D() if spline_attr[s.index] == attr)
        print(f"boundary {attr} -> {attr_to_role[attr]!r} ({n} segments)")
    assert produced == sorted(ROLE_TO_ATTR.values()), \
        f"boundary attribute mismatch: got {produced}"

    write_areamesh2(mesh, spline_attr, out)
    print(f"\nwrote {out}: {len(list(mesh.Points()))} vertices, "
          f"{len(list(mesh.Elements2D()))} elements, "
          f"{len(list(mesh.Elements1D()))} boundary segments")


if __name__ == "__main__":
    main()
