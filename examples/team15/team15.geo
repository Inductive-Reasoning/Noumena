// TEAM Workshop Problem 15: rectangular slot in a thick plate (eddy-current
// nondestructive evaluation).
//
// A circular air-cored coil above a 12.22 mm aluminium-alloy plate with a
// surface-breaking slot, 12.6 mm long (along x), 5 mm deep and 0.28 mm wide,
// centred on the origin. The plate's top face is z = 0; the coil's axis is
// vertical through (X, 0). Coordinates in metres.
//
// Half model: y >= 0, with y = 0 (the plane of symmetry through the slot and
// the coil's axis) part of the n x A = 0 boundary. The slot is its own volume,
// so the flawed and unflawed plates share one mesh and differ only in the
// slot's material.
//
// Groups: Air 1, Plate 2, Slot 3, Coil 4; Outer 1 (including y = 0).
//
// gmsh -3 -format msh2 -setnumber problem <1|2> -setnumber X <x in m> team15.geo -o team15.msh

SetFactory("OpenCASCADE");

If (!Exists(problem))
  problem = 1;
EndIf
If (!Exists(X))
  X = 0;
EndIf

// Coil (inner radius, outer radius, length, lift-off) and mesh sizes: the
// plate is resolved to a fraction of the skin depth (3.04 mm at 900 Hz,
// 1.09 mm at 7 kHz) where the coil's field reaches it.
If (problem == 1)
  r_in = 6.15e-3; r_out = 12.4e-3; b = 6.15e-3; lift = 0.88e-3;
  h_near = 1.0e-3;
Else
  r_in = 9.34e-3; r_out = 18.4e-3; b = 9.0e-3; lift = 2.03e-3;
  h_near = 0.6e-3;
EndIf
If (!Exists(h_plate))
  h_plate = h_near;
EndIf
If (!Exists(h_slot))
  h_slot = 0.25e-3;
EndIf

t = 12.22e-3;            // plate thickness
c = 6.3e-3;              // slot half-length
d = 5.0e-3;              // slot depth
w2 = 0.14e-3;            // slot half-width (the model's share of it)
plate_half = 0.1;        // plate extent in x and y, far beyond the coil's reach
box_half = 0.15;         // air box

plate_body = newv;
Box(plate_body) = {-plate_half, 0, -t, 2 * plate_half, plate_half, t};
slot_body = newv;
Box(slot_body) = {-c, 0, -d, 2 * c, w2, d};

// Half of the annular coil, y >= 0.
outer_cyl = newv;
Cylinder(outer_cyl) = {X, 0, lift, 0, 0, b, r_out};
inner_cyl = newv;
Cylinder(inner_cyl) = {X, 0, lift, 0, 0, b, r_in};
ring() = BooleanDifference{ Volume{outer_cyl}; Delete; }{ Volume{inner_cyl}; Delete; };
half = newv;
Box(half) = {X - 2 * r_out, 0, lift - b, 4 * r_out, 2 * r_out, 3 * b};
coil_body() = BooleanIntersection{ Volume{ring()}; Delete; }{ Volume{half}; Delete; };

box = newv;
Box(box) = {-box_half, 0, -box_half, 2 * box_half, box_half, 2 * box_half};

BooleanFragments{ Volume{box, plate_body, slot_body, coil_body()}; Delete; }{}

eps = 1e-5;  // selection tolerance, well under the 0.14 mm slot
slot() = Volume In BoundingBox{-c - eps, -eps, -d - eps, c + eps, w2 + eps, eps};
plate() = Volume In BoundingBox{-plate_half - eps, -eps, -t - eps, plate_half + eps, plate_half + eps, eps};
plate() -= slot();
coil() = Volume In BoundingBox{X - r_out - eps, -eps, lift - eps, X + r_out + eps, r_out + eps, lift + b + eps};
air() = Volume{:};
air() -= plate();
air() -= slot();
air() -= coil();
outer() = CombinedBoundary{ Volume{:}; };

Physical Volume("Air", 1) = air();
Physical Volume("Plate", 2) = plate();
Physical Volume("Slot", 3) = slot();
Physical Volume("Coil", 4) = coil();
Physical Surface("Outer", 1) = outer();

// Sizes: h_slot at the slot, h_plate in the part of the plate and air the
// coil's field reaches along its scan, growing to 3 cm at the box.
slot_surface() = Boundary{ Volume{slot()}; };
Field[1] = Distance;
Field[1].SurfacesList = {slot_surface()};
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_slot;
Field[2].SizeMax = 0.03;
Field[2].DistMin = 0.5e-3;
Field[2].DistMax = 0.1;
reach = r_out + 3 * b;
Field[3] = Box;
Field[3].VIn = h_plate;
Field[3].VOut = 0.03;
Field[3].XMin = Min(-c, X - r_out) - b;
Field[3].XMax = Max(c, X + r_out) + b;
Field[3].YMin = 0;
Field[3].YMax = r_out + b;
Field[3].ZMin = -Min(t, 3 * b);
Field[3].ZMax = lift + b + b;
Field[3].Thickness = reach;
Field[4] = Min;
Field[4].FieldsList = {2, 3};
Background Field = 4;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm3D = 10;  // HXT
Mesh.Optimize = 1;
Mesh.ElementOrder = 2;  // curved coil faces
Mesh.HighOrderOptimize = 1;
