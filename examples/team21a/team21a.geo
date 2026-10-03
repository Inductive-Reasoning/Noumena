// TEAM Workshop Problem 21a: stray-field loss in a non-magnetic steel plate.
//
// Two square racetrack coils (axis z) carrying opposite currents beside a
// 10 mm non-magnetic steel plate with 0-3 slits (P21a-0 ... P21a-3).
// Coordinates in metres: the plate is centred on x = 0 (|x| <= 5 mm), the
// coils' near face is 12 mm from it (x = 17 mm), z = 0 is the mid-plane
// between the coils.
//
// Groups: Air 1, Plate 2, Coil1 3 (z > 0), Coil2 4 (z < 0); Outer 1 and,
// for each coil, a cut through its near side on the plane y = 0 (Cut1 2,
// Cut2 3), crossed by its current along y.
//
// gmsh -3 -format msh2 -setnumber slits <0..3> team21a.geo -o team21a-<n>.msh

SetFactory("OpenCASCADE");

If (!Exists(slits))
  slits = 0;
EndIf
// The plate is thin against its 60 mm skin depth at 50 Hz: at order 2 these
// sizes give losses within 0.2% of a mesh refined to 7 and 15 mm.
If (!Exists(h_plate))
  h_plate = 0.010;
EndIf
If (!Exists(h_coil))
  h_coil = 0.020;
EndIf
pad = 0.6;  // air beyond the plate and coils on every side

// Plate 10 x 360 x 820 mm, less 10 mm wide, 660 mm long through-slits.
plate_body = newv;
Box(plate_body) = {-0.005, -0.180, -0.410, 0.010, 0.360, 0.820};
slit_y() = {};
If (slits == 1)
  slit_y() = {0};
ElseIf (slits == 2)
  slit_y() = {-0.060, 0.060};
ElseIf (slits == 3)
  slit_y() = {-0.090, 0, 0.090};
EndIf
For i In {0 : #slit_y() - 1}
  s = newv;
  Box(s) = {-0.005, slit_y(i) - 0.005, -0.330, 0.010, 0.010, 0.660};
  plate_body() = BooleanDifference{ Volume{plate_body}; Delete; }{ Volume{s}; Delete; };
EndFor

// Coils: 270 x 270 mm outline (R45 corners) less a 200 x 200 mm window
// (R10, concentric corners), each 217 mm tall.
For c In {1 : 2}
  z0 = (c == 1) ? 0.012 : -0.229;
  outline = news;
  Rectangle(outline) = {0.017, -0.135, z0, 0.270, 0.270, 0.045};
  window = news;
  Rectangle(window) = {0.052, -0.100, z0, 0.200, 0.200, 0.010};
  section() = BooleanDifference{ Surface{outline}; Delete; }{ Surface{window}; Delete; };
  extruded() = Extrude {0, 0, 0.217} { Surface{section(0)}; };
  coil_body~{c} = extruded(1);
  // The cut: the near side's cross-section on y = 0.
  p = newp;
  Point(p) = {0.017, 0, z0};
  Point(p + 1) = {0.052, 0, z0};
  Point(p + 2) = {0.052, 0, z0 + 0.217};
  Point(p + 3) = {0.017, 0, z0 + 0.217};
  l = newl;
  Line(l) = {p, p + 1};
  Line(l + 1) = {p + 1, p + 2};
  Line(l + 2) = {p + 2, p + 3};
  Line(l + 3) = {p + 3, p};
  loop = newcl;
  Curve Loop(loop) = {l, l + 1, l + 2, l + 3};
  cut_face~{c} = news;
  Plane Surface(cut_face~{c}) = {loop};
EndFor

box = newv;
Box(box) = {-0.005 - pad, -0.180 - pad, -0.410 - pad, 0.292 + 2 * pad, 0.360 + 2 * pad, 0.820 + 2 * pad};

BooleanFragments{ Volume{box, plate_body(0), coil_body~{1}, coil_body~{2}}; Delete; }{ Surface{cut_face~{1}, cut_face~{2}}; Delete; }

eps = 1e-6;
plate() = Volume In BoundingBox{-0.005 - eps, -0.180 - eps, -0.410 - eps, 0.005 + eps, 0.180 + eps, 0.410 + eps};
coil1() = Volume In BoundingBox{0.017 - eps, -0.135 - eps, 0.012 - eps, 0.287 + eps, 0.135 + eps, 0.229 + eps};
coil2() = Volume In BoundingBox{0.017 - eps, -0.135 - eps, -0.229 - eps, 0.287 + eps, 0.135 + eps, -0.012 + eps};
air() = Volume{:};
air() -= plate();
air() -= coil1();
air() -= coil2();
cut1() = Surface In BoundingBox{0.017 - eps, -eps, 0.012 - eps, 0.052 + eps, eps, 0.229 + eps};
cut2() = Surface In BoundingBox{0.017 - eps, -eps, -0.229 - eps, 0.052 + eps, eps, -0.012 + eps};
outer() = Surface In BoundingBox{-0.005 - pad - eps, -0.180 - pad - eps, -0.410 - pad - eps,
                                 0.287 + pad + eps, 0.180 + pad + eps, 0.410 + pad + eps};
inner() = Surface In BoundingBox{-0.005 - pad + eps, -0.180 - pad + eps, -0.410 - pad + eps,
                                 0.287 + pad - eps, 0.180 + pad - eps, 0.410 + pad - eps};
outer() -= inner();

Physical Volume("Air", 1) = air();
Physical Volume("Plate", 2) = plate();
Physical Volume("Coil1", 3) = coil1();
Physical Volume("Coil2", 4) = coil2();
Physical Surface("Outer", 1) = outer();
Physical Surface("Cut1", 2) = cut1();
Physical Surface("Cut2", 3) = cut2();

// Size: h_plate in and near the plate, h_coil in the coils, growing away.
plate_surface() = Boundary{ Volume{plate()}; };
coil_surface() = Boundary{ Volume{coil1(), coil2()}; };
Field[1] = Distance;
Field[1].SurfacesList = {plate_surface()};
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_plate;
Field[2].SizeMax = 0.25;
Field[2].DistMin = 0.01;
Field[2].DistMax = 1.0;
Field[3] = Distance;
Field[3].SurfacesList = {coil_surface()};
Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = h_coil;
Field[4].SizeMax = 0.25;
Field[4].DistMin = 0.01;
Field[4].DistMax = 1.0;
Field[5] = Min;
Field[5].FieldsList = {2, 4};
Background Field = 5;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm3D = 10;  // HXT
Mesh.Optimize = 1;
