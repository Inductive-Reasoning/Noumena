// TEAM Workshop Problem 7: asymmetrical conductor with a hole.
//
// A 19 mm aluminium plate with an eccentric square hole, under a racetrack
// coil, in the air box of the problem's recommended mesh (B . n = 0 on its
// walls). Coordinates in metres, in the problem's frame (the plate's corner
// at the origin).
//
// Groups: Air 1, Plate 2, Coil 3 (volumes); Outer 1, Cut 2 (surfaces). The
// cut is a cross-section of the coil's straight side along y = 0, crossed
// by the coil current in +x.
//
// The mesh size is set by h_plate (in the plate, which must resolve the
// 6 mm skin depth at 200 Hz) and h_coil; both grow with distance from the
// plate. Regenerate with: gmsh -3 -format msh2 team7.geo -o team7.msh

SetFactory("OpenCASCADE");

// Override with gmsh -setnumber h_plate <size> (and h_coil).
If (!Exists(h_plate))
  h_plate = 0.006;
EndIf
If (!Exists(h_coil))
  h_coil = 0.012;
EndIf

// Air box (Fig. 2 of the problem).
box = newv;
Box(box) = {-1.353, -1.353, -0.300, 3.0, 3.0, 0.749};

// Plate, 294 x 294 x 19 mm, less the 108 x 108 mm hole at (18, 18).
slab = newv;
Box(slab) = {0, 0, 0, 0.294, 0.294, 0.019};
hole = newv;
Box(hole) = {0.018, 0.018, 0, 0.108, 0.108, 0.019};
plate_body() = BooleanDifference{ Volume{slab}; Delete; }{ Volume{hole}; Delete; };

// Coil: a 200 x 200 mm outline with 50 mm corner radii, less a 150 x 150 mm
// window with 25 mm radii, from z = 49 to 149 mm.
outline = news;
Rectangle(outline) = {0.094, 0, 0.049, 0.200, 0.200, 0.050};
window = news;
Rectangle(window) = {0.119, 0.025, 0.049, 0.150, 0.150, 0.025};
section() = BooleanDifference{ Surface{outline}; Delete; }{ Surface{window}; Delete; };
coil_body() = Extrude {0, 0, 0.100} { Surface{section(0)}; };

// The cut: the coil's cross-section at x = 194 mm on its straight side y < 25 mm.
p = newp;
Point(p) = {0.194, 0, 0.049};
Point(p + 1) = {0.194, 0.025, 0.049};
Point(p + 2) = {0.194, 0.025, 0.149};
Point(p + 3) = {0.194, 0, 0.149};
l = newl;
Line(l) = {p, p + 1};
Line(l + 1) = {p + 1, p + 2};
Line(l + 2) = {p + 2, p + 3};
Line(l + 3) = {p + 3, p};
loop = newcl;
Curve Loop(loop) = {l, l + 1, l + 2, l + 3};
cut_face = news;
Plane Surface(cut_face) = {loop};

BooleanFragments{ Volume{box, plate_body(0), coil_body(1)}; Delete; }{ Surface{cut_face}; Delete; }

// Identify the pieces after fragmentation by location.
eps = 1e-6;
plate() = Volume In BoundingBox{-eps, -eps, -eps, 0.294 + eps, 0.294 + eps, 0.019 + eps};
coils() = Volume In BoundingBox{0.094 - eps, -eps, 0.049 - eps, 0.294 + eps, 0.200 + eps, 0.149 + eps};
all() = Volume{:};
air() = all();
air() -= plate();
air() -= coils();
cut() = Surface In BoundingBox{0.194 - eps, -eps, 0.049 - eps, 0.194 + eps, 0.025 + eps, 0.149 + eps};
outer() = Surface In BoundingBox{-1.353 - eps, -1.353 - eps, -0.300 - eps, 1.647 + eps, 1.647 + eps, 0.449 + eps};
inner() = Surface In BoundingBox{-1.353 + eps, -1.353 + eps, -0.300 + eps, 1.647 - eps, 1.647 - eps, 0.449 - eps};
outer() -= inner();

Physical Volume("Air", 1) = air();
Physical Volume("Plate", 2) = plate();
Physical Volume("Coil", 3) = coils();
Physical Surface("Outer", 1) = outer();
Physical Surface("Cut", 2) = cut();

// Size: h_plate in and near the plate, h_coil in the coil, growing to a
// tenth of the box away from them.
plate_surface() = Boundary{ Volume{plate()}; };
coil_surface() = Boundary{ Volume{coils()}; };
Field[1] = Distance;
Field[1].SurfacesList = {plate_surface()};
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_plate;
Field[2].SizeMax = 0.3;
Field[2].DistMin = 0.01;
Field[2].DistMax = 1.0;
Field[3] = Distance;
Field[3].SurfacesList = {coil_surface()};
Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = h_coil;
Field[4].SizeMax = 0.3;
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
