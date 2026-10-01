// Two coaxial rings of round wire in free space, truncated by a sphere: the
// meridian (r, z) section for the axisymmetric solvers. The 3D model is the
// same geometry revolved (coaxial_rings.geo).
//
// Regenerate with
//   gmsh coaxial_rings_axi.geo -2 -order 2 -format msh22 -bin -o coaxial_rings_axi.msh
//
// Physical groups (mesh attributes):
//   surface 1 "Air"      the half-disk r^2 + z^2 < D^2, r > 0, outside the wires
//   surface 2 "Ring1"    wire radius a, centred at (R1, Z1)
//   surface 3 "Ring2"    wire radius a, centred at (R2, Z2)
//   curve 1   "Outer"    the arc r^2 + z^2 = D^2
//   curve 2   "Axis"     r = 0

SetFactory("OpenCASCADE");

R1 = 0.04;  Z1 = -0.01;
R2 = 0.07;  Z2 = 0.01;
a = 0.01;
D = DefineNumber[0.25, Name "Parameters/outer radius"];
h = DefineNumber[0.0025, Name "Parameters/element size at the wires"];

Disk(1) = {0, 0, 0, D};
Rectangle(2) = {-D, -D, 0, D, 2 * D};
BooleanDifference(3) = { Surface{1}; Delete; }{ Surface{2}; Delete; };
Disk(4) = {R1, Z1, 0, a};
Disk(5) = {R2, Z2, 0, a};
BooleanFragments{ Surface{3}; Delete; }{ Surface{4, 5}; Delete; }

eps = 1e-6;
ring1[] = Surface In BoundingBox{R1 - a - eps, Z1 - a - eps, -eps, R1 + a + eps, Z1 + a + eps, eps};
ring2[] = Surface In BoundingBox{R2 - a - eps, Z2 - a - eps, -eps, R2 + a + eps, Z2 + a + eps, eps};
all[] = Surface{:};
air[] = all[];
air[] -= {ring1[], ring2[]};
axis[] = Curve In BoundingBox{-eps, -D - eps, -eps, eps, D + eps, eps};
outer[] = CombinedBoundary{ Surface{all[]}; };
outer[] -= {axis[]};

Physical Surface("Air", 1) = {air[]};
Physical Surface("Ring1", 2) = {ring1[]};
Physical Surface("Ring2", 3) = {ring2[]};
Physical Curve("Outer", 1) = {outer[]};
Physical Curve("Axis", 2) = {axis[]};

// Size h at the wires, growing linearly with distance to D / 10.
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Field[1] = Distance;
wires[] = Boundary{ Surface{ring1[], ring2[]}; };
Field[1].CurvesList = {wires[]};
Field[2] = MathEval;
Field[2].F = Sprintf("min(%g + 0.25 * F1, %g)", h, D / 10);
Background Field = 2;
Mesh.Algorithm = 6;
Mesh.HighOrderOptimize = 2;
