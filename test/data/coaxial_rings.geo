// Two coaxial rings of round wire (tori) in free space, truncated by a
// sphere: the 3D counterpart of coaxial_rings_axi.geo, which is its meridian
// section. One quadrant (x > 0, y > 0) is modelled: the current and A are
// azimuthal, so A is normal to every meridian plane and n x A = 0 holds
// there exactly. Inductances of the quadrant are a quarter of the rings'.
//
// Regenerate with
//   gmsh coaxial_rings.geo -3 -order 2 -format msh22 -bin -o coaxial_rings.msh
//
// Physical groups (mesh attributes):
//   volume 1  "Air"       the quadrant of the ball r < D outside the wires
//   volume 2  "Ring1"     torus, ring radius R1 at height Z1, wire radius a
//   volume 3  "Ring2"     torus, ring radius R2 at height Z2, wire radius a
//   surface 1 "Outer"     the sphere r = D
//   surface 2 "Meridian"  the planes y = 0 and x = 0

SetFactory("OpenCASCADE");

R1 = 0.04;  Z1 = -0.01;
R2 = 0.07;  Z2 = 0.01;
a = 0.01;
D = DefineNumber[0.25, Name "Parameters/outer radius"];
h = DefineNumber[0.008, Name "Parameters/element size at the wires"];

Sphere(1) = {0, 0, 0, D};
Torus(2) = {0, 0, Z1, R1, a};
Torus(3) = {0, 0, Z2, R2, a};
BooleanFragments{ Volume{1}; Delete; }{ Volume{2, 3}; Delete; }
pieces[] = Volume{:};
quadrant = newv;
Box(quadrant) = {0, 0, -D, D, D, 2 * D};
BooleanIntersection{ Volume{pieces[]}; Delete; }{ Volume{quadrant}; Delete; }

// OpenCASCADE's bounding box of a torus is exact in z but loose radially,
// so each ring is selected by its z range with a generous radial margin,
// which still excludes the other ring and the air.
eps = 1e-4;
w1 = 1.2 * (R1 + a);
w2 = 1.2 * (R2 + a);
ring1[] = Volume In BoundingBox{-eps, -eps, Z1 - a - eps, w1, w1, Z1 + a + eps};
ring2[] = Volume In BoundingBox{-eps, -eps, Z2 - a - eps, w2, w2, Z2 + a + eps};
all[] = Volume{:};
air[] = all[];
air[] -= {ring1[], ring2[]};
plane_y[] = Surface In BoundingBox{-eps, -eps, -D - eps, D + eps, eps, D + eps};
plane_x[] = Surface In BoundingBox{-eps, -eps, -D - eps, eps, D + eps, D + eps};
outer[] = CombinedBoundary{ Volume{all[]}; };
outer[] -= {plane_y[], plane_x[]};

Physical Volume("Air", 1) = {air[]};
Physical Volume("Ring1", 2) = {ring1[]};
Physical Volume("Ring2", 3) = {ring2[]};
Physical Surface("Outer", 1) = {outer[]};
Physical Surface("Meridian", 2) = {plane_y[], plane_x[]};

// Size h at the wires, growing linearly with distance to D / 6, and at most
// 12 elements per 2 pi of curvature.
wires[] = Boundary{ Volume{ring1[], ring2[]}; };
wires[] -= {plane_y[], plane_x[]};
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 12;
Field[1] = Distance;
Field[1].SurfacesList = {wires[]};
Field[2] = MathEval;
Field[2].F = Sprintf("min(%g + 0.3 * F1, %g)", h, D / 6);
Background Field = 2;
Mesh.Algorithm = 6;
Mesh.Algorithm3D = 1;
Mesh.Optimize = 1;
Mesh.HighOrderOptimize = 2;
