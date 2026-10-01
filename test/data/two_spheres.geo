// Two conducting spheres in free space, truncated by a far-field sphere:
// the capacitance-matrix benchmark of J. Lekner, "Capacitance coefficients
// of two spheres", J. Electrostatics 69 (2011) 11-14, in the configuration
// of the Palace "spheres" example (a = 1 cm, b = 2 cm, centres 5 cm apart).
//
// Regenerate with
//   gmsh two_spheres.geo -3 -order 2 -format msh22 -bin -o two_spheres.msh
//
// Physical groups (mesh attributes):
//   volume 1  "Domain"    the vacuum between the spheres
//   surface 1 "FarField"  the truncating sphere, radius R about the origin
//   surface 2 "SphereA"   radius a, centred at (-c/2, 0, 0)
//   surface 3 "SphereB"   radius b, centred at (+c/2, 0, 0)

SetFactory("OpenCASCADE");

a = 0.01;   // m
b = 0.02;
c = 0.05;
R = DefineNumber[0.25, Name "Parameters/far-field radius"];
n = DefineNumber[8, Name "Parameters/elements per 2 pi on the spheres"];

Sphere(1) = {0, 0, 0, R};
Sphere(2) = {-c / 2, 0, 0, a};
Sphere(3) = {c / 2, 0, 0, b};
BooleanDifference(4) = { Volume{1}; Delete; }{ Volume{2, 3}; Delete; };

eps = 1e-6;
far[] = Surface In BoundingBox{-R - eps, -R - eps, -R - eps, R + eps, R + eps, R + eps};
sa[] = Surface In BoundingBox{-c/2 - a - eps, -a - eps, -a - eps, -c/2 + a + eps, a + eps, a + eps};
sb[] = Surface In BoundingBox{c/2 - b - eps, -b - eps, -b - eps, c/2 + b + eps, b + eps, b + eps};
far[] -= {sa[], sb[]};

Physical Volume("Domain", 1) = {4};
Physical Surface("FarField", 1) = {far[]};
Physical Surface("SphereA", 2) = {sa[]};
Physical Surface("SphereB", 3) = {sb[]};

// Element size: n elements per 2 pi of curvature on the spheres, growing
// linearly away from them to R / 4 at the far field.
Mesh.MeshSizeFromCurvature = n;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeMin = 2 * Pi * a / n;
Mesh.MeshSizeMax = R / 4;
Field[1] = Distance;
Field[1].SurfacesList = {sa[], sb[]};
Field[2] = MathEval;
Field[2].F = Sprintf("%g + 0.3 * F1", 2 * Pi * a / n);
Background Field = 2;
Mesh.Algorithm = 6;
Mesh.Algorithm3D = 1;
Mesh.Optimize = 1;
Mesh.HighOrderOptimize = 2;
