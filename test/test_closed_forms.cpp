// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// Closed-form verification on round geometry: a round conductor in
// magnetostatics (planar and 3D), its AC impedance with skin effect (Bessel
// functions), a conducting tube shielding an axial AC field, and dielectric,
// permeable and conducting spheres in uniform fields.
// The curved meshes are generated with Gmsh at second order, so circles and
// spheres are represented to the accuracy of the discretization; those tests
// skip without Gmsh.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

#include "bessel.hpp"
#include "gmsh_fixture.hpp"
#include "config/input_parser.hpp"
#include "core/constants.hpp"
#include "solvers/electrostatic_solver.hpp"
#include "solvers/magnetoquasistatic_solver.hpp"
#include "solvers/magnetoquasistatic_solver_3d.hpp"
#include "solvers/magnetostatic_solver.hpp"
#include "solvers/magnetostatic_solver_3d.hpp"

namespace fs = std::filesystem;
using complex = std::complex<double>;

ProblemConfig DecodeConfig(const json& config, const std::string& archive = {});

namespace {

using Matrix = std::vector<std::vector<double>>;

Matrix ReadMatrix(const std::string& archive, const std::string& quantity) {
	Matrix values;
	HighFive::File(archive, HighFive::File::ReadOnly)
		.getDataSet("/coupling/" + quantity + "/values").read(values);
	return values;
}

// R and L of a one-frequency MQS coupling archive, read from the sweep layout
// [frequency][row][column].
std::pair<double, double> ReadImpedance(const std::string& archive) {
	std::vector<Matrix> R, L;
	HighFive::File file(archive, HighFive::File::ReadOnly);
	file.getDataSet("/coupling/Resistance/values").read(R);
	file.getDataSet("/coupling/Inductance/values").read(L);
	return { R[0][0][0], L[0][0][0] };
}

// Value of a vector field at reference point @p ip of the element of @p T.
mfem::Vector Sample(mfem::VectorCoefficient& field, mfem::ElementTransformation& T,
					const mfem::IntegrationPoint& ip) {
	T.SetIntPoint(&ip);
	mfem::Vector value(field.GetVDim());
	field.Eval(value, T, ip);
	return value;
}

mfem::VectorCoefficient& DerivedVector(const FieldExportSet& fields, const std::string& name) {
	const auto& exported = fields.Fields();
	const auto field = std::find_if(exported.begin(), exported.end(),
		[&](const FieldExport& f) { return f.name == name; });
	REQUIRE(field != exported.end());
	REQUIRE(field->kind == FieldExport::Kind::DerivedVector);
	return *field->vector;
}

// A round wire of radius a in a coaxial air region out to radius b, in the
// plane: domain attribute 1 the wire, 2 the air, boundary attribute 1 the
// outer circle. Element size h_wire in the wire and h_air at the outer
// circle, refined to h_skin within skin of the wire's surface.
std::unique_ptr<mfem::Mesh> WireMesh(double a, double b, double h_wire, double h_air,
									 double h_skin, double skin) {
	std::ostringstream geo;
	geo << "SetFactory(\"OpenCASCADE\");\n"
		<< "a = " << a << "; b = " << b << ";\n"
		<< "Disk(1) = {0, 0, 0, a};\n"
		<< "Disk(2) = {0, 0, 0, b};\n"
		<< "BooleanFragments{ Surface{1, 2}; Delete; }{}\n"
		<< "wire[] = Surface In BoundingBox{-1.01*a, -1.01*a, -1, 1.01*a, 1.01*a, 1};\n"
		<< "air[] = Surface{:}; air[] -= wire[];\n"
		<< "surface[] = Abs(Boundary{ Surface{wire[]}; });\n"
		<< "outer[] = Abs(Boundary{ Surface{air[]}; }); outer[] -= surface[];\n"
		<< "Physical Surface(1) = {wire[]};\n"
		<< "Physical Surface(2) = {air[]};\n"
		<< "Physical Curve(1) = {outer[]};\n"
		<< "Field[1] = Distance; Field[1].CurvesList = {surface[]};\n"
		<< "Field[2] = Threshold; Field[2].InField = 1;\n"
		<< "Field[2].SizeMin = " << h_skin << "; Field[2].SizeMax = " << h_wire << ";\n"
		<< "Field[2].DistMin = " << skin << "; Field[2].DistMax = " << 2.0 * skin << ";\n"
		<< "Field[3] = Threshold; Field[3].InField = 1;\n"
		<< "Field[3].SizeMin = " << h_wire << "; Field[3].SizeMax = " << h_air << ";\n"
		<< "Field[3].DistMin = 0; Field[3].DistMax = b - a;\n"
		<< "Field[4] = Ball; Field[4].Radius = a; Field[4].VIn = " << h_wire
		<< "; Field[4].VOut = " << h_air << ";\n"
		<< "Field[5] = Min; Field[5].FieldsList = {2, 3, 4};\n"
		<< "Background Field = 5;\n"
		<< "Mesh.MeshSizeFromPoints = 0; Mesh.MeshSizeExtendFromBoundary = 0;\n"
		<< "Mesh.MeshSizeFromCurvature = 0;\n";
	return gmsh_fixture::Mesh("wire", geo.str(), 2, 2);
}

// The wire as a current terminal "Wire" with n x A = 0 (A = 0) on the outer
// circle. A massive wire conducts with sigma; a stranded one is given no
// conductivity.
json WireConfig(const std::string& physics, const std::string& conductor, double sigma) {
	return json{
		{"simulation", {
			{"physics_type", physics}, {"mesh", "unused.msh"}, {"order", 2},
			{"geometry_type", "planar"}, {"analysis_type", "coupling_matrix"},
			{"linear_solver", "direct"}, {"solver_print_level", 0}}},
		{"entity_groups", json::array({
			{{"name", "Wire"}, {"dim", 2}, {"attribute_ids", {1}}},
			{{"name", "Air"}, {"dim", 2}, {"attribute_ids", {2}}},
			{{"name", "Outer"}, {"dim", 1}, {"attribute_ids", {1}}}})},
		{"regions", json::array({
			{{"name", "Wire"}, {"entity_group", "Wire"}, {"material", "Metal"}},
			{{"name", "Air"}, {"entity_group", "Air"}, {"material", "Air"}}})},
		{"materials", json::array({
			{{"name", "Metal"}, {"properties", {{"mu_r", 1.0},
				{"sigma", conductor == "massive" ? sigma : 0.0}}}},
			{{"name", "Air"}, {"properties", {{"mu_r", 1.0}}}}})},
		{"terminals", json::array({
			{{"name", "Wire"}, {"quantity", "current"}, {"entity_group", "Wire"},
			 {"conductor_type", conductor}}})},
		{"boundary_conditions", json::array({
			{{"name", "Outer"}, {"type", "dirichlet"}, {"entity_group", "Outer"}, {"value", 0.0}}})},
		{"scenarios", json::array()}
	};
}

// One eighth of a sphere of radius a at the origin (x, y, z >= 0) in a
// quarter cylinder of radius Rw and height H, with a solenoid sector
// R1 <= r <= R2 over the full height. Domain attributes: 1 air, 2 sphere,
// 3 solenoid; boundary attributes: 1 the plane z = 0, 2 the plane z = H,
// 3 the planes x = 0 and y = 0, 4 the cylinder r = Rw. The planes through
// the sphere's center are exact symmetry planes for a field applied along z;
// the top and the cylinder truncate the model, imaging the sphere's dipole
// there, which perturbs the field at the sphere by about 7e-4.
struct SphereModel {
	double a = 0.01;
	double R1 = 0.06, R2 = 0.07, Rw = 0.08, H = 0.08;
};

std::unique_ptr<mfem::Mesh> SphereMesh(const SphereModel& m, double h_sphere, double h_far) {
	std::ostringstream geo;
	geo << "SetFactory(\"OpenCASCADE\");\n"
		<< "a = " << m.a << "; R1 = " << m.R1 << "; R2 = " << m.R2 << "; Rw = " << m.Rw
		<< "; H = " << m.H << "; e = 1e-3 * a;\n"
		<< "Cylinder(1) = {0, 0, 0, 0, 0, H, Rw, Pi/2};\n"
		<< "Cylinder(2) = {0, 0, 0, 0, 0, H, R2, Pi/2};\n"
		<< "Cylinder(3) = {0, 0, 0, 0, 0, H, R1, Pi/2};\n"
		<< "Sphere(4) = {0, 0, 0, a, 0, Pi/2, Pi/2};\n"
		<< "BooleanFragments{ Volume{1, 2, 3, 4}; Delete; }{}\n"
		<< "sphere[] = Volume In BoundingBox{-e, -e, -e, a + e, a + e, a + e};\n"
		<< "core[] = Volume In BoundingBox{-e, -e, -e, R2 + e, R2 + e, H + e};\n"
		<< "coil[] = core[]; coil[] -= Volume In BoundingBox{-e, -e, -e, R1 + e, R1 + e, H + e};\n"
		<< "air[] = Volume{:}; air[] -= sphere[]; air[] -= coil[];\n"
		<< "Physical Volume(1) = {air[]};\n"
		<< "Physical Volume(2) = {sphere[]};\n"
		<< "Physical Volume(3) = {coil[]};\n"
		<< "bottom[] = Surface In BoundingBox{-e, -e, -e, Rw + e, Rw + e, e};\n"
		<< "top[] = Surface In BoundingBox{-e, -e, H - e, Rw + e, Rw + e, H + e};\n"
		<< "planes[] = Surface In BoundingBox{-e, -e, -e, e, Rw + e, H + e};\n"
		<< "planes[] += Surface In BoundingBox{-e, -e, -e, Rw + e, e, H + e};\n"
		<< "wall[] = Abs(CombinedBoundary{ Volume{:}; });\n"
		<< "wall[] -= bottom[]; wall[] -= top[]; wall[] -= planes[];\n"
		<< "Physical Surface(1) = {bottom[]};\n"
		<< "Physical Surface(2) = {top[]};\n"
		<< "Physical Surface(3) = {planes[]};\n"
		<< "Physical Surface(4) = {wall[]};\n"
		<< "Field[1] = Distance; Field[1].SurfacesList = {Abs(Boundary{ Volume{sphere[]}; })};\n"
		<< "Field[2] = Threshold; Field[2].InField = 1;\n"
		<< "Field[2].SizeMin = " << h_sphere << "; Field[2].SizeMax = " << h_far << ";\n"
		<< "Field[2].DistMin = 0; Field[2].DistMax = 4 * a;\n"
		<< "Field[3] = Ball; Field[3].Radius = a; Field[3].VIn = " << h_sphere
		<< "; Field[3].VOut = " << h_far << ";\n"
		<< "Field[4] = Min; Field[4].FieldsList = {2, 3};\n"
		<< "Background Field = 4;\n"
		<< "Mesh.MeshSizeFromPoints = 0; Mesh.MeshSizeExtendFromBoundary = 0;\n"
		<< "Mesh.MeshSizeFromCurvature = 0;\n";
	return gmsh_fixture::Mesh("sphere", geo.str(), 3, 2);
}

// The values of a vector field at points inside the sphere, at most 0.6 a
// from its center.
std::vector<mfem::Vector> SampleInsideSphere(mfem::Mesh& mesh, const SphereModel& m,
											 mfem::VectorCoefficient& field) {
	const double fractions[][3] = { { 0.1, 0.1, 0.1 }, { 0.5, 0.1, 0.2 }, { 0.2, 0.4, 0.3 },
									{ 0.1, 0.2, 0.5 }, { 0.3, 0.3, 0.1 } };
	const int n = static_cast<int>(std::size(fractions));
	mfem::DenseMatrix points(3, n);
	for (int i = 0; i < n; ++i) {
		for (int d = 0; d < 3; ++d) { points(d, i) = fractions[i][d] * m.a; }
	}
	mfem::Array<int> elements;
	mfem::Array<mfem::IntegrationPoint> ips;
	REQUIRE(mesh.FindPoints(points, elements, ips) == n);
	std::vector<mfem::Vector> values;
	for (int i = 0; i < n; ++i) {
		values.push_back(Sample(field, *mesh.GetElementTransformation(elements[i]), ips[i]));
	}
	return values;
}

// The magnetic model of SphereModel: the solenoid a stranded azimuthal
// terminal "Coil", n x A = 0 on the planes x = 0 and y = 0, natural
// (n x H = 0) on z = 0, z = H and r = Rw. The natural planes z = 0 and z = H
// image the solenoid into an infinite one and the natural cylinder is an
// ideal return path, so with no sphere the field inside the winding is
// exactly H0 = I / H for ampere-turns I over the height H.
json SphereMagneticConfig(const std::string& physics, double mu_r, double sigma) {
	return json{
		{"simulation", {
			{"physics_type", physics}, {"mesh", "unused.msh"}, {"order", 2},
			{"geometry_type", "3d"}, {"analysis_type", "coupling_matrix"},
			{"linear_solver", "direct"}, {"solver_print_level", 0}}},
		{"entity_groups", json::array({
			{{"name", "Air"}, {"dim", 3}, {"attribute_ids", {1}}},
			{{"name", "Sphere"}, {"dim", 3}, {"attribute_ids", {2}}},
			{{"name", "Coil"}, {"dim", 3}, {"attribute_ids", {3}}},
			{{"name", "Planes"}, {"dim", 2}, {"attribute_ids", {3}}}})},
		{"regions", json::array({
			{{"name", "Air"}, {"entity_group", "Air"}, {"material", "Air"}},
			{{"name", "Sphere"}, {"entity_group", "Sphere"}, {"material", "Sphere"}},
			{{"name", "Coil"}, {"entity_group", "Coil"}, {"material", "Air"}}})},
		{"materials", json::array({
			{{"name", "Air"}, {"properties", {{"mu_r", 1.0}}}},
			{{"name", "Sphere"}, {"properties", {{"mu_r", mu_r}, {"sigma", sigma}}}}})},
		{"terminals", json::array({
			{{"name", "Coil"}, {"quantity", "current"}, {"entity_group", "Coil"},
			 {"conductor_type", "stranded"},
			 {"direction", {{"type", "azimuthal"}, {"origin", {0.0, 0.0, 0.0}},
							{"axis", {0.0, 0.0, 1.0}}}}}})},
		{"boundary_conditions", json::array({
			{{"name", "Planes"}, {"type", "dirichlet"}, {"entity_group", "Planes"}, {"value", 0.0}}})},
		{"scenarios", json::array()}
	};
}

} // namespace

// A straight round conductor of radius a carrying I, with the flux returning
// beyond radius b (A = 0 there): B is azimuthal,
//
//   B = mu0 I r / (2 pi a^2)   (r <= a),     B = mu0 I / (2 pi r)   (a <= r <= b),
//
// and the inductance per unit length is L' = mu0 / (8 pi) + mu0 ln(b/a) / (2 pi),
// the wire's internal part plus the coaxial gap's. A massive conductor's DC
// current is uniform here too (every path through it is equally long), so
// both kinds of terminal give the same answer. At second order B agrees to
// 1.2e-3 of its surface value at the element centers, and L' to 2e-6.
TEST_CASE("A round conductor's field and inductance match the closed form",
		  "[solvers][analytic][magnetostatic][planar][round]") {
	constexpr double a = 1e-3, b = 5e-3, I = 1.0;
	const std::string conductor = GENERATE("stranded", "massive");
	INFO(conductor);
	auto mesh = WireMesh(a, b, a / 6, b / 8, a / 6, a / 6);

	json config = WireConfig("magnetostatics", conductor, 5.8e7);
	{
		json field = config;
		field["simulation"]["analysis_type"] = "field";
		field["scenarios"] = json::array({{{"name", "I"},
			{"excitations", json::array({{{"terminal", "Wire"}, {"value", I}}})}}});
		mfem::Mesh copy(*mesh);
		MagnetostaticSolver solver(copy, DecodeConfig(field));
		solver.Setup();
		solver.Run();
		const FieldExportSet fields = solver.CollectExportFields();
		mfem::VectorCoefficient& B = DerivedVector(fields, "B");
		const double B_surface = Constants::MU_0 * I / (Constants::TWO_PI * a);
		double worst_phi = 0.0, worst_r = 0.0;
		mfem::IntegrationPoint center;
		center.Set2(1.0 / 3.0, 1.0 / 3.0);
		for (int e = 0; e < copy.GetNE(); ++e) {
			mfem::ElementTransformation& T = *copy.GetElementTransformation(e);
			mfem::Vector x(2);
			T.Transform(center, x);
			const double r = x.Norml2();
			const double exact = r <= a ? B_surface * r / a : B_surface * a / r;
			const mfem::Vector value = Sample(B, T, center);
			const double B_phi = (-x(1) * value(0) + x(0) * value(1)) / r;
			const double B_r = (x(0) * value(0) + x(1) * value(1)) / r;
			worst_phi = std::max(worst_phi, std::abs(B_phi - exact) / B_surface);
			worst_r = std::max(worst_r, std::abs(B_r) / B_surface);
		}
		INFO("worst error relative to the surface field: B_phi " << worst_phi << ", B_r " << worst_r);
		REQUIRE(worst_phi < 3e-3);
		REQUIRE(worst_r < 3e-3);
	}

	const std::string archive = "round_conductor.h5";
	MagnetostaticSolver solver(*mesh, DecodeConfig(config, archive));
	solver.Setup();
	solver.Run();
	solver.SaveAnalysis();
	const double L = ReadMatrix(archive, "Inductance")[0][0];
	fs::remove(archive);
	const double exact = Constants::MU_0 / (4.0 * Constants::TWO_PI) +
		Constants::MU_0 * std::log(b / a) / Constants::TWO_PI;
	INFO("relative error " << L / exact - 1.0);
	REQUIRE(L == Catch::Approx(exact).epsilon(1e-5));
}

// The same conductor in 3D: a bar of radius a and length l between electrodes
// on its end faces, in a coaxial return at radius b, every wall n x A = 0.
// The field is the planar one (A = A_z(r) z-hat satisfies n x A = 0 on the
// end walls, and on r = b with A_z(b) = 0), so L = l L'. Massive, with the DC
// distribution solved between the electrodes, which is uniform here.
TEST_CASE("A round bar's 3D inductance matches the closed form",
		  "[solvers][analytic][magnetostatic][3d][round]") {
	constexpr double a = 1e-3, b = 3e-3, l = 2e-3;
	std::ostringstream geo;
	geo << "SetFactory(\"OpenCASCADE\");\n"
		<< "a = " << a << "; b = " << b << "; l = " << l << ";\n"
		<< "Cylinder(1) = {0, 0, 0, 0, 0, l, a};\n"
		<< "Cylinder(2) = {0, 0, 0, 0, 0, l, b};\n"
		<< "BooleanFragments{ Volume{1, 2}; Delete; }{}\n"
		<< "bar[] = Volume In BoundingBox{-1.01*a, -1.01*a, -1, 1.01*a, 1.01*a, 1};\n"
		<< "air[] = Volume{:}; air[] -= bar[];\n"
		<< "bottom[] = Surface In BoundingBox{-1.01*a, -1.01*a, -0.01*l, 1.01*a, 1.01*a, 0.01*l};\n"
		<< "top[] = Surface In BoundingBox{-1.01*a, -1.01*a, 0.99*l, 1.01*a, 1.01*a, 1.01*l};\n"
		<< "walls[] = Abs(CombinedBoundary{ Volume{:}; }); walls[] -= bottom[]; walls[] -= top[];\n"
		<< "Physical Volume(1) = {bar[]};\n"
		<< "Physical Volume(2) = {air[]};\n"
		<< "Physical Surface(1) = {bottom[]};\n"
		<< "Physical Surface(2) = {top[]};\n"
		<< "Physical Surface(3) = {walls[]};\n"
		<< "Mesh.MeshSizeMax = " << a / 2 << ";\n";
	auto mesh = gmsh_fixture::Mesh("bar3d", geo.str(), 3, 2);

	const json config = {
		{"simulation", {
			{"physics_type", "magnetostatics"}, {"mesh", "unused.msh"}, {"order", 2},
			{"geometry_type", "3d"}, {"analysis_type", "coupling_matrix"},
			{"linear_solver", "direct"}, {"solver_print_level", 0}}},
		{"entity_groups", json::array({
			{{"name", "Bar"}, {"dim", 3}, {"attribute_ids", {1}}},
			{{"name", "Air"}, {"dim", 3}, {"attribute_ids", {2}}},
			{{"name", "Bottom"}, {"dim", 2}, {"attribute_ids", {1}}},
			{{"name", "Top"}, {"dim", 2}, {"attribute_ids", {2}}},
			{{"name", "Walls"}, {"dim", 2}, {"attribute_ids", {1, 2, 3}}}})},
		{"regions", json::array({
			{{"name", "Bar"}, {"entity_group", "Bar"}, {"material", "Copper"}},
			{{"name", "Air"}, {"entity_group", "Air"}, {"material", "Air"}}})},
		{"materials", json::array({
			{{"name", "Copper"}, {"properties", {{"mu_r", 1.0}, {"sigma", 5.8e7}}}},
			{{"name", "Air"}, {"properties", {{"mu_r", 1.0}}}}})},
		{"terminals", json::array({
			{{"name", "Bar"}, {"quantity", "current"}, {"entity_group", "Bar"},
			 {"conductor_type", "massive"},
			 {"direction", {{"type", "electrodes"}, {"input", "Bottom"}, {"output", "Top"}}}}})},
		{"boundary_conditions", json::array({
			{{"name", "Walls"}, {"type", "dirichlet"}, {"entity_group", "Walls"}, {"value", 0.0}}})},
		{"scenarios", json::array()}
	};
	const std::string archive = "round_bar_3d.h5";
	MagnetostaticSolver3D solver(*mesh, DecodeConfig(config, archive));
	solver.Setup();
	solver.Run();
	solver.SaveAnalysis();
	const double L = ReadMatrix(archive, "Inductance")[0][0];
	fs::remove(archive);
	const double exact = l * (Constants::MU_0 / (4.0 * Constants::TWO_PI) +
		Constants::MU_0 * std::log(b / a) / Constants::TWO_PI);
	INFO("relative error " << L / exact - 1.0);
	REQUIRE(L == Catch::Approx(exact).epsilon(5e-4));
}

// Skin effect in a round wire (time dependence exp(j omega t)): E_z = C J0(k r)
// with k^2 = -j omega mu0 sigma, so the internal impedance per unit length is
//
//   Z_int = k J0(k a) / (2 pi a sigma J1(k a)),
//
// which falls to 1 / (pi a^2 sigma) at DC. The air gap out to b adds the
// reactance j omega mu0 ln(b/a) / (2 pi). Swept from a = 0.5 to 4 skin depths,
// where R rises to 2.3 times its DC value and the internal inductance falls
// to a third. The mesh is graded to a quarter of the smallest skin depth at
// the wire's surface.
TEST_CASE("A round wire's AC impedance matches the Bessel solution",
		  "[solvers][analytic][mqs][planar][round][skin]") {
	constexpr double a = 1e-3, b = 5e-3, sigma = 5.8e7;
	const double ratio = GENERATE(0.5, 1.0, 2.0, 4.0);  // a / skin depth
	const double delta = a / ratio;
	const double omega = 2.0 / (Constants::MU_0 * sigma * delta * delta);
	INFO("a / delta = " << ratio);
	auto mesh = WireMesh(a, b, a / 5, b / 6, a / 16, a / 3);

	json config = WireConfig("magnetoquasistatics", "massive", sigma);
	config["scenarios"] = json::array({{{"name", "f"}, {"frequency", omega / Constants::TWO_PI},
										{"excitations", json::array()}}});
	const std::string archive = "round_wire_ac.h5";
	MagnetoquasistaticSolver solver(*mesh, DecodeConfig(config, archive));
	solver.Setup();
	solver.Run();
	solver.SaveAnalysis();
	const auto [R, L] = ReadImpedance(archive);
	fs::remove(archive);

	const complex k = complex(1.0, -1.0) / delta;
	const complex Z_int = k * bessel::J0(k * a) / (Constants::TWO_PI * a * sigma * bessel::J1(k * a));
	const double L_gap = Constants::MU_0 * std::log(b / a) / Constants::TWO_PI;
	const double R_dc = 1.0 / (0.5 * Constants::TWO_PI * a * a * sigma);
	INFO("R / R_dc " << R / R_dc << " (exact " << Z_int.real() / R_dc << "), internal L "
		 << L - L_gap << " (exact " << Z_int.imag() / omega << ")");
	REQUIRE(R == Catch::Approx(Z_int.real()).epsilon(1e-5));
	REQUIRE(L - L_gap == Catch::Approx(Z_int.imag() / omega).epsilon(2e-5));
}

TEST_CASE("Complex Bessel series agree with the standard library and the Wronskian",
		  "[analytic][bessel]") {
	for (const double x : { 0.1, 1.0, 3.7, 8.0 }) {
		INFO("x = " << x);
		REQUIRE(bessel::J0(x).real() == Catch::Approx(std::cyl_bessel_j(0.0, x)).epsilon(1e-12));
		REQUIRE(bessel::J1(x).real() == Catch::Approx(std::cyl_bessel_j(1.0, x)).epsilon(1e-12));
		REQUIRE(bessel::I0(x).real() == Catch::Approx(std::cyl_bessel_i(0.0, x)).epsilon(1e-12));
		REQUIRE(bessel::I1(x).real() == Catch::Approx(std::cyl_bessel_i(1.0, x)).epsilon(1e-12));
		REQUIRE(bessel::K0(x).real() == Catch::Approx(std::cyl_bessel_k(0.0, x)).epsilon(1e-8));
		REQUIRE(bessel::K1(x).real() == Catch::Approx(std::cyl_bessel_k(1.0, x)).epsilon(1e-8));
	}
	// I0 K1 + I1 K0 = 1/z, and J0(z) = I0(-j z), off the real axis.
	for (const complex z : { complex(0.3, 0.3), complex(2.0, 2.0), complex(5.0, 5.0), complex(1.0, -4.0) }) {
		INFO("z = " << z);
		const complex w = bessel::I0(z) * bessel::K1(z) + bessel::I1(z) * bessel::K0(z);
		REQUIRE(std::abs(w * z - 1.0) < 1e-8);
		REQUIRE(std::abs(bessel::J0(z) - bessel::I0(complex(0.0, -1.0) * z)) < 1e-12 * std::abs(bessel::J0(z)));
	}
}

// A long conducting tube a <= r <= b in an axial AC field, axisymmetric: the
// field is H_z(r) alone, uniform in the bore and outside the tube, and in the
// wall H'' + H'/r = gamma^2 H with gamma^2 = j omega mu0 sigma. Faraday's law
// around the bore gives E_phi(a) = -j omega mu0 a H_in / 2, so with
// J_phi = -dH_z/dr = sigma E_phi the wall's inner edge has H(a) = H_in and
// H'(a) = gamma^2 a H_in / 2. Solving with I0, K0 (W = I0 K1 + I1 K0 = 1/z):
//
//   H(r) / H_in = C1 I0(gamma r) + C2 K0(gamma r),
//   C1 = z (K1(z) + (z/2) K0(z)),   C2 = z (I1(z) - (z/2) I0(z)),   z = gamma a,
//
// and the shielding ratio is H_in / H_out = 1 / H(b). The model runs from the
// axis to r = R > b with natural boundaries top and bottom (an infinitely long
// tube) and a constant A_phi on r = R, which sets the flux; the ratio of the
// bore field to the field outside the wall is independent of it. Swept from a
// quarter to two skin depths of wall thickness, where the bore field falls
// to 4.7% of the outside field and lags it by 152 degrees. The order-3
// solution agrees with the closed form to 1e-8.
TEST_CASE("A conducting tube shields an axial AC field as the closed form says",
		  "[solvers][analytic][mqs][axisymmetric][shield]") {
	constexpr double a = 4e-3, b = 5e-3, R = 6e-3, height = 1e-4, sigma = 5.8e7;
	constexpr int nr = 60;  // 0.1 mm cells: 40 in the bore, 10 in the wall, 10 outside
	const double thickness_ratio = GENERATE(0.25, 0.5, 1.0, 2.0);  // (b - a) / skin depth
	const double delta = (b - a) / thickness_ratio;
	const double omega = 2.0 / (Constants::MU_0 * sigma * delta * delta);
	INFO("(b - a) / delta = " << thickness_ratio);

	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(nr, 1, mfem::Element::QUADRILATERAL,
												  true, R, height);
	for (int e = 0; e < mesh.GetNE(); ++e) {
		mfem::Vector c;
		mesh.GetElementCenter(e, c);
		mesh.SetAttribute(e, c(0) > a && c(0) < b ? 2 : 1);
	}
	mesh.SetAttributes();

	// MakeCartesian2D's boundary attributes: 1 bottom, 2 r = R, 3 top, 4 axis.
	const json config = {
		{"simulation", {
			{"physics_type", "magnetoquasistatics"}, {"mesh", "unused.mesh"}, {"order", 3},
			{"geometry_type", "axisymmetric"}, {"analysis_type", "field"},
			{"linear_solver", "direct"}, {"solver_print_level", 0}}},
		{"entity_groups", json::array({
			{{"name", "Air"}, {"dim", 2}, {"attribute_ids", {1}}},
			{{"name", "Tube"}, {"dim", 2}, {"attribute_ids", {2}}},
			{{"name", "Outer"}, {"dim", 1}, {"attribute_ids", {2}}}})},
		{"regions", json::array({
			{{"name", "Air"}, {"entity_group", "Air"}, {"material", "Air"}},
			{{"name", "Tube"}, {"entity_group", "Tube"}, {"material", "Copper"}}})},
		{"materials", json::array({
			{{"name", "Air"}, {"properties", {{"mu_r", 1.0}}}},
			{{"name", "Copper"}, {"properties", {{"mu_r", 1.0}, {"sigma", sigma}}}}})},
		{"terminals", json::array()},
		{"boundary_conditions", json::array({
			{{"name", "Outer"}, {"type", "dirichlet"}, {"entity_group", "Outer"}, {"value", 1e-6}}})},
		{"scenarios", json::array({{{"name", "f"}, {"frequency", omega / Constants::TWO_PI},
									{"excitations", json::array()}}})}
	};
	MagnetoquasistaticSolver solver(mesh, DecodeConfig(config));
	solver.Setup();
	solver.Run();
	const FieldExportSet fields = solver.CollectExportFields();
	mfem::VectorCoefficient& B_re = DerivedVector(fields, "B_Real");
	mfem::VectorCoefficient& B_im = DerivedVector(fields, "B_Imag");

	// B_z at the center of every bore and every outside cell.
	std::vector<complex> bore, outside;
	mfem::IntegrationPoint center;
	center.Set2(0.5, 0.5);
	for (int e = 0; e < mesh.GetNE(); ++e) {
		if (mesh.GetAttribute(e) == 2) continue;
		mfem::ElementTransformation& T = *mesh.GetElementTransformation(e);
		mfem::Vector x(2);
		T.Transform(center, x);
		const complex Bz(Sample(B_re, T, center)(1), Sample(B_im, T, center)(1));
		(x(0) < a ? bore : outside).push_back(Bz);
	}
	auto spread = [](const std::vector<complex>& values) {
		double worst = 0.0;
		for (const complex& v : values) { worst = std::max(worst, std::abs(v - values.front())); }
		return worst / std::abs(values.front());
	};
	const complex z = complex(1.0, 1.0) / delta * a;
	const complex zb = complex(1.0, 1.0) / delta * b;
	const complex C1 = z * (bessel::K1(z) + 0.5 * z * bessel::K0(z));
	const complex C2 = z * (bessel::I1(z) - 0.5 * z * bessel::I0(z));
	const complex exact = 1.0 / (C1 * bessel::I0(zb) + C2 * bessel::K0(zb));
	const complex computed = bore.front() / outside.front();
	INFO("H_in / H_out " << computed << " (exact " << exact << "), spread in the bore "
		 << spread(bore) << ", outside " << spread(outside));
	REQUIRE(spread(bore) < 1e-8);
	REQUIRE(spread(outside) < 1e-8);
	INFO("relative error " << std::abs(computed - exact) / std::abs(exact));
	REQUIRE(std::abs(computed - exact) < 5e-8 * std::abs(exact));
}

// A dielectric sphere (eps_r) in a uniform field E0 along z: the field inside
// is uniform, E_in = 3 E0 / (eps_r + 2), and outside it adds a dipole
// p = 4 pi eps0 a^3 beta E0, beta = (eps_r - 1) / (eps_r + 2). In the eighth
// model the field is applied by V = 0 on z = 0 (the antisymmetry plane) and
// V0 on the plate z = H, natural on the sides, so E0 = V0 / H; the dipole adds
// the charge p / (8 H) to the plate, and so the capacitance
// dC = 4 pi eps0 a^3 beta / (8 H^2), measured as the difference with and
// without the sphere on one mesh. Both agree with the closed form to 1e-3,
// the discretization and the truncation contributing comparably.
TEST_CASE("A dielectric sphere in a uniform field matches the closed form",
		  "[solvers][analytic][electrostatic][3d][sphere]") {
	constexpr double eps_r = 4.0, V0 = 1.0;
	const SphereModel m;
	auto mesh = SphereMesh(m, m.a / 6, m.Rw / 5);
	auto config = [&](double eps, const std::string& analysis) {
		return json{
			{"simulation", {
				{"physics_type", "electrostatics"}, {"mesh", "unused.msh"}, {"order", 2},
				{"geometry_type", "3d"}, {"analysis_type", analysis},
				{"solver_tolerance", 1e-12}, {"solver_print_level", 0}}},
			{"entity_groups", json::array({
				{{"name", "Air"}, {"dim", 3}, {"attribute_ids", {1, 3}}},
				{{"name", "Sphere"}, {"dim", 3}, {"attribute_ids", {2}}},
				{{"name", "Bottom"}, {"dim", 2}, {"attribute_ids", {1}}},
				{{"name", "Top"}, {"dim", 2}, {"attribute_ids", {2}}}})},
			{"regions", json::array({
				{{"name", "Air"}, {"entity_group", "Air"}, {"material", "Air"}},
				{{"name", "Sphere"}, {"entity_group", "Sphere"}, {"material", "Sphere"}}})},
			{"materials", json::array({
				{{"name", "Air"}, {"properties", {{"epsilon_r", 1.0}}}},
				{{"name", "Sphere"}, {"properties", {{"epsilon_r", eps}}}}})},
			{"terminals", json::array({
				{{"name", "Top"}, {"quantity", "voltage"}, {"entity_group", "Top"}}})},
			{"boundary_conditions", json::array({
				{{"name", "Bottom"}, {"type", "dirichlet"}, {"entity_group", "Bottom"}, {"value", 0.0}}})},
			{"scenarios", json::array({{{"name", "V0"},
				{"excitations", json::array({{{"terminal", "Top"}, {"value", V0}}})}}})}
		};
	};
	auto capacitance = [&](double eps) {
		mfem::Mesh copy(*mesh);
		const std::string archive = "dielectric_sphere.h5";
		ElectrostaticSolver solver(copy, DecodeConfig(config(eps, "coupling_matrix"), archive));
		solver.Setup();
		solver.Run();
		solver.SaveAnalysis();
		const double C = ReadMatrix(archive, "Capacitance")[0][0];
		fs::remove(archive);
		return C;
	};

	const double E0 = V0 / m.H, beta = (eps_r - 1.0) / (eps_r + 2.0);
	{
		mfem::Mesh copy(*mesh);
		ElectrostaticSolver solver(copy, DecodeConfig(config(eps_r, "field")));
		solver.Setup();
		solver.Run();
		const FieldExportSet fields = solver.CollectExportFields();
		// V rises toward the plate, so E points along -z.
		const double E_in = -3.0 * E0 / (eps_r + 2.0);
		for (const mfem::Vector& E : SampleInsideSphere(copy, m, DerivedVector(fields, "E"))) {
			INFO("E inside " << E(0) << ", " << E(1) << ", " << E(2) << " (exact 0, 0, " << E_in << ")");
			REQUIRE(E(2) == Catch::Approx(E_in).epsilon(3e-3));
			REQUIRE(std::hypot(E(0), E(1)) < -3e-3 * E_in);
		}
	}
	const double dC = capacitance(eps_r) - capacitance(1.0);
	const double exact = 2.0 * Constants::TWO_PI * Constants::EPSILON_0 * std::pow(m.a, 3) * beta /
		(8.0 * m.H * m.H);
	INFO("dC " << dC << " (exact " << exact << "), relative error " << dC / exact - 1.0);
	REQUIRE(dC == Catch::Approx(exact).epsilon(3e-3));
}

// A permeable sphere (mu_r) in a uniform field H0: inside, B is uniform,
// B_in = 3 mu_r mu0 H0 / (mu_r + 2), and outside it adds the dipole
// m = 4 pi a^3 beta H0, beta = (mu_r - 1) / (mu_r + 2). In the eighth model
// the solenoid applies H0 = I / H and the sphere raises its inductance by
// dL = mu0 4 pi a^3 beta H0^2 / (8 I^2). Both agree with the closed form to
// 2e-4 on this mesh; refined, the error settles at the truncation's 7e-4.
TEST_CASE("A permeable sphere in a uniform field matches the closed form",
		  "[solvers][analytic][magnetostatic][3d][sphere]") {
	constexpr double mu_r = 10.0, I = 1.0;
	const SphereModel m;
	auto mesh = SphereMesh(m, m.a / 4, m.Rw / 5);
	const double H0 = I / m.H, beta = (mu_r - 1.0) / (mu_r + 2.0);

	auto inductance = [&](double mu) {
		mfem::Mesh copy(*mesh);
		const std::string archive = "permeable_sphere.h5";
		MagnetostaticSolver3D solver(copy, DecodeConfig(SphereMagneticConfig("magnetostatics", mu, 0.0), archive));
		solver.Setup();
		solver.Run();
		solver.SaveAnalysis();
		const double L = ReadMatrix(archive, "Inductance")[0][0];
		fs::remove(archive);
		return L;
	};
	for (const double mu : { 1.0, mu_r }) {
		json config = SphereMagneticConfig("magnetostatics", mu, 0.0);
		config["simulation"]["analysis_type"] = "field";
		config["scenarios"] = json::array({{{"name", "I"},
			{"excitations", json::array({{{"terminal", "Coil"}, {"value", I}}})}}});
		mfem::Mesh copy(*mesh);
		MagnetostaticSolver3D solver(copy, DecodeConfig(config));
		solver.Setup();
		solver.Run();
		const FieldExportSet fields = solver.CollectExportFields();
		const double B_in = Constants::MU_0 * H0 * (mu == 1.0 ? 1.0 : 3.0 * mu / (mu + 2.0));
		for (const mfem::Vector& B : SampleInsideSphere(copy, m, DerivedVector(fields, "B"))) {
			INFO("mu_r " << mu << ": B inside " << B(0) << ", " << B(1) << ", " << B(2)
				 << " (exact 0, 0, " << B_in << ")");
			REQUIRE(B(2) == Catch::Approx(B_in).epsilon(2e-3));
			REQUIRE(std::hypot(B(0), B(1)) < 2e-3 * B_in);
		}
	}
	const double dL = inductance(mu_r) - inductance(1.0);
	const double exact = Constants::MU_0 * 2.0 * Constants::TWO_PI * std::pow(m.a, 3) * beta *
		H0 * H0 / (8.0 * I * I);
	INFO("dL " << dL << " (exact " << exact << "), relative error " << dL / exact - 1.0);
	REQUIRE(dL == Catch::Approx(exact).epsilon(2e-3));
}

// A conducting sphere (sigma, mu0) in a uniform AC field H0 exp(j omega t):
// its eddy currents give it the dipole moment m = 4 pi a^3 beta H0, with
//
//   beta = -(1/2) [1 - 3 / (k a)^2 + 3 cot(k a) / (k a)],   k = (1 - j) / delta,
//
// which tends to -j omega mu0 sigma a^2 / 30 at low frequency and to -1/2 (a
// perfect diamagnet) at high frequency (Landau and Lifshitz, Electrodynamics
// of Continuous Media, section 59). The solenoid of the eighth model sees the
// impedance change dZ = j omega mu0 4 pi a^3 beta H0^2 / (8 I^2): its real
// part is the sphere's eddy-current loss, its imaginary part the inductance
// the sphere's screening removes. Measured as the difference with and without
// the sphere's conductivity on one mesh, at one and three skin depths, where
// it agrees with the closed form to 1e-4 and 3e-4.
TEST_CASE("A conducting sphere in a uniform AC field matches the closed form",
		  "[solvers][analytic][mqs][3d][sphere]") {
	constexpr double sigma = 5.8e7, I = 1.0;
	const double ratio = GENERATE(1.0, 3.0);  // a / skin depth
	INFO("a / delta = " << ratio);
	const SphereModel m;
	const double delta = m.a / ratio;
	const double omega = 2.0 / (Constants::MU_0 * sigma * delta * delta);
	auto mesh = SphereMesh(m, std::min(m.a / 4, delta / 2), m.Rw / 5);

	auto impedance = [&](double s) {
		json config = SphereMagneticConfig("magnetoquasistatics", 1.0, s);
		config["scenarios"] = json::array({{{"name", "f"}, {"frequency", omega / Constants::TWO_PI},
											{"excitations", json::array()}}});
		mfem::Mesh copy(*mesh);
		const std::string archive = "conducting_sphere.h5";
		MagnetoquasistaticSolver3D solver(copy, DecodeConfig(config, archive));
		solver.Setup();
		solver.Run();
		solver.SaveAnalysis();
		const auto [R, L] = ReadImpedance(archive);
		fs::remove(archive);
		return complex(R, omega * L);
	};
	const complex dZ = impedance(sigma) - impedance(0.0);

	const complex ka = complex(1.0, -1.0) * ratio;
	const complex beta = -0.5 * (1.0 - 3.0 / (ka * ka) + 3.0 * std::cos(ka) / (std::sin(ka) * ka));
	const double H0 = I / m.H;
	const complex exact = complex(0.0, omega * Constants::MU_0) * 2.0 * Constants::TWO_PI *
		std::pow(m.a, 3) * beta * H0 * H0 / (8.0 * I * I);
	INFO("dZ " << dZ << " (exact " << exact << "), relative error " << std::abs(dZ - exact) / std::abs(exact));
	REQUIRE(std::abs(dZ - exact) < 2e-3 * std::abs(exact));
}
