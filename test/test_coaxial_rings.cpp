// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// Two coaxial rings of round wire against their free-space inductances
// (coaxial_rings_reference.hpp): axisymmetric and 3D magnetostatics, and
// axisymmetric magnetoquasistatics in its DC limit.
//
// The models are bounded by a sphere of radius D, the references are for
// free space. The two homogeneous conditions on that sphere bracket the free-
// space value from either side: n x A = 0 (flux tangent, B . n = 0) holds the
// returning flux in and lowers every inductance, n x H = 0 (flux normal, an
// infinitely permeable wall) raises it. For a dipole moment m at the centre
// the walls add a uniform field of -2 and +1 times mu0 m / (4 pi D^3), so
// (L_tangent + 2 L_normal) / 3 cancels the leading O((R/D)^3) truncation
// error, leaving O((R/D)^5) and the discretization error.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

#include "coaxial_rings_reference.hpp"
#include "config/input_parser.hpp"
#include "io/mesh_loader.hpp"
#include "solvers/magnetoquasistatic_solver.hpp"
#include "solvers/magnetostatic_solver.hpp"
#include "solvers/magnetostatic_solver_3d.hpp"

namespace fs = std::filesystem;
using namespace coaxial_rings;

ProblemConfig DecodeConfig(const json& config, const std::string& archive = {});

namespace {

using Matrix = std::vector<std::vector<double>>;

// test/data/coaxial_rings_axi.msh and coaxial_rings.msh (D = 0.25 m; see
// their .geo scripts). The 3D mesh is one quadrant: its inductances are a
// quarter of the rings'.
const Ring kRing1{ 0.04, -0.01, 0.01 }, kRing2{ 0.07, 0.01, 0.01 };
constexpr double kCopper = 5.8e7;
constexpr double kQuadrants = 4.0;

std::string DataFile(const std::string& name) {
	return std::string(MFEM_ELECTROMAG_TEST_DATA) + "/" + name;
}

enum class Wall { FluxTangent, FluxNormal };  // n x A = 0, n x H = 0

// Ring k is terminal "R<k>" of conductor type types[k-1]; a massive ring is
// copper.
json RingsConfig(bool three_d, const std::string& physics, Wall wall,
				 const std::vector<std::string>& types, int order) {
	const int dim = three_d ? 3 : 2;
	json groups = json::array({
		{{"name", "Air"}, {"dim", dim}, {"attribute_ids", {1}}},
		{{"name", "Ring1"}, {"dim", dim}, {"attribute_ids", {2}}},
		{{"name", "Ring2"}, {"dim", dim}, {"attribute_ids", {3}}},
		{{"name", "Outer"}, {"dim", dim - 1}, {"attribute_ids", {1}}}});
	json bcs = json::array();
	if (three_d) {
		groups.push_back({{"name", "Meridian"}, {"dim", 2}, {"attribute_ids", {2}}});
		bcs.push_back({{"name", "Meridian"}, {"type", "dirichlet"},
					   {"entity_group", "Meridian"}, {"value", 0.0}});
	}
	if (wall == Wall::FluxTangent) {
		bcs.push_back({{"name", "Outer"}, {"type", "dirichlet"},
					   {"entity_group", "Outer"}, {"value", 0.0}});
	}
	json regions = json::array({{{"name", "Air"}, {"entity_group", "Air"}, {"material", "Air"}}});
	json terminals = json::array();
	for (int k = 1; k <= 2; ++k) {
		const std::string ring = "Ring" + std::to_string(k);
		const bool massive = types[k - 1] == "massive";
		regions.push_back({{"name", ring}, {"entity_group", ring},
						   {"material", massive ? "Copper" : "Air"}});
		json terminal = {{"name", "R" + std::to_string(k)}, {"quantity", "current"},
						 {"entity_group", ring}, {"conductor_type", types[k - 1]}};
		if (three_d) {
			terminal["direction"] = {{"type", "azimuthal"}, {"origin", {0.0, 0.0, 0.0}},
									 {"axis", {0.0, 0.0, 1.0}}};
		}
		terminals.push_back(terminal);
	}
	return json{
		{"simulation", {
			{"physics_type", physics},
			{"mesh", DataFile(three_d ? "coaxial_rings.msh" : "coaxial_rings_axi.msh")},
			{"order", order}, {"geometry_type", three_d ? "3d" : "axisymmetric"},
			{"analysis_type", "coupling_matrix"}, {"linear_solver", "direct"}
		}},
		{"entity_groups", groups},
		{"regions", regions},
		{"materials", json::array({
			{{"name", "Air"}, {"properties", {{"mu_r", 1.0}}}},
			{{"name", "Copper"}, {"properties", {{"mu_r", 1.0}, {"sigma", kCopper}}}}})},
		{"terminals", terminals},
		{"boundary_conditions", bcs},
		{"scenarios", json::array()}
	};
}

// The inductance matrix of a coupling run [H]; for MQS, at its first
// frequency, together with the resistance matrix.
struct Coupling { Matrix L, R; };

template <class Solver>
Coupling Solve(const json& config) {
	const std::string archive = "coaxial_rings.h5";
	const auto mesh = mesh_io::LoadMesh(config["simulation"]["mesh"].get<std::string>());
	Solver solver(*mesh, DecodeConfig(config, archive));
	solver.Setup();
	solver.Run();
	solver.SaveAnalysis();
	Coupling c;
	HighFive::File file(archive, HighFive::File::ReadOnly);
	if (config["simulation"]["physics_type"] == "magnetoquasistatics") {
		std::vector<Matrix> L, R;
		file.getDataSet("/coupling/Inductance/values").read(L);
		file.getDataSet("/coupling/Resistance/values").read(R);
		c.L = L[0];
		c.R = R[0];
	} else {
		file.getDataSet("/coupling/Inductance/values").read(c.L);
	}
	fs::remove(archive);
	return c;
}

// Every entry of the free-space matrix @p exact lies strictly between the
// two walls' values (scaled by @p scale), and their extrapolation
// (tangent + 2 normal) / 3 matches it to @p tolerance.
void RequireBracketed(const Matrix& tangent, const Matrix& normal, const Matrix& exact,
					  double scale, double tolerance) {
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			const double low = scale * tangent[i][j], high = scale * normal[i][j];
			const double extrapolated = (low + 2.0 * high) / 3.0;
			INFO("L(" << i << "," << j << "): n x A = 0 " << low << ", n x H = 0 " << high
				 << ", extrapolated " << extrapolated << ", free space " << exact[i][j]);
			REQUIRE(low < exact[i][j]);
			REQUIRE(exact[i][j] < high);
			REQUIRE(extrapolated == Catch::Approx(exact[i][j]).epsilon(tolerance));
		}
	}
}

} // namespace

TEST_CASE("The coaxial-ring references are converged and match the round-wire series",
		  "[analytic][magnetostatic]") {
	// Thin enough for the second-order series to be exact to ~1e-8.
	const Ring thin{ 1.0, 0.0, 0.05 };
	REQUIRE(Self(thin, Uniform) == Catch::Approx(SelfUniformSeries(thin)).epsilon(1e-6));
	REQUIRE(Self(kRing1, Uniform, 24) == Catch::Approx(Self(kRing1, Uniform, 32)).epsilon(2e-6));
	REQUIRE(Mutual(kRing1, Uniform, kRing2, Uniform, 16) ==
			Catch::Approx(Mutual(kRing1, Uniform, kRing2, Uniform, 24)).epsilon(1e-10));
	// Thick rings: the centre filaments are a poor stand-in.
	REQUIRE(Mutual(kRing1, Uniform, kRing2, Uniform) /
			FilamentMutual(kRing1.R, kRing1.Z, kRing2.R, kRing2.Z) - 1.0 > 0.01);
}

// Two stranded rings in the axisymmetric model, at order 3: the walls
// bracket the free-space matrix by up to 2.3%, and their extrapolation
// matches it to about 1e-5.
TEST_CASE("Axisymmetric ring inductances bracket and extrapolate to free space",
		  "[solvers][analytic][magnetostatic][axisymmetric]") {
	const std::vector<std::string> stranded = { "stranded", "stranded" };
	const Coupling tangent = Solve<MagnetostaticSolver>(
		RingsConfig(false, "magnetostatics", Wall::FluxTangent, stranded, 3));
	const Coupling normal = Solve<MagnetostaticSolver>(
		RingsConfig(false, "magnetostatics", Wall::FluxNormal, stranded, 3));
	const double M = Mutual(kRing1, Uniform, kRing2, Uniform);
	const Matrix exact = { { Self(kRing1, Uniform), M }, { M, Self(kRing2, Uniform) } };
	RequireBracketed(tangent.L, normal.L, exact, 1.0, 5e-5);
}

// The same rings in 3D, one quadrant bounded by the meridian planes (n x A =
// 0, which the azimuthal current crosses normally), at order 2: ring 1
// stranded (uniform current), ring 2 massive (its DC distribution, J ~ 1/r),
// so the mutual term couples the two shapes. The quadrant's sector rings are
// recognized from their ends on the meridian planes; a wrong angular extent
// would scale every entry by the square of its error.
TEST_CASE("3D ring inductances bracket and extrapolate to free space",
		  "[solvers][analytic][magnetostatic][3d]") {
	const std::vector<std::string> types = { "stranded", "massive" };
	const Coupling tangent = Solve<MagnetostaticSolver3D>(
		RingsConfig(true, "magnetostatics", Wall::FluxTangent, types, 2));
	const Coupling normal = Solve<MagnetostaticSolver3D>(
		RingsConfig(true, "magnetostatics", Wall::FluxNormal, types, 2));
	const double M = Mutual(kRing1, Uniform, kRing2, DcMassive);
	const Matrix exact = { { Self(kRing1, Uniform), M }, { M, Self(kRing2, DcMassive) } };
	RequireBracketed(tangent.L, normal.L, exact, kQuadrants, 2e-3);
}

// Massive copper rings at 1 mHz, where omega mu sigma a^2 = 5e-5: each is a
// DC resistor, R = 1 / G with G = sigma (R - sqrt(R^2 - a^2)), the two do
// not couple resistively, and the inductances are those of the DC current
// distributions (J ~ 1/r), which differ from the uniform ones by up to 2.5%.
TEST_CASE("Axisymmetric MQS rings reach the DC limit",
		  "[solvers][analytic][mqs][axisymmetric]") {
	const std::vector<std::string> massive = { "massive", "massive" };
	auto mqs = [&](Wall wall) {
		json config = RingsConfig(false, "magnetoquasistatics", wall, massive, 3);
		config["scenarios"] = json::array({{{"name", "DC"}, {"frequency", 1e-3},
											{"excitations", json::array()}}});
		return Solve<MagnetoquasistaticSolver>(config);
	};
	const Coupling tangent = mqs(Wall::FluxTangent), normal = mqs(Wall::FluxNormal);

	REQUIRE(tangent.R[0][0] == Catch::Approx(1.0 / DcConductance(kRing1, kCopper)).epsilon(1e-5));
	REQUIRE(tangent.R[1][1] == Catch::Approx(1.0 / DcConductance(kRing2, kCopper)).epsilon(1e-5));
	REQUIRE(std::abs(tangent.R[0][1]) < 1e-6 * tangent.R[0][0]);

	const double M = Mutual(kRing1, DcMassive, kRing2, DcMassive);
	const Matrix exact = { { Self(kRing1, DcMassive), M }, { M, Self(kRing2, DcMassive) } };
	RequireBracketed(tangent.L, normal.L, exact, 1.0, 1e-4);
}
