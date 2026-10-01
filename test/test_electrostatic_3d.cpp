// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// 3D electrostatics against an analytic multi-conductor reference: the
// capacitance matrix of two spheres (Lekner), on a curved tetrahedral Gmsh
// mesh, and the multigrid-preconditioned solver at high order.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

#include "config/input_parser.hpp"
#include "core/constants.hpp"
#include "io/mesh_loader.hpp"
#include "linalg/amg_preconditioner.hpp"
#include "solvers/electrostatic_solver.hpp"

namespace fs = std::filesystem;

ProblemConfig DecodeConfig(const json& config, const std::string& archive = {});

namespace {

// Maxwell capacitance coefficients of two spheres of radii a and b with
// centres c apart in free space, from the series of J. Lekner, "Capacitance
// coefficients of two spheres", J. Electrostatics 69 (2011) 11-14, with
// cosh u = (c^2 - a^2 - b^2) / (2ab). The terms fall off as exp(-n u), so 200
// is far past convergence for any separated pair.
struct TwoSphereCapacitance { double aa, bb, ab; };

TwoSphereCapacitance Lekner(double a, double b, double c) {
	const double u = std::acosh((c * c - a * a - b * b) / (2.0 * a * b));
	const double k = 2.0 * Constants::TWO_PI * Constants::EPSILON_0 * a * b * std::sinh(u);
	TwoSphereCapacitance C{ 0.0, 0.0, 0.0 };
	for (int n = 0; n < 200; ++n) {
		C.aa += k / (a * std::sinh(n * u) + b * std::sinh((n + 1) * u));
		C.bb += k / (b * std::sinh(n * u) + a * std::sinh((n + 1) * u));
		if (n >= 1) C.ab -= k / (c * std::sinh(n * u));
	}
	return C;
}

// test/data/two_spheres.msh: a = 1 cm at (-2.5 cm, 0, 0), b = 2 cm at
// (+2.5 cm, 0, 0), truncated by a sphere of radius R = 25 cm about the origin
// (second-order curved tetrahedra; see two_spheres.geo).
constexpr double kA = 0.01, kB = 0.02, kC = 0.05, kFarField = 0.25;

std::string TwoSphereMesh() { return std::string(MFEM_ELECTROMAG_TEST_DATA) + "/two_spheres.msh"; }

json TwoSphereConfig(int order, const std::string& linear_solver, const json& far_field) {
	return json{
		{"simulation", {
			{"physics_type", "electrostatics"}, {"mesh", TwoSphereMesh()}, {"order", order},
			{"geometry_type", "3d"}, {"analysis_type", "coupling_matrix"},
			{"linear_solver", linear_solver}, {"solver_tolerance", 1e-12},
			{"solver_max_iter", 1000}, {"solver_print_level", 0}
		}},
		{"entity_groups", json::array({
			{{"name", "Domain"}, {"dim", 3}, {"attribute_ids", {1}}},
			{{"name", "FarField"}, {"dim", 2}, {"attribute_ids", {1}}},
			{{"name", "SphereA"}, {"dim", 2}, {"attribute_ids", {2}}},
			{{"name", "SphereB"}, {"dim", 2}, {"attribute_ids", {3}}}})},
		{"regions", json::array({{{"name", "Domain"}, {"entity_group", "Domain"}, {"material", "Vacuum"}}})},
		{"materials", json::array({{{"name", "Vacuum"}, {"properties", {{"epsilon_r", 1.0}}}}})},
		{"terminals", json::array({
			{{"name", "A"}, {"quantity", "voltage"}, {"entity_group", "SphereA"}},
			{{"name", "B"}, {"quantity", "voltage"}, {"entity_group", "SphereB"}}})},
		{"boundary_conditions", json::array({far_field})},
		{"scenarios", json::array()}
	};
}

// Capacitance matrix [F] of a two-sphere run, terminal order (A, B).
std::vector<std::vector<double>> SolveTwoSpheres(const json& config) {
	const std::string archive = "two_spheres.h5";
	const auto mesh = mesh_io::LoadMesh(TwoSphereMesh());
	ElectrostaticSolver solver(*mesh, DecodeConfig(config, archive));
	solver.Setup();
	solver.Run();
	solver.SaveAnalysis();
	std::vector<std::vector<double>> C;
	HighFive::File(archive, HighFive::File::ReadOnly)
		.getDataSet("/coupling/Capacitance/values").read(C);
	fs::remove(archive);
	return C;
}

// Free space beyond the far-field sphere r = R: a monopole potential q / r
// satisfies dV/dn + V / R = 0 there exactly, and the higher multipoles of
// the pair decay faster, so a Robin condition with coefficient eps0 / R
// truncates the domain far more accurately than grounding it.
const json kRobinFarField = {{"name", "FarField"}, {"type", "robin"},
	{"entity_group", "FarField"}, {"value", 0.0},
	{"robin_coefficient", Constants::EPSILON_0 / kFarField}};

} // namespace

TEST_CASE("The Lekner series reproduces the published two-sphere coefficients",
		  "[analytic][electrostatic]") {
	// Palace "spheres" example (a = 1 cm, b = 2 cm, c = 5 cm), in pF.
	const TwoSphereCapacitance C = Lekner(kA, kB, kC);
	REQUIRE(C.aa * 1e12 == Catch::Approx(1.230518).epsilon(1e-6));
	REQUIRE(C.bb * 1e12 == Catch::Approx(2.431543).epsilon(1e-6));
	REQUIRE(C.ab * 1e12 == Catch::Approx(-0.4945668).epsilon(1e-6));
}

// The Maxwell capacitance matrix of two spheres, computed from the induced
// charges on a curved tetrahedral mesh at order 3 with the Robin far field,
// matches Lekner's series. On this coarse mesh (8 elements per 2 pi on the
// spheres) the errors are -0.10%, -0.07% and -0.22%; Palace's example, on a
// finer cubic mesh grounded at R = 75 cm, reports 0.57%, 1.9% and 3.5%.
TEST_CASE("Two-sphere capacitance matrix matches Lekner's series",
		  "[solvers][analytic][electrostatic][coupling][3d]") {
	const TwoSphereCapacitance exact = Lekner(kA, kB, kC);
	const auto C = SolveTwoSpheres(TwoSphereConfig(3, "iterative", kRobinFarField));
	INFO("C_aa " << C[0][0] << " C_bb " << C[1][1] << " C_ab " << C[0][1] << " C_ba " << C[1][0]);
	REQUIRE(C[0][0] == Catch::Approx(exact.aa).epsilon(3e-3));
	REQUIRE(C[1][1] == Catch::Approx(exact.bb).epsilon(3e-3));
	REQUIRE(C[0][1] == Catch::Approx(exact.ab).epsilon(5e-3));
	REQUIRE(C[0][1] == Catch::Approx(C[1][0]).epsilon(1e-8));

	SECTION("the multigrid-preconditioned and direct solves agree at order 3") {
		const auto direct = SolveTwoSpheres(TwoSphereConfig(3, "direct", kRobinFarField));
		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < 2; ++j) {
				REQUIRE(C[i][j] == Catch::Approx(direct[i][j]).epsilon(1e-8));
			}
		}
	}

	SECTION("grounding the far field instead is a modeling error of several percent") {
		const json ground = {{"name", "FarField"}, {"type", "dirichlet"},
							 {"entity_group", "FarField"}, {"value", 0.0}};
		const auto grounded = SolveTwoSpheres(TwoSphereConfig(3, "iterative", ground));
		REQUIRE(grounded[1][1] / exact.bb - 1.0 > 0.03);
	}
}

// High-order Lagrange stiffness matrices are hard on pointwise smoothers: with
// SPAI(0) relaxation the preconditioned CG broke down (a NaN residual within
// four iterations) on order-3 tetrahedra, even in a uniform cube. The
// Chebyshev-smoothed hierarchy converges in a mesh-independent count.
TEST_CASE("Multigrid-preconditioned CG converges on order-3 tetrahedra", "[linalg][amg]") {
	for (int n : { 8, 16 }) {
		mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(n, n, n, mfem::Element::TETRAHEDRON);
		mfem::H1_FECollection fec(3, 3);
		mfem::FiniteElementSpace fes(&mesh, &fec);
		mfem::ConstantCoefficient one(1.0);
		mfem::BilinearForm a(&fes);
		a.AddDomainIntegrator(new mfem::DiffusionIntegrator(one));
		a.Assemble();
		mfem::LinearForm f(&fes);
		f.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
		f.Assemble();
		mfem::Array<int> walls(mesh.bdr_attributes.Max()), ess;
		walls = 1;
		fes.GetEssentialTrueDofs(walls, ess);
		mfem::GridFunction u(&fes);
		u = 0.0;
		mfem::OperatorHandle A;
		mfem::Vector X, B;
		a.FormLinearSystem(ess, u, f, A, X, B);

		AmgPreconditioner amg(*A.As<mfem::SparseMatrix>());
		mfem::CGSolver cg;
		cg.SetOperator(*A);
		cg.SetPreconditioner(amg);
		cg.SetRelTol(1e-10);
		cg.SetMaxIter(100);
		cg.SetPrintLevel(0);
		cg.Mult(B, X);
		INFO(n << "^3 cells: " << cg.GetNumIterations() << " iterations");
		REQUIRE(cg.GetConverged());
		REQUIRE(cg.GetNumIterations() < 30);
	}
}
