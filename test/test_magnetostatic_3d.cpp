// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// 3D magnetostatics in the vector potential (milestone M0): Nedelec assembly,
// boundary conditions, the regularized direct solve, B recovery, routing and
// output. Coil sources arrive in M1, so sources and tangential boundary data
// are supplied programmatically here.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

#include <highfive/H5File.hpp>

#include "config/input_parser.hpp"
#include "io/mesh_loader.hpp"
#include "solvers/magnetostatic_solver_3d.hpp"
#include "solvers/solver_factory.hpp"

namespace fs = std::filesystem;

ProblemConfig DecodeConfig(const json& config, const std::string& archive = {});

namespace {

// Unit cube; MFEM's Cartesian generator labels its six faces 1-6.
json MakeCubeConfig(double mu_r) {
	return json{
		{"simulation", {
			{"physics_type", "magnetostatics"},
			{"mesh", "unused.mesh"},
			{"order", 3},
			{"geometry_type", "3d"},
			{"analysis_type", "field"},
			{"linear_solver", "direct"}
		}},
		{"entity_groups", json::array({
			{{"name", "Domain"}, {"dim", 3}, {"attribute_ids", {1}}},
			{{"name", "Walls"},  {"dim", 2}, {"attribute_ids", {1, 2, 3, 4, 5, 6}}}
		})},
		{"regions", json::array({
			{{"name", "Domain"}, {"entity_group", "Domain"}, {"material", "Core"}}
		})},
		{"materials", json::array({
			{{"name", "Core"}, {"properties", {{"mu_r", mu_r}}}}
		})},
		{"boundary_conditions", json::array({
			{{"name", "Walls"}, {"type", "dirichlet"}, {"entity_group", "Walls"}, {"value", 0.0}}
		})},
		{"scenarios", json::array({
			{{"name", "Manufactured"}, {"excitations", json::array()}}
		})}
	};
}

// Largest pointwise deviation of the solved A and of B = curl A from the exact
// fields, sampled at several reference points in every element.
struct FieldError { double a = 0.0; double b = 0.0; };

FieldError MaxFieldError(const mfem::GridFunction& A, mfem::Mesh& mesh,
						 mfem::VectorCoefficient& A_exact, mfem::VectorCoefficient& B_exact) {
	mfem::CurlGridFunctionCoefficient curl(&A);
	const double points[][3] = { { 0.25, 0.25, 0.25 }, { 0.1, 0.2, 0.3 }, { 0.3, 0.1, 0.5 } };
	FieldError err;
	mfem::Vector a, a_ex, b, b_ex;
	for (int e = 0; e < mesh.GetNE(); ++e) {
		mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
		for (const auto& p : points) {
			mfem::IntegrationPoint ip;
			ip.Set3(p[0], p[1], p[2]);
			T->SetIntPoint(&ip);
			A.GetVectorValue(*T, ip, a);
			A_exact.Eval(a_ex, *T, ip);
			curl.Eval(b, *T, ip);
			B_exact.Eval(b_ex, *T, ip);
			a -= a_ex;
			b -= b_ex;
			err.a = std::max(err.a, a.Normlinf());
			err.b = std::max(err.b, b.Normlinf());
		}
	}
	return err;
}

} // namespace

// Divergence-free quadratic A = (y^2, z^2, x^2): B = curl A = (-2z, -2x, -2y)
// and curl(nu curl A) = nu (-2, -2, -2), a constant, divergence-free source.
// A third-order Nedelec space contains every quadratic vector field, so with
// the exact tangential trace imposed the discrete solution is exact up to the
// relative 1e-6 regularization -- in A (the regularization selects the
// Coulomb gauge, which A already satisfies), in B, and in the energy
// W = 1/2 nu integral |B|^2 = 2 nu over the unit cube.
TEST_CASE("3D magnetostatics reproduces a quadratic manufactured solution",
		  "[solvers][magnetostatic][3d][manufactured]") {
	constexpr double mu_r = 2.0;
	const double nu = 1.0 / (Constants::MU_0 * mu_r);

	mfem::VectorFunctionCoefficient A_exact(3, [](const mfem::Vector& x, mfem::Vector& A) {
		A.SetSize(3);
		A(0) = x(1) * x(1);
		A(1) = x(2) * x(2);
		A(2) = x(0) * x(0);
	});
	mfem::VectorFunctionCoefficient B_exact(3, [](const mfem::Vector& x, mfem::Vector& B) {
		B.SetSize(3);
		B(0) = -2.0 * x(2);
		B(1) = -2.0 * x(0);
		B(2) = -2.0 * x(1);
	});
	mfem::Vector j(3);
	j = -2.0 * nu;
	mfem::VectorConstantCoefficient J(j);

	for (auto etype : { mfem::Element::TETRAHEDRON, mfem::Element::HEXAHEDRON }) {
		DYNAMIC_SECTION((etype == mfem::Element::TETRAHEDRON ? "tetrahedra" : "hexahedra")) {
			mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(2, 2, 2, etype);
			MagnetostaticSolver3D solver(mesh, DecodeConfig(MakeCubeConfig(mu_r)));
			solver.SetSourceCurrentDensity(&J);
			solver.SetTangentialBoundaryValue(&A_exact);
			solver.Setup();
			solver.Run();

			const FieldError err = MaxFieldError(solver.GetSolution(), mesh, A_exact, B_exact);
			// |A| and |B| are O(1) on the unit cube.
			REQUIRE(err.a < 1e-5);
			REQUIRE(err.b < 1e-5);
			REQUIRE(solver.MagneticEnergy() == Catch::Approx(2.0 * nu).epsilon(1e-5));
			// |B| = 2|x| peaks at the far corner, which interior quadrature
			// points approach but never reach.
			const double peak = solver.ComputePeakFieldMagnitude();
			REQUIRE(peak <= 2.0 * std::sqrt(3.0) * (1.0 + 1e-5));
			REQUIRE(peak > 3.0);
		}
	}
}

// Lowest-order (Whitney) edge elements represent A = 1/2 B0 x r exactly, the
// potential of a uniform field; with no source and its tangential trace on the
// boundary the solve must return B = B0 everywhere.
TEST_CASE("3D magnetostatics reproduces a uniform field at lowest order",
		  "[solvers][magnetostatic][3d][manufactured]") {
	mfem::Vector B0(3);
	B0(0) = 0.3; B0(1) = -0.2; B0(2) = 1.0;
	mfem::VectorFunctionCoefficient A_exact(3, [&B0](const mfem::Vector& x, mfem::Vector& A) {
		A.SetSize(3);
		A(0) = 0.5 * (B0(1) * x(2) - B0(2) * x(1));
		A(1) = 0.5 * (B0(2) * x(0) - B0(0) * x(2));
		A(2) = 0.5 * (B0(0) * x(1) - B0(1) * x(0));
	});
	mfem::VectorConstantCoefficient B_exact(B0);

	mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(3, 3, 3, mfem::Element::TETRAHEDRON);
	json config = MakeCubeConfig(1.0);
	config["simulation"]["order"] = 1;
	MagnetostaticSolver3D solver(mesh, DecodeConfig(config));
	solver.SetTangentialBoundaryValue(&A_exact);
	solver.Setup();
	solver.Run();

	const FieldError err = MaxFieldError(solver.GetSolution(), mesh, A_exact, B_exact);
	REQUIRE(err.b < 1e-5);
}

TEST_CASE("SolverFactory routes magnetostatics by geometry", "[solvers][factory][3d]") {
	mfem::Mesh cube = mfem::Mesh::MakeCartesian3D(1, 1, 1, mfem::Element::TETRAHEDRON);
	auto three_d = SolverFactory::Instance().Create(cube, DecodeConfig(MakeCubeConfig(1.0)));
	REQUIRE(dynamic_cast<MagnetostaticSolver3D*>(three_d.get()) != nullptr);

	json planar = MakeCubeConfig(1.0);
	planar["simulation"]["geometry_type"] = "planar";
	mfem::Mesh square = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::TRIANGLE);
	auto two_d = SolverFactory::Instance().Create(square, DecodeConfig(planar));
	REQUIRE(dynamic_cast<MagnetostaticSolver*>(two_d.get()) != nullptr);
	REQUIRE(dynamic_cast<MagnetostaticSolver3D*>(two_d.get()) == nullptr);
}

TEST_CASE("3D magnetostatics rejects what it does not yet support",
		  "[solvers][magnetostatic][3d]") {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(1, 1, 1, mfem::Element::TETRAHEDRON);
	auto setup_error = [&mesh](const json& config) {
		MagnetostaticSolver3D solver(mesh, DecodeConfig(config));
		try { solver.Setup(); } catch (const std::exception& e) { return std::string(e.what()); }
		return std::string();
	};
	using Catch::Matchers::ContainsSubstring;

	json nonzero = MakeCubeConfig(1.0);
	nonzero["boundary_conditions"][0]["value"] = 1.0;
	REQUIRE_THAT(setup_error(nonzero), ContainsSubstring("only homogeneous conditions"));

	json terminal = MakeCubeConfig(1.0);
	terminal["terminals"] = json::array({
		{{"name", "Coil"}, {"quantity", "current"}, {"entity_group", "Domain"}}});
	REQUIRE_THAT(setup_error(terminal), ContainsSubstring("terminals"));

	json coupling = MakeCubeConfig(1.0);
	coupling["simulation"]["analysis_type"] = "coupling_matrix";
	REQUIRE_THAT(setup_error(coupling), ContainsSubstring("coupling-matrix"));

	json iterative = MakeCubeConfig(1.0);
	iterative["simulation"]["linear_solver"] = "iterative";
	REQUIRE_THAT(setup_error(iterative), ContainsSubstring("AMS"));

	json amr = MakeCubeConfig(1.0);
	amr["simulation"]["amr"] = {{"enabled", true}};
	REQUIRE_THAT(setup_error(amr), ContainsSubstring("adaptive refinement"));
}

// A (a Nedelec vector field) and B go out through every writer: ParaView, Gmsh
// (per-element vector views, read back by MFEM's Gmsh reader), and HDF5.
TEST_CASE("3D magnetostatic fields are written in every output format",
		  "[solvers][magnetostatic][3d][output]") {
	const fs::path root = fs::temp_directory_path() / "mfem_ms3d_output";
	fs::remove_all(root);
	json config = MakeCubeConfig(1.0);
	config["simulation"]["order"] = 2;
	config["output"] = {{"directory", root.string()},
		{"paraview", {{"directory", "vtk"}}}, {"gmsh", {{"directory", "msh"}}},
		{"hdf5", {{"file", "run.h5"}}}};

	mfem::Vector j(3);
	j(0) = 0.0; j(1) = 0.0; j(2) = 1.0e6;
	mfem::VectorConstantCoefficient J(j);

	mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(2, 2, 2, mfem::Element::TETRAHEDRON);
	{
		MagnetostaticSolver3D solver(mesh, DecodeConfig(config));
		solver.SetSourceCurrentDensity(&J);
		solver.Setup();
		solver.Run();
		solver.SaveAnalysis();
		REQUIRE(solver.ComputePeakFieldMagnitude() > 0.0);
	}

	const std::string artifact = "scenario_000000_Manufactured";
	REQUIRE(fs::exists(root / "vtk" / artifact / "Cycle000000/proc000000.vtu"));

	const fs::path msh = root / "msh" / (artifact + ".msh");
	REQUIRE(fs::exists(msh));
	std::ifstream in(msh);
	std::stringstream text;
	text << in.rdbuf();
	// Both A and B are three-component per-element views.
	REQUIRE(text.str().find("$ElementNodeData\n2\n\"A\"") != std::string::npos);
	REQUIRE(text.str().find("$ElementNodeData\n2\n\"B\"") != std::string::npos);
	std::istringstream reread(text.str());
	mfem::Mesh reloaded(reread, 1, 0);
	REQUIRE(reloaded.GetNE() == mesh.GetNE());

	HighFive::File archive((root / "run.h5").string(), HighFive::File::ReadOnly);
	const std::string a_path = "scenarios/scenario_000000/fields/A/values";
	REQUIRE(archive.exist(a_path));
	std::string collection;
	archive.getDataSet(a_path).getAttribute("finite_element_collection").read(collection);
	REQUIRE(collection.rfind("ND", 0) == 0);

	fs::remove_all(root);
}
