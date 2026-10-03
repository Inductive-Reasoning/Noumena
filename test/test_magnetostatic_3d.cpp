// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// 3D magnetostatics in the vector potential: Nedelec assembly, boundary
// conditions, the regularized direct solve, B recovery, routing and output
// (manufactured solutions, with programmatic sources and tangential boundary
// data), and azimuthal coil terminals with the inductance matrix (checked
// against the axisymmetric solver on an equivalent geometry).

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

#include "annulus_fixture.hpp"
#include "stderr_capture.hpp"
#include "coefficients/conductor_path.hpp"
#include "config/input_parser.hpp"
#include "io/mesh_loader.hpp"
#include "solvers/magnetostatic_solver_3d.hpp"
#include "solvers/solver_factory.hpp"

namespace fs = std::filesystem;

ProblemConfig DecodeConfig(const json& config, const std::string& archive = {});

using namespace annulus;

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


// Inductance matrix of a coupling run, read back from its HDF5 archive.
std::vector<std::vector<double>> SolveInductance(PhysicsSolver& solver, const std::string& archive) {
	solver.Setup();
	solver.Run();
	solver.SaveAnalysis();
	std::vector<std::vector<double>> values;
	HighFive::File(archive, HighFive::File::ReadOnly)
		.getDataSet("/coupling/Inductance/values").read(values);
	fs::remove(archive);
	return values;
}

std::vector<std::vector<double>> AxisymmetricInductance(const AnnulusSpec& spec) {
	mfem::Mesh mesh = MakeAnnulus2D(spec, 4);
	MagnetostaticSolver solver(mesh, DecodeConfig(MakeAnnulusConfig(spec, false, 3), "axi_ref.h5"));
	return SolveInductance(solver, "axi_ref.h5");
}

std::vector<std::vector<double>> Inductance3D(const AnnulusSpec& spec, int n_theta, int order) {
	mfem::Mesh mesh = MakeAnnulus3D(spec, n_theta);
	MagnetostaticSolver3D solver(mesh, DecodeConfig(MakeAnnulusConfig(spec, true, order), "ms3d.h5"));
	return SolveInductance(solver, "ms3d.h5");
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

#ifndef MFEM_USE_MPI
	json iterative = MakeCubeConfig(1.0);
	iterative["simulation"]["linear_solver"] = "iterative";
	REQUIRE_THAT(setup_error(iterative), ContainsSubstring("MPI/HYPRE build"));
#endif

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

// Two coaxial coils in the annular cylinder: the 3D inductance matrix must
// match the axisymmetric solver on the same (r, z) geometry -- an exact
// equivalence of the continuum problems (see AnnulusSpec), so the difference
// is discretization only. The reference uses a 4x refined lattice at order 3
// and is converged well past the tolerance; the 3D run is second order on the
// base lattice with 16 cells around, where the single-coil difference was
// measured at 5e-4. L must also be symmetric: it is B'^T K^-1 B' by
// construction.
TEST_CASE("3D coil inductances match the axisymmetric solver",
		  "[solvers][magnetostatic][3d][coupling][analytic]") {
	AnnulusSpec spec;
	spec.conductors = { { 0.04, 0.06, 0.02, 0.04 }, { 0.05, 0.08, 0.06, 0.08 } };
	const auto reference = AxisymmetricInductance(spec);
	const auto l3d = Inductance3D(spec, 16, 2);

	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			INFO("L(" << i << "," << j << ") 3D " << l3d[i][j] << " axisymmetric " << reference[i][j]);
			REQUIRE(l3d[i][j] == Catch::Approx(reference[i][j]).epsilon(2e-3));
		}
	}
	REQUIRE(l3d[0][1] == Catch::Approx(l3d[1][0]).epsilon(1e-10));
	REQUIRE(l3d[0][1] > 0.0);  // coaxial coils in the same sense couple positively
}

// A field scenario and the coupling matrix must tell the same story: the flux
// linkages of a two-current scenario are L I, and the stored energy is
// 1/2 I^T L I. Coarse lattice (fast); only internal consistency is checked.
TEST_CASE("3D field scenario is consistent with the inductance matrix",
		  "[solvers][magnetostatic][3d][coupling]") {
	AnnulusSpec spec;
	spec.nr = 4;
	spec.nz = 5;
	spec.conductors = { { 0.04, 0.06, 0.02, 0.04 }, { 0.06, 0.08, 0.06, 0.08 } };
	const auto L = Inductance3D(spec, 12, 2);

	const double I[2] = { 3.0, -1.5 };
	json config = MakeAnnulusConfig(spec, true, 2, "magnetostatics", "field");
	config["scenarios"] = json::array({{{"name", "Drive"}, {"excitations", json::array({
		{{"terminal", "C1"}, {"value", I[0]}}, {{"terminal", "C2"}, {"value", I[1]}}})}}});
	mfem::Mesh mesh = MakeAnnulus3D(spec, 12);
	MagnetostaticSolver3D solver(mesh, DecodeConfig(config));
	solver.Setup();
	solver.Run();

	const auto lambda = solver.FluxLinkages();
	double energy = 0.0;
	for (int i = 0; i < 2; ++i) {
		const double expected = L[i][0] * I[0] + L[i][1] * I[1];
		REQUIRE(lambda[i] == Catch::Approx(expected).epsilon(1e-8));
		energy += 0.5 * I[i] * expected;
	}
	// The field energy omits the (relative 1e-6) regularization term that the
	// operator includes, hence the looser tolerance.
	REQUIRE(solver.MagneticEnergy() == Catch::Approx(energy).epsilon(1e-5));
}

// The projector must remove exactly the gradient part of a load. A uniform
// J = z-hat confined to the coil region is divergence-free inside but its
// current starts and stops at the coil's end faces, so its load is far from
// balanced; after projection G^T b must vanish to solver precision. The
// solver's own azimuthal coil load (already nearly balanced on this curved
// mesh) must also come out balanced.
TEST_CASE("Coil loads are made discretely divergence-free",
		  "[solvers][magnetostatic][3d]") {
	AnnulusSpec spec;
	spec.nr = 4;
	spec.nz = 5;
	spec.conductors = { { 0.04, 0.06, 0.04, 0.06 } };
	mfem::Mesh mesh = MakeAnnulus3D(spec, 12);
	MagnetostaticSolver3D solver(mesh, DecodeConfig(MakeAnnulusConfig(spec, true, 2), "proj.h5"));
	solver.Setup();
	const DivergenceFreeProjector& projector = solver.Projector();

	mfem::FiniteElementSpace nd(&mesh, solver.GetSolution().FESpace()->FEColl());
	mfem::Array<int> coil(mesh.attributes.Max());
	coil = 0;
	coil[1] = 1;
	mfem::Vector z_hat(3);
	z_hat = 0.0;
	z_hat(2) = 1.0;
	mfem::VectorConstantCoefficient axial(z_hat);
	mfem::LinearForm raw(&nd);
	raw.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(axial), coil);
	raw.Assemble();

	mfem::Vector projected(raw);
	projector.Project(projected);
	const double before = projector.GradientResidual(raw);
	const double after = projector.GradientResidual(projected);
	INFO("gradient residual before " << before << " after " << after);
	REQUIRE(before > 1e-3 * raw.Norml2());
	REQUIRE(after < 1e-9 * before);

	const mfem::Vector& coil_load = solver.TerminalLoads()[0];
	REQUIRE(projector.GradientResidual(coil_load) < 1e-9 * coil_load.Norml2());
}

// A closed coil described by a cut must carry the same current as the same
// coil described analytically: the conduction potential of a coil of
// revolution is theta / (2 pi), so its direction is phi-hat and its
// cross-section the meridional area. The two agree up to the discretization
// of that potential, and both match the axisymmetric solver.
TEST_CASE("A cut coil reproduces the azimuthal coil", "[solvers][magnetostatic][3d][conductor]") {
	AnnulusSpec spec;
	spec.conductors = { { 0.04, 0.06, 0.04, 0.06 } };
	const double azimuthal = Inductance3D(spec, 16, 2)[0][0];
	spec.cut = true;
	const double cut = Inductance3D(spec, 16, 2)[0][0];
	const double reference = AxisymmetricInductance(spec)[0][0];
	INFO("cut " << cut << ", azimuthal " << azimuthal << ", axisymmetric " << reference);
	REQUIRE(cut == Catch::Approx(azimuthal).epsilon(1e-3));
	REQUIRE(cut == Catch::Approx(reference).epsilon(2e-3));
}

// The cut normal only has to cross the cut: a normal tilted 45 degrees off
// the cut plane (y = 0) describes the same crossing and gives the same
// inductance, while one lying in the plane cannot say which side is
// downstream and is rejected.
TEST_CASE("The cut normal must cross the cut", "[solvers][magnetostatic][3d][conductor]") {
	using Catch::Matchers::ContainsSubstring;
	AnnulusSpec spec;
	spec.conductors = { { 0.04, 0.06, 0.04, 0.06 } };
	spec.cut = true;
	auto solve = [&](const std::vector<double>& normal) {
		mfem::Mesh mesh = MakeAnnulus3D(spec, 16);
		json config = MakeAnnulusConfig(spec, true, 1);
		config["terminals"][0]["direction"]["normal"] = normal;
		MagnetostaticSolver3D solver(mesh, DecodeConfig(config, "cut_normal.h5"));
		return SolveInductance(solver, "cut_normal.h5")[0][0];
	};
	const double straight = solve({ 0.0, 1.0, 0.0 });
	REQUIRE(solve({ 0.0, 1.0, 1.0 }) == Catch::Approx(straight).epsilon(1e-12));
	REQUIRE_THROWS_WITH(solve({ 1.0, 0.0, 0.0 }), ContainsSubstring("must cross the cut"));
	REQUIRE_THROWS_WITH(solve({ 1.0, 0.5, 0.0 }), ContainsSubstring("must cross the cut"));
}

// Which side of a cut an element lies on is decided by connectivity, so a cut
// need not be planar. A staircase cut -- the coil's outer radial half one cell
// further around, joined to the inner half by constant-r faces, half of them
// parallel to the normal -- describes the same circulation and gives the same
// inductance as the planar cut.
TEST_CASE("A staircase cut reproduces the planar cut", "[solvers][magnetostatic][3d][conductor]") {
	AnnulusSpec spec;
	spec.conductors = { { 0.04, 0.06, 0.04, 0.06 } };
	spec.cut = true;
	const double planar = Inductance3D(spec, 16, 2)[0][0];
	spec.staircase_cut = true;
	const double staircase = Inductance3D(spec, 16, 2)[0][0];
	INFO("planar " << planar << ", staircase " << staircase);
	REQUIRE(staircase == Catch::Approx(planar).epsilon(1e-6));
}

// A cut must sever its conductor. With the cut faces of the coil's outer
// radial half moved to an unused attribute, the cut covers half the cross-
// section and its rim runs through the conductor, so the jump would end
// inside it.
TEST_CASE("A cut must sever its conductor", "[solvers][magnetostatic][3d][conductor]") {
	using Catch::Matchers::ContainsSubstring;
	AnnulusSpec spec;
	spec.conductors = { { 0.04, 0.06, 0.04, 0.06 } };
	spec.cut = true;
	mfem::Mesh mesh = MakeAnnulus3D(spec, 16);
	int moved = 0;
	for (int be = 0; be < mesh.GetNBE(); ++be) {
		if (mesh.GetBdrAttribute(be) != 2) continue;
		mfem::Vector c;
		mesh.GetBdrElementTransformation(be)->Transform(
			mfem::Geometries.GetCenter(mesh.GetBdrElementGeometry(be)), c);
		if (std::hypot(c(0), c(1)) > 0.05) {
			mesh.SetBdrAttribute(be, 99);
			++moved;
		}
	}
	mesh.SetAttributes();
	REQUIRE(moved > 0);
	MagnetostaticSolver3D solver(mesh, DecodeConfig(MakeAnnulusConfig(spec, true, 1)));
	REQUIRE_THROWS_WITH(solver.Setup(), ContainsSubstring("does not span its conductor"));
}

namespace {
// Unit cube; a square copper bar (attribute 2, 0.25 < x, y < 0.75) runs from
// the bottom wall (z = 0, attribute 1) to the top wall (z = 1, attribute 6).
json MakeBarConfig(const std::string& conductor_type = "stranded") {
	json config = MakeCubeConfig(1.0);
	config["simulation"]["order"] = 2;
	config["entity_groups"].push_back({{"name", "Bar"}, {"dim", 3}, {"attribute_ids", {2}}});
	config["entity_groups"].push_back({{"name", "Bottom"}, {"dim", 2}, {"attribute_ids", {1}}});
	config["entity_groups"].push_back({{"name", "Top"}, {"dim", 2}, {"attribute_ids", {6}}});
	config["regions"].push_back({{"name", "Bar"}, {"entity_group", "Bar"}, {"material", "Copper"}});
	config["materials"].push_back(
		{{"name", "Copper"}, {"properties", {{"mu_r", 1.0}, {"sigma", 5.8e7}}}});
	config["terminals"] = json::array({
		{{"name", "Bar"}, {"quantity", "current"}, {"entity_group", "Bar"},
		 {"conductor_type", conductor_type},
		 {"direction", {{"type", "electrodes"}, {"input", "Bottom"}, {"output", "Top"}}}}});
	config["scenarios"] = json::array({{{"name", "Drive"}, {"excitations", json::array({
		{{"terminal", "Bar"}, {"value", 1.0}}})}}});
	return config;
}

mfem::Mesh MakeBarMesh() {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::HEXAHEDRON);
	for (int e = 0; e < mesh.GetNE(); ++e) {
		mfem::Vector c;
		mesh.GetElementCenter(e, c);
		const bool bar = c(0) > 0.25 && c(0) < 0.75 && c(1) > 0.25 && c(1) < 0.75;
		mesh.SetAttribute(e, bar ? 2 : 1);
	}
	mesh.SetAttributes();
	return mesh;
}
} // namespace

// Conductor loads are projected within their conductor, so whatever
// imbalance the projection removes is made up inside it, not by current in
// the air. An azimuthal direction about an axis 1 mm off the coil's (below
// the warning threshold) leaves an imbalance of about 1%: the projected load
// must stay on the coil's own DOFs and still be orthogonal to every gradient
// of the mesh, while the global projection of the same load spreads into the
// air.
TEST_CASE("Conductor loads are projected within their conductor",
		  "[solvers][magnetostatic][3d][conductor]") {
	AnnulusSpec spec;
	spec.conductors = { { 0.04, 0.06, 0.04, 0.06 } };
	mfem::Mesh mesh = MakeAnnulus3D(spec, 16);
	json config = MakeAnnulusConfig(spec, true, 1, "magnetostatics", "field");
	config["terminals"][0]["direction"]["origin"] = { 0.001, 0.0, 0.0 };
	MagnetostaticSolver3D solver(mesh, DecodeConfig(config));
	solver.Setup();
	const mfem::Vector& load = solver.TerminalLoads()[0];

	// Nedelec DOFs of the coil's elements (same space, same numbering).
	mfem::FiniteElementSpace nd(&mesh, solver.GetSolution().FESpace()->FEColl());
	std::vector<bool> in_coil(nd.GetVSize(), false);
	mfem::Array<int> dofs;
	for (int e = 0; e < mesh.GetNE(); ++e) {
		if (mesh.GetAttribute(e) != 2) continue;
		nd.GetElementDofs(e, dofs);
		for (int d : dofs) { in_coil[d >= 0 ? d : -1 - d] = true; }
	}
	auto outside = [&](const mfem::Vector& b) {
		double norm = 0.0;
		for (int i = 0; i < b.Size(); ++i) { if (!in_coil[i]) norm += b(i) * b(i); }
		return std::sqrt(norm);
	};

	REQUIRE(outside(load) == 0.0);
	REQUIRE(solver.Projector().GradientResidual(load) < 1e-9 * load.Norml2());

	// The same raw load projected globally leaks into the air.
	CurrentDirection d;
	d.Origin = { 0.001, 0.0, 0.0 };
	AzimuthalPath path(d);
	mfem::Array<int> coil(mesh.attributes.Max());
	coil = 0;
	coil[1] = 1;
	ConductorCurrentCoefficient J(path, nullptr, 1.0 / ConductorPathIntegral(mesh, coil, path, nullptr, 1));
	mfem::LinearForm raw(&nd);
	raw.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(J), coil);
	raw.Assemble();
	mfem::Vector global(raw);
	solver.Projector().Project(global);
	INFO("global projection outside the coil " << outside(global) << " of " << global.Norml2());
	REQUIRE(outside(global) > 1e-3 * global.Norml2());
}

// A current that does not balance in its conductor is reported, as the part
// of it the divergence-free projection has to remove. Correct directions
// lose only round-off: an azimuthal one on the annulus, faceted or curved,
// and the massive (conduction) current of a stepped bar, re-entrant corner
// and all. Reported: an azimuthal direction about an axis 5 mm off the
// coil's, or on the off-axis bar; and a stranded (uniform) current in the
// stepped bar, whose cross-section halves along the path.
TEST_CASE("A current that does not balance in its conductor is reported",
		  "[solvers][magnetostatic][3d][conductor]") {
	auto warnings = [](mfem::Mesh& mesh, const json& config) {
		StderrCapture capture;
		MagnetostaticSolver3D solver(mesh, DecodeConfig(config));
		solver.Setup();
		return capture.Text();
	};
	const std::string unbalanced = "does not stay balanced";

	AnnulusSpec spec;
	spec.conductors = { { 0.04, 0.06, 0.04, 0.06 } };
	for (bool curved : { false, true }) {
		mfem::Mesh mesh = MakeAnnulus3D(spec, 16, curved);
		REQUIRE(warnings(mesh, MakeAnnulusConfig(spec, true, 1, "magnetostatics", "field"))
			.find(unbalanced) == std::string::npos);
	}
	{
		mfem::Mesh mesh = MakeAnnulus3D(spec, 16);
		json config = MakeAnnulusConfig(spec, true, 1, "magnetostatics", "field");
		config["terminals"][0]["direction"]["origin"] = { 0.005, 0.0, 0.0 };
		REQUIRE_THAT(warnings(mesh, config),
			Catch::Matchers::ContainsSubstring("'azimuthal' direction fits only"));
	}
	{
		mfem::Mesh mesh = MakeBarMesh();
		json config = MakeBarConfig();
		config["terminals"][0]["direction"] = {{"type", "azimuthal"},
			{"origin", {0.0, 0.0, 0.0}}, {"axis", {0.0, 0.0, 1.0}}};
		REQUIRE_THAT(warnings(mesh, config),
			Catch::Matchers::ContainsSubstring("'azimuthal' direction fits only"));
	}
	{
		mfem::Mesh mesh = MakeBarMesh();
		for (int e = 0; e < mesh.GetNE(); ++e) {
			mfem::Vector c;
			mesh.GetElementCenter(e, c);
			if (mesh.GetAttribute(e) == 2 && c(2) > 0.5 && c(0) > 0.5) mesh.SetAttribute(e, 1);
		}
		mesh.SetAttributes();
		REQUIRE(warnings(mesh, MakeBarConfig("massive")).find(unbalanced) == std::string::npos);
		REQUIRE_THAT(warnings(mesh, MakeBarConfig("stranded")),
			Catch::Matchers::ContainsSubstring("stranded current is uniform"));
	}
}

// An open conductor between electrodes on two n x A = 0 walls: the conduction
// potential of a straight uniform bar is linear, so stranded and massive alike
// must carry exactly a uniform 1 A / (0.5 x 0.5) axial current -- the same
// field as that source given directly.
TEST_CASE("An electrode conductor reproduces a uniform bar current",
		  "[solvers][magnetostatic][3d][conductor]") {
	const std::string type = GENERATE(std::string("stranded"), std::string("massive"));
	INFO("conductor_type " << type);
	mfem::Mesh mesh = MakeBarMesh();
	MagnetostaticSolver3D bar(mesh, DecodeConfig(MakeBarConfig(type)));
	bar.Setup();
	bar.Run();

	json direct = MakeBarConfig();
	direct["terminals"] = json::array();
	direct["scenarios"][0]["excitations"] = json::array();
	mfem::Vector j(3);
	j = 0.0;
	j(2) = 1.0 / 0.25;
	mfem::VectorConstantCoefficient uniform(j);
	mfem::Array<int> marker(2);
	marker[0] = 0;
	marker[1] = 1;
	mfem::VectorRestrictedCoefficient J(uniform, marker);
	MagnetostaticSolver3D reference(mesh, DecodeConfig(direct));
	reference.SetSourceCurrentDensity(&J);
	reference.Setup();
	reference.Run();

	REQUIRE(bar.MagneticEnergy() > 0.0);
	REQUIRE(bar.MagneticEnergy() == Catch::Approx(reference.MagneticEnergy()).epsilon(1e-9));
}

TEST_CASE("3D current terminals are validated", "[solvers][magnetostatic][3d]") {
	using Catch::Matchers::ContainsSubstring;

	SECTION("a conductor touching its axis is rejected") {
		// Unit cube with the axis through its centre line: interior vertices
		// of the 2x2x2 mesh lie on it.
		mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(2, 2, 2, mfem::Element::TETRAHEDRON);
		json config = MakeCubeConfig(1.0);
		config["simulation"]["order"] = 1;
		config["terminals"] = json::array({
			{{"name", "Coil"}, {"quantity", "current"}, {"entity_group", "Domain"},
			 {"conductor_type", "stranded"},
			 {"direction", {{"type", "azimuthal"}, {"origin", {0.5, 0.5, 0.0}},
							{"axis", {0.0, 0.0, 1.0}}}}}});
		MagnetostaticSolver3D solver(mesh, DecodeConfig(config));
		REQUIRE_THROWS_WITH(solver.Setup(), ContainsSubstring("reaches its own axis"));
	}

	SECTION("electrodes off an n x A = 0 wall are rejected") {
		mfem::Mesh mesh = MakeBarMesh();
		json config = MakeBarConfig();
		config["entity_groups"].push_back({{"name", "Sides"}, {"dim", 2}, {"attribute_ids", {2, 3, 4, 5}}});
		config["boundary_conditions"][0]["entity_group"] = "Sides";
		MagnetostaticSolver3D solver(mesh, DecodeConfig(config));
		REQUIRE_THROWS_WITH(solver.Setup(), ContainsSubstring("must lie on a 'dirichlet'"));
	}

	SECTION("electrodes on unconnected pieces of the n x A = 0 wall are rejected") {
		// Only the bottom and top faces are n x A = 0: no path along the wall
		// joins the electrodes for the current to return by.
		mfem::Mesh mesh = MakeBarMesh();
		json config = MakeBarConfig();
		config["entity_groups"].push_back({{"name", "Ends"}, {"dim", 2}, {"attribute_ids", {1, 6}}});
		config["boundary_conditions"][0]["entity_group"] = "Ends";
		MagnetostaticSolver3D solver(mesh, DecodeConfig(config));
		REQUIRE_THROWS_WITH(solver.Setup(), ContainsSubstring("separate pieces"));
	}

	SECTION("a terminal without a direction is rejected") {
		mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(1, 1, 1, mfem::Element::TETRAHEDRON);
		json config = MakeCubeConfig(1.0);
		config["terminals"] = json::array({
			{{"name", "Coil"}, {"quantity", "current"}, {"entity_group", "Domain"}}});
		MagnetostaticSolver3D solver(mesh, DecodeConfig(config));
		REQUIRE_THROWS_WITH(solver.Setup(), ContainsSubstring("needs a 'direction'"));
	}
}

#ifdef MFEM_USE_MPI

// ---- Iterative (AMS-preconditioned CG) path, MPI/HYPRE build ---------------

// The quadratic manufactured solution again, now through the singular system
// with no regularization. After the gauge fix A is in the Coulomb gauge, which
// A_exact satisfies, so A as well as B must be exact to solver precision --
// about four orders tighter than the regularized direct path allows.
TEST_CASE("3D magnetostatics iterative solve reproduces the manufactured solution",
		  "[solvers][magnetostatic][3d][manufactured][ams]") {
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
			json config = MakeCubeConfig(mu_r);
			config["simulation"]["linear_solver"] = "iterative";
			config["simulation"]["solver_tolerance"] = 1e-12;
			MagnetostaticSolver3D solver(mesh, DecodeConfig(config));
			solver.SetSourceCurrentDensity(&J);
			solver.SetTangentialBoundaryValue(&A_exact);
			solver.Setup();
			solver.Run();

			const FieldError err = MaxFieldError(solver.GetSolution(), mesh, A_exact, B_exact);
			INFO("A error " << err.a << ", B error " << err.b);
			REQUIRE(err.a < 1e-9);
			REQUIRE(err.b < 1e-9);
			REQUIRE(solver.MagneticEnergy() == Catch::Approx(2.0 * nu).epsilon(1e-10));
		}
	}
}

// Both linear solvers must give the same inductance matrix; they differ only
// by the direct path's relative-1e-6 regularization.
TEST_CASE("3D inductances agree between the AMS and direct solvers",
		  "[solvers][magnetostatic][3d][coupling][ams]") {
	AnnulusSpec spec;
	spec.conductors = { { 0.04, 0.06, 0.02, 0.04 }, { 0.05, 0.08, 0.06, 0.08 } };
	const auto direct = Inductance3D(spec, 16, 2);

	mfem::Mesh mesh = MakeAnnulus3D(spec, 16);
	json config = MakeAnnulusConfig(spec, true, 2);
	config["simulation"]["linear_solver"] = "iterative";
	config["simulation"]["solver_tolerance"] = 1e-12;
	MagnetostaticSolver3D solver(mesh, DecodeConfig(config, "ms3d_ams.h5"));
	const auto iterative = SolveInductance(solver, "ms3d_ams.h5");

	for (int i = 0; i < 2; ++i) {
		for (int k = 0; k < 2; ++k) {
			INFO("L(" << i << "," << k << ") AMS " << iterative[i][k] << " direct " << direct[i][k]);
			REQUIRE(iterative[i][k] == Catch::Approx(direct[i][k]).epsilon(1e-5));
		}
	}
	REQUIRE(iterative[0][1] == Catch::Approx(iterative[1][0]).epsilon(1e-9));
}

// Hidden benchmark (run with "[ams-benchmark]"): the 2x refined annulus that
// the direct factorization did not finish in 10 minutes.
TEST_CASE("AMS scales past the direct solver", "[.][ams-benchmark]") {
	AnnulusSpec spec;
	spec.nr *= 2;
	spec.nz *= 2;
	spec.conductors = { { 0.04, 0.06, 0.04, 0.06 } };
	const double reference = AxisymmetricInductance(spec)[0][0];
	mfem::Mesh mesh = MakeAnnulus3D(spec, 32);
	json config = MakeAnnulusConfig(spec, true, 2);
	config["simulation"]["linear_solver"] = "iterative";
	config["simulation"]["solver_tolerance"] = 1e-10;
	config["simulation"]["solver_print_level"] = 1;
	MagnetostaticSolver3D solver(mesh, DecodeConfig(config, "ms3d_bench.h5"));
	// Print the CG iteration counts and phase timings.
	StatusReporter::Global().SetVerbosity(StatusReporter::Verbosity::Diagnostics);
	const auto start = std::chrono::steady_clock::now();
	const double l = SolveInductance(solver, "ms3d_bench.h5")[0][0];
	const double seconds =
		std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	WARN("elements " << mesh.GetNE() << ", L " << l << ", axisymmetric " << reference
		 << ", relative difference " << (l - reference) / reference << ", " << seconds << " s");
	REQUIRE(l == Catch::Approx(reference).epsilon(1e-3));
}

#endif // MFEM_USE_MPI
