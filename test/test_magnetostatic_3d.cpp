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
#include <catch2/matchers/catch_matchers_string.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

#include "config/input_parser.hpp"
#include "io/mesh_loader.hpp"
#include "coefficients/azimuthal_coil_coefficient.hpp"
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


// ---- Annular-cylinder coil geometry shared by the 3D and axisymmetric runs --
//
// Domain r_in <= r <= r_out, 0 <= z <= height about the z axis, with n x A = 0
// on every wall. For an azimuthal source that is EXACTLY the axisymmetric
// problem on the (r, z) rectangle with A_phi = 0 on its boundary (n x A = 0
// for an azimuthal A is A_phi = 0), so the 2D solver is a reference with no
// modeling difference, only discretization. Coils are rectangles in (r, z)
// aligned with the lattice; attribute 1 is air, 2, 3, ... the coils.
struct CoilRect { double r0, r1, z0, z1; };
struct AnnulusSpec {
	double r_in = 0.02, r_out = 0.10, height = 0.10;
	int nr = 8, nz = 10;
	std::vector<CoilRect> coils;
};

int CellAttribute(const AnnulusSpec& spec, double r, double z) {
	for (size_t c = 0; c < spec.coils.size(); ++c) {
		const auto& k = spec.coils[c];
		if (r > k.r0 && r < k.r1 && z > k.z0 && z < k.z1) return static_cast<int>(c) + 2;
	}
	return 1;
}

mfem::Mesh MakeAnnulus2D(const AnnulusSpec& spec, int refine) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(spec.nr * refine, spec.nz * refine,
		mfem::Element::QUADRILATERAL, true, spec.r_out - spec.r_in, spec.height);
	for (int v = 0; v < mesh.GetNV(); ++v) { mesh.GetVertex(v)[0] += spec.r_in; }
	for (int e = 0; e < mesh.GetNE(); ++e) {
		mfem::Vector c;
		mesh.GetElementCenter(e, c);
		mesh.SetAttribute(e, CellAttribute(spec, c(0), c(1)));
	}
	mesh.SetAttributes();
	return mesh;
}

// The same lattice revolved about z with n_theta cells around, as curved
// (order-2 geometry) hexahedra whose nodes are snapped radially onto their
// lattice circle. Boundary attribute 1 on every wall.
mfem::Mesh MakeAnnulus3D(const AnnulusSpec& spec, int n_theta) {
	const int nr = spec.nr, nz = spec.nz;
	const double dr = (spec.r_out - spec.r_in) / nr, dz = spec.height / nz;
	auto id = [&](int i, int j, int k) { return (k * (nr + 1) + i) * n_theta + (j % n_theta); };

	mfem::Mesh mesh(3, (nr + 1) * (nz + 1) * n_theta, nr * nz * n_theta,
					2 * n_theta * (nr + nz), 3);
	for (int k = 0; k <= nz; ++k) {
		for (int i = 0; i <= nr; ++i) {
			for (int j = 0; j < n_theta; ++j) {
				const double r = spec.r_in + i * dr, t = Constants::TWO_PI * j / n_theta;
				mesh.AddVertex(r * std::cos(t), r * std::sin(t), k * dz);
			}
		}
	}
	for (int k = 0; k < nz; ++k) {
		for (int i = 0; i < nr; ++i) {
			const int attr = CellAttribute(spec, spec.r_in + (i + 0.5) * dr, (k + 0.5) * dz);
			for (int j = 0; j < n_theta; ++j) {
				// (r, theta, z) is right-handed, so this corner order is too.
				const int v[8] = { id(i, j, k), id(i + 1, j, k), id(i + 1, j + 1, k), id(i, j + 1, k),
								   id(i, j, k + 1), id(i + 1, j, k + 1), id(i + 1, j + 1, k + 1),
								   id(i, j + 1, k + 1) };
				mesh.AddHex(v, attr);
			}
		}
	}
	for (int j = 0; j < n_theta; ++j) {
		for (int k = 0; k < nz; ++k) {
			for (int i : { 0, nr }) {
				const int q[4] = { id(i, j, k), id(i, j + 1, k), id(i, j + 1, k + 1), id(i, j, k + 1) };
				mesh.AddBdrQuad(q, 1);
			}
		}
		for (int i = 0; i < nr; ++i) {
			for (int k : { 0, nz }) {
				const int q[4] = { id(i, j, k), id(i + 1, j, k), id(i + 1, j + 1, k), id(i, j + 1, k) };
				mesh.AddBdrQuad(q, 1);
			}
		}
	}
	mesh.FinalizeHexMesh(1, 0, true);

	// The radial snap below is unambiguous only while the chord sag of an
	// angular edge stays under a quarter of the radial spacing; beyond that a
	// node can snap to the wrong circle and fold its element.
	const double sag = spec.r_out * (1.0 - std::cos(0.5 * Constants::TWO_PI / n_theta));
	REQUIRE(sag < 0.25 * dr);

	mesh.SetCurvature(2);
	const double half = 0.5 * dr;
	mesh.Transform([&](const mfem::Vector& x, mfem::Vector& y) {
		y = x;
		const double r = std::hypot(x(0), x(1));
		const double snapped = spec.r_in + half * std::round((r - spec.r_in) / half);
		y(0) *= snapped / r;
		y(1) *= snapped / r;
	});
	return mesh;
}

// Coupling-matrix config for the annulus in either model; one terminal per
// coil, named Coil1, Coil2, ...
json MakeAnnulusConfig(const AnnulusSpec& spec, bool three_d, int order) {
	json groups = json::array();
	json regions = json::array();
	json terminals = json::array();
	std::vector<int> all;
	for (int a = 1; a <= static_cast<int>(spec.coils.size()) + 1; ++a) all.push_back(a);
	groups.push_back({{"name", "Domain"}, {"dim", three_d ? 3 : 2}, {"attribute_ids", all}});
	groups.push_back({{"name", "Walls"}, {"dim", three_d ? 2 : 1},
					  {"attribute_ids", three_d ? std::vector<int>{1} : std::vector<int>{1, 2, 3, 4}}});
	regions.push_back({{"name", "Domain"}, {"entity_group", "Domain"}, {"material", "Air"}});
	for (size_t c = 0; c < spec.coils.size(); ++c) {
		const std::string name = "Coil" + std::to_string(c + 1);
		groups.push_back({{"name", name}, {"dim", three_d ? 3 : 2},
						  {"attribute_ids", {static_cast<int>(c) + 2}}});
		json terminal = {{"name", name}, {"quantity", "current"}, {"entity_group", name}};
		if (three_d) {
			terminal["direction"] = {{"type", "azimuthal"}, {"origin", {0.0, 0.0, 0.0}},
									 {"axis", {0.0, 0.0, 1.0}}};
		}
		terminals.push_back(terminal);
	}
	return json{
		{"simulation", {
			{"physics_type", "magnetostatics"}, {"mesh", "unused.mesh"}, {"order", order},
			{"geometry_type", three_d ? "3d" : "axisymmetric"},
			{"analysis_type", "coupling_matrix"}, {"linear_solver", "direct"}
		}},
		{"entity_groups", groups},
		{"regions", regions},
		{"materials", json::array({{{"name", "Air"}, {"properties", {{"mu_r", 1.0}}}}})},
		{"terminals", terminals},
		{"boundary_conditions", json::array({
			{{"name", "Walls"}, {"type", "dirichlet"}, {"entity_group", "Walls"}, {"value", 0.0}}})},
		{"scenarios", json::array()}
	};
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
	spec.coils = { { 0.04, 0.06, 0.02, 0.04 }, { 0.05, 0.08, 0.06, 0.08 } };
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
	spec.coils = { { 0.04, 0.06, 0.02, 0.04 }, { 0.06, 0.08, 0.06, 0.08 } };
	const auto L = Inductance3D(spec, 12, 2);

	const double I[2] = { 3.0, -1.5 };
	json config = MakeAnnulusConfig(spec, true, 2);
	config["simulation"]["analysis_type"] = "field";
	config["scenarios"] = json::array({{{"name", "Drive"}, {"excitations", json::array({
		{{"terminal", "Coil1"}, {"value", I[0]}}, {{"terminal", "Coil2"}, {"value", I[1]}}})}}});
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
	spec.coils = { { 0.04, 0.06, 0.04, 0.06 } };
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

	const mfem::Vector& coil_load = solver.CoilLoads()[0];
	REQUIRE(projector.GradientResidual(coil_load) < 1e-9 * coil_load.Norml2());
}

TEST_CASE("3D coil terminals are validated", "[solvers][magnetostatic][3d]") {
	using Catch::Matchers::ContainsSubstring;

	SECTION("a coil touching its axis is rejected") {
		// Unit cube with the axis through its centre line: interior vertices
		// of the 2x2x2 mesh lie on it.
		mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(2, 2, 2, mfem::Element::TETRAHEDRON);
		json config = MakeCubeConfig(1.0);
		config["simulation"]["order"] = 1;
		config["terminals"] = json::array({
			{{"name", "Coil"}, {"quantity", "current"}, {"entity_group", "Domain"},
			 {"direction", {{"type", "azimuthal"}, {"origin", {0.5, 0.5, 0.0}},
							{"axis", {0.0, 0.0, 1.0}}}}}});
		MagnetostaticSolver3D solver(mesh, DecodeConfig(config));
		REQUIRE_THROWS_WITH(solver.Setup(), ContainsSubstring("reaches its own axis"));
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
	spec.coils = { { 0.04, 0.06, 0.02, 0.04 }, { 0.05, 0.08, 0.06, 0.08 } };
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
	spec.coils = { { 0.04, 0.06, 0.04, 0.06 } };
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
