// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// 3D time-harmonic magnetoquasistatics: massive ports, stranded sources and
// passive eddy-current conductors, checked against the axisymmetric solver on
// an equivalent geometry, against closed forms in the DC limit, against 3D
// magnetostatics at low frequency, and by the loss/impedance energy balance.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

#include "annulus_fixture.hpp"
#include "config/input_parser.hpp"
#include "solvers/magnetoquasistatic_solver.hpp"
#include "solvers/magnetoquasistatic_solver_3d.hpp"
#include "solvers/magnetostatic_solver_3d.hpp"
#include "solvers/solver_factory.hpp"

namespace fs = std::filesystem;
using namespace annulus;

ProblemConfig DecodeConfig(const json& config, const std::string& archive = {});

namespace {

using Matrix = std::vector<std::vector<double>>;

// R and L of a coupling run, one matrix per frequency, from its HDF5 archive.
struct ImpedanceSweep {
	std::vector<Matrix> R, L;
};

ImpedanceSweep SolveImpedance(PhysicsSolver& solver, const std::string& archive) {
	solver.Setup();
	solver.Run();
	solver.SaveAnalysis();
	ImpedanceSweep sweep;
	HighFive::File file(archive, HighFive::File::ReadOnly);
	file.getDataSet("/coupling/Resistance/values").read(sweep.R);
	file.getDataSet("/coupling/Inductance/values").read(sweep.L);
	fs::remove(archive);
	return sweep;
}

json FrequencyScenarios(const std::vector<double>& frequencies) {
	json scenarios = json::array();
	for (double f : frequencies) {
		scenarios.push_back({{"name", std::to_string(f) + " Hz"}, {"frequency", f},
							 {"excitations", json::array()}});
	}
	return scenarios;
}

ImpedanceSweep AxisymmetricImpedance(const AnnulusSpec& spec, const std::vector<double>& f) {
	mfem::Mesh mesh = MakeAnnulus2D(spec, 4);
	json config = MakeAnnulusConfig(spec, false, 3, "magnetoquasistatics");
	config["scenarios"] = FrequencyScenarios(f);
	MagnetoquasistaticSolver solver(mesh, DecodeConfig(config, "mqs_axi.h5"));
	return SolveImpedance(solver, "mqs_axi.h5");
}

ImpedanceSweep Impedance3D(const AnnulusSpec& spec, const std::vector<double>& f,
						   const std::string& linear_solver = "direct") {
	mfem::Mesh mesh = MakeAnnulus3D(spec, 16);
	json config = MakeAnnulusConfig(spec, true, 2, "magnetoquasistatics");
	config["scenarios"] = FrequencyScenarios(f);
	config["simulation"]["linear_solver"] = linear_solver;
	config["simulation"]["solver_tolerance"] = 1e-12;
	MagnetoquasistaticSolver3D solver(mesh, DecodeConfig(config, "mqs3d.h5"));
	return SolveImpedance(solver, "mqs3d.h5");
}

// A massive ring (C1), a stranded coil (C2) and a passive shield (C3) in the
// annulus. sigma = 1e6 S/m puts the skin depth (1.1 cm at 2 kHz) on the scale
// of the conductors, so both frequencies carry real eddy-current effects.
AnnulusSpec EddyCurrentAnnulus() {
	AnnulusSpec spec;
	spec.sigma = 1e6;
	spec.conductors = {
		{ 0.04, 0.06, 0.02, 0.04, ConductorRole::Massive },
		{ 0.05, 0.08, 0.06, 0.08, ConductorRole::Stranded },
		{ 0.08, 0.09, 0.02, 0.08, ConductorRole::Passive } };
	return spec;
}

// Unit cube with a square bar (attribute 2, 0.25 < x, y < 0.75, sigma) from
// the bottom wall (z = 0, attribute 1) to the top (z = 1, attribute 6), all
// walls n x A = 0.
json MakeBarConfig(double sigma, const std::string& analysis) {
	return json{
		{"simulation", {
			{"physics_type", "magnetoquasistatics"}, {"mesh", "unused.mesh"}, {"order", 2},
			{"geometry_type", "3d"}, {"analysis_type", analysis}, {"linear_solver", "direct"}
		}},
		{"entity_groups", json::array({
			{{"name", "Air"}, {"dim", 3}, {"attribute_ids", {1}}},
			{{"name", "Bar"}, {"dim", 3}, {"attribute_ids", {2}}},
			{{"name", "Walls"}, {"dim", 2}, {"attribute_ids", {1, 2, 3, 4, 5, 6}}},
			{{"name", "Bottom"}, {"dim", 2}, {"attribute_ids", {1}}},
			{{"name", "Top"}, {"dim", 2}, {"attribute_ids", {6}}}})},
		{"regions", json::array({
			{{"name", "Air"}, {"entity_group", "Air"}, {"material", "Air"}},
			{{"name", "Bar"}, {"entity_group", "Bar"}, {"material", "Metal"}}})},
		{"materials", json::array({
			{{"name", "Air"}, {"properties", {{"mu_r", 1.0}}}},
			{{"name", "Metal"}, {"properties", {{"mu_r", 1.0}, {"sigma", sigma}}}}})},
		{"terminals", json::array({
			{{"name", "Bar"}, {"quantity", "current"}, {"entity_group", "Bar"},
			 {"conductor_type", "massive"},
			 {"direction", {{"type", "electrodes"}, {"input", "Bottom"}, {"output", "Top"}}}}})},
		{"boundary_conditions", json::array({
			{{"name", "Walls"}, {"type", "dirichlet"}, {"entity_group", "Walls"}, {"value", 0.0}}})},
		{"scenarios", json::array()}
	};
}

mfem::Mesh MakeBarMesh(int n) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(n, n, n, mfem::Element::HEXAHEDRON);
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

// The 3D annulus with n x A = 0 walls is the axisymmetric problem exactly
// (see AnnulusSpec), eddy currents included: the massive port's DC path is
// azimuthal, so its drive field V w = V phi-hat / (2 pi r) is the 2D solver's
// V / (2 pi r), and the passive shield's induced current is azimuthal too.
// The coupling matrices must agree up to discretization.
TEST_CASE("3D MQS impedances match the axisymmetric solver",
		  "[solvers][mqs][3d][coupling]") {
	const AnnulusSpec spec = EddyCurrentAnnulus();
	const std::vector<double> f = { 200.0, 2000.0 };
	const ImpedanceSweep axi = AxisymmetricImpedance(spec, f);
	const ImpedanceSweep z3d = Impedance3D(spec, f);

	for (size_t p = 0; p < f.size(); ++p) {
		for (int i = 0; i < 2; ++i) {
			for (int k = 0; k < 2; ++k) {
				INFO(f[p] << " Hz, (" << i << "," << k << "): R 3D " << z3d.R[p][i][k]
					 << " axi " << axi.R[p][i][k] << ", L 3D " << z3d.L[p][i][k]
					 << " axi " << axi.L[p][i][k]);
				REQUIRE(z3d.R[p][i][k] == Catch::Approx(axi.R[p][i][k]).epsilon(5e-3));
				REQUIRE(z3d.L[p][i][k] == Catch::Approx(axi.L[p][i][k]).epsilon(5e-3));
			}
		}
		REQUIRE(z3d.R[p][0][1] == Catch::Approx(z3d.R[p][1][0]).epsilon(1e-8));
		REQUIRE(z3d.L[p][0][1] == Catch::Approx(z3d.L[p][1][0]).epsilon(1e-8));
	}
	// Eddy currents in the ring and shield: the stranded coil sees resistance
	// that grows with frequency, and inductance that falls.
	REQUIRE(z3d.R[1][1][1] > z3d.R[0][1][1]);
	REQUIRE(z3d.L[1][1][1] < z3d.L[0][1][1]);
}

// At low frequency the ring is a DC resistor, R = 1/G with the closed-form
// conductance G = sigma h ln(r1/r0) / (2 pi) of an annular ring, and its
// inductance is 3D magnetostatics' for the same DC current distribution.
TEST_CASE("3D MQS approaches the DC limit", "[solvers][mqs][3d][coupling]") {
	AnnulusSpec spec;
	spec.sigma = 1e6;
	spec.conductors = { { 0.04, 0.06, 0.02, 0.04, ConductorRole::Massive } };
	const ImpedanceSweep z = Impedance3D(spec, { 1e-3 });

	const double G = spec.sigma * 0.02 * std::log(0.06 / 0.04) / Constants::TWO_PI;
	REQUIRE(z.R[0][0][0] == Catch::Approx(1.0 / G).epsilon(1e-6));

	mfem::Mesh mesh = MakeAnnulus3D(spec, 16);
	MagnetostaticSolver3D statics(mesh, DecodeConfig(MakeAnnulusConfig(spec, true, 2), "dc.h5"));
	statics.Setup();
	statics.Run();
	statics.SaveAnalysis();
	Matrix L;
	HighFive::File(std::string("dc.h5"), HighFive::File::ReadOnly)
		.getDataSet("/coupling/Inductance/values").read(L);
	fs::remove("dc.h5");
	REQUIRE(z.L[0][0][0] == Catch::Approx(L[0][0]).epsilon(1e-6));
}

// An open conductor between electrodes: R -> 1/G = length / (sigma area) at
// low frequency; at any frequency the dissipated power is the real input
// power 1/2 Re(V I*) = 1/2 R I^2. The discrete equations give that balance
// exactly (it is the real part of the Galerkin energy identity), so it holds
// to solver round-off whatever the resolution of the skin effect.
TEST_CASE("3D MQS electrode bar: DC resistance and power balance",
		  "[solvers][mqs][3d][loss]") {
	constexpr double sigma = 1e7;
	mfem::Mesh mesh = MakeBarMesh(4);

	{
		json config = MakeBarConfig(sigma, "coupling_matrix");
		// omega mu sigma a^2 = 2e-4: the eddy correction to R is ~1e-8.
		config["scenarios"] = FrequencyScenarios({ 1e-5 });
		mfem::Mesh copy(mesh);
		MagnetoquasistaticSolver3D solver(copy, DecodeConfig(config, "bar.h5"));
		const ImpedanceSweep z = SolveImpedance(solver, "bar.h5");
		REQUIRE(z.R[0][0][0] == Catch::Approx(1.0 / (sigma * 0.25)).epsilon(1e-7));
	}

	json config = MakeBarConfig(sigma, "field");
	config["scenarios"] = json::array({{{"name", "AC"}, {"frequency", 1000.0},
		{"excitations", json::array({{{"terminal", "Bar"}, {"value", 2.0}}})}}});
	MagnetoquasistaticSolver3D solver(mesh, DecodeConfig(config));
	solver.Setup();
	solver.Run();
	const std::complex<double> V = solver.GetPortVoltage("Bar");
	const auto losses = solver.ComputeRegionLosses();
	REQUIRE(losses.size() == 1);
	REQUIRE(losses[0].Name == "Bar");
	// Skin effect raises R above its DC value.
	REQUIRE(V.real() / 2.0 > 1.0001 / (sigma * 0.25));
	REQUIRE(losses[0].Power == Catch::Approx(0.5 * V.real() * 2.0).epsilon(1e-8));
}

TEST_CASE("3D MQS routing, exports and rejections", "[solvers][mqs][3d]") {
	using Catch::Matchers::ContainsSubstring;
	mfem::Mesh mesh = MakeBarMesh(4);
	json config = MakeBarConfig(1e6, "field");
	config["scenarios"] = json::array({{{"name", "AC"}, {"frequency", 50.0},
		{"excitations", json::array({{{"terminal", "Bar"}, {"value", 1.0}}})}}});

	SECTION("the factory routes '3d' MQS to the vector solver") {
		auto solver = SolverFactory::Instance().Create(mesh, DecodeConfig(config));
		auto* mqs = dynamic_cast<MagnetoquasistaticSolver3D*>(solver.get());
		REQUIRE(mqs != nullptr);
		solver->Setup();
		solver->Run();
		std::vector<std::string> names;
		const FieldExportSet fields = solver->CollectExportFields();
		for (const auto& field : fields.Fields()) { names.push_back(field.name); }
		for (const char* expected : { "A_Real", "A_Imag", "B_Real", "B_Imag", "B_Magnitude", "P_Loss" }) {
			REQUIRE(std::find(names.begin(), names.end(), expected) != names.end());
		}
		REQUIRE(mqs->ComputePeakFieldMagnitude() > 0.0);
	}

	SECTION("an open-current region is rejected") {
		config["regions"][1]["current_constraint"] = "open";
		config["terminals"] = json::array();
		config["scenarios"][0]["excitations"] = json::array();
		MagnetoquasistaticSolver3D solver(mesh, DecodeConfig(config));
		REQUIRE_THROWS_WITH(solver.Setup(), ContainsSubstring("does not support"));
	}

	SECTION("a massive conductor must conduct") {
		config["materials"][1]["properties"]["sigma"] = 0.0;
		MagnetoquasistaticSolver3D solver(mesh, DecodeConfig(config));
		REQUIRE_THROWS_WITH(solver.Setup(), ContainsSubstring("non-positive conductivity"));
	}
}

// A stranded conductor is a winding: its current is the imposed source alone,
// so its material's sigma (the wire's) must not add eddy currents to the field
// solve. The same winding in copper and in air gives the same impedance, and
// reports no loss. Checked in the axisymmetric model (a coil beside a massive
// ring) and in 3D (the electrode bar as a stranded conductor).
TEST_CASE("Stranded conductors carry no eddy currents", "[solvers][mqs][3d][2d][stranded]") {
	SECTION("axisymmetric") {
		AnnulusSpec spec;
		spec.sigma = 1e6;
		spec.conductors = {
			{ 0.04, 0.06, 0.02, 0.04, ConductorRole::Massive },
			{ 0.05, 0.08, 0.06, 0.08, ConductorRole::Stranded } };
		const std::vector<double> f = { 2000.0 };
		const ImpedanceSweep air = AxisymmetricImpedance(spec, f);

		// The coil's region is air in the fixture; make it copper.
		mfem::Mesh mesh = MakeAnnulus2D(spec, 4);
		json config = MakeAnnulusConfig(spec, false, 3, "magnetoquasistatics");
		config["scenarios"] = FrequencyScenarios(f);
		config["entity_groups"].push_back({{"name", "Winding"}, {"dim", 2}, {"attribute_ids", {3}}});
		for (auto& group : config["entity_groups"]) {
			if (group["name"] == "Air") group["attribute_ids"] = {1};
		}
		config["regions"].push_back({{"name", "Winding"}, {"entity_group", "Winding"},
									 {"material", "Copper"}});
		MagnetoquasistaticSolver solver(mesh, DecodeConfig(config, "stranded_cu.h5"));
		const ImpedanceSweep copper = SolveImpedance(solver, "stranded_cu.h5");
		for (int i = 0; i < 2; ++i) {
			for (int k = 0; k < 2; ++k) {
				REQUIRE(copper.R[0][i][k] == Catch::Approx(air.R[0][i][k]).epsilon(1e-12));
				REQUIRE(copper.L[0][i][k] == Catch::Approx(air.L[0][i][k]).epsilon(1e-12));
			}
		}
	}

	SECTION("3d") {
		json config = MakeBarConfig(1e7, "field");
		config["terminals"][0]["conductor_type"] = "stranded";
		config["scenarios"] = json::array({{{"name", "AC"}, {"frequency", 1000.0},
			{"excitations", json::array({{{"terminal", "Bar"}, {"value", 1.0}}})}}});
		json insulating = config;
		insulating["materials"][1]["properties"]["sigma"] = 0.0;

		mfem::Mesh mesh = MakeBarMesh(4), copy(mesh);
		MagnetoquasistaticSolver3D copper(mesh, DecodeConfig(config));
		MagnetoquasistaticSolver3D air(copy, DecodeConfig(insulating));
		for (auto* solver : { &copper, &air }) {
			solver->Setup();
			solver->Run();
		}
		REQUIRE(copper.ComputeRegionLosses().empty());
		mfem::Vector difference(copper.GetSolutionReal());
		difference -= air.GetSolutionReal();
		REQUIRE(difference.Normlinf() == 0.0);
		difference = copper.GetSolutionImag();
		difference -= air.GetSolutionImag();
		REQUIRE(difference.Normlinf() == 0.0);
	}
}

#ifdef MFEM_USE_MPI

// Both linear solvers solve the same regularized system, so they must agree
// to the iterative tolerance.
TEST_CASE("3D MQS impedances agree between the GMRES and direct solvers",
		  "[solvers][mqs][3d][coupling][ams]") {
	const AnnulusSpec spec = EddyCurrentAnnulus();
	const std::vector<double> f = { 2000.0 };
	const ImpedanceSweep direct = Impedance3D(spec, f, "direct");
	const ImpedanceSweep iterative = Impedance3D(spec, f, "iterative");
	for (int i = 0; i < 2; ++i) {
		for (int k = 0; k < 2; ++k) {
			INFO("(" << i << "," << k << ") R " << iterative.R[0][i][k] << " vs "
				 << direct.R[0][i][k] << ", L " << iterative.L[0][i][k] << " vs "
				 << direct.L[0][i][k]);
			REQUIRE(iterative.R[0][i][k] == Catch::Approx(direct.R[0][i][k]).epsilon(1e-7));
			REQUIRE(iterative.L[0][i][k] == Catch::Approx(direct.L[0][i][k]).epsilon(1e-7));
		}
	}
}

#endif // MFEM_USE_MPI
