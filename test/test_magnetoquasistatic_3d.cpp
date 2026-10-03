// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// 3D time-harmonic magnetoquasistatics: massive ports, stranded sources and
// passive eddy-current conductors, checked against the axisymmetric solver on
// an equivalent geometry, against closed forms in the DC limit, against 3D
// magnetostatics at low frequency, and by the loss/impedance energy balance.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

#include "annulus_fixture.hpp"
#include "stderr_capture.hpp"
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

// 16 cells around the full ring, as many per radian in a sector.
int CellsAround(const AnnulusSpec& spec) {
	return static_cast<int>(std::lround(16 * spec.extent / Constants::TWO_PI));
}

ImpedanceSweep Impedance3D(const AnnulusSpec& spec, const std::vector<double>& f,
						   const std::string& linear_solver = "direct") {
	mfem::Mesh mesh = MakeAnnulus3D(spec, CellsAround(spec));
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

// How a test describes the annulus's terminals in 3D.
enum class Path { Azimuthal, Cut, SectorAzimuthal, SectorElectrodes };

std::string Describe(Path path) {
	switch (path) {
	case Path::Azimuthal: return "azimuthal ring";
	case Path::Cut: return "ring through a cut";
	case Path::SectorAzimuthal: return "azimuthal quarter sector";
	case Path::SectorElectrodes: return "quarter sector between electrodes";
	}
	return {};
}

// @p spec with its terminals described by @p path; returns the fraction of
// the ring the 3D model spans.
double Apply(Path path, AnnulusSpec& spec) {
	spec.cut = path == Path::Cut;
	spec.electrodes = path == Path::SectorElectrodes;
	const bool sector = path == Path::SectorAzimuthal || path == Path::SectorElectrodes;
	spec.extent = sector ? 0.25 * Constants::TWO_PI : Constants::TWO_PI;
	return spec.extent / Constants::TWO_PI;
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
// The coupling matrices must agree up to discretization, for the full ring
// and for a quarter sector bounded by n x A = 0 meridian planes, its paths
// described analytically or between electrodes on those planes (then solved
// on each conductor, the massive ring's with its own conductivity). A sector's
// matrices are the ring's times its fraction of the full turn. (A cut path
// is checked in the DC limit below: a massive path does not depend on
// frequency, and the full ring is the expensive model.)
TEST_CASE("3D MQS impedances match the axisymmetric solver",
		  "[solvers][mqs][3d][coupling][conductor]") {
	const Path path = GENERATE(Path::Azimuthal, Path::SectorAzimuthal, Path::SectorElectrodes);
	INFO(Describe(path));
	AnnulusSpec spec = EddyCurrentAnnulus();
	const double fraction = Apply(path, spec);
	const std::vector<double> f = { 200.0, 2000.0 };
	const ImpedanceSweep axi = AxisymmetricImpedance(spec, f);
	ImpedanceSweep z3d = Impedance3D(spec, f);
	for (auto* matrices : { &z3d.R, &z3d.L }) {
		for (Matrix& m : *matrices) {
			for (auto& row : m) {
				for (double& x : row) { x /= fraction; }
			}
		}
	}

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
// inductance is 3D magnetostatics' for the same DC current distribution --
// for the analytic path and for the one solved through a cut, whose
// discretization error leaves G a few parts in 1e5 off.
TEST_CASE("3D MQS approaches the DC limit", "[solvers][mqs][3d][coupling][conductor]") {
	const Path path = GENERATE(Path::Azimuthal, Path::Cut);
	INFO(Describe(path));
	AnnulusSpec spec;
	spec.sigma = 1e6;
	spec.conductors = { { 0.04, 0.06, 0.02, 0.04, ConductorRole::Massive } };
	Apply(path, spec);
	const ImpedanceSweep z = Impedance3D(spec, { 1e-3 });

	const double G = spec.sigma * 0.02 * std::log(0.06 / 0.04) / Constants::TWO_PI;
	const double tolerance = path == Path::Cut ? 1e-4 : 1e-6;
	REQUIRE(z.R[0][0][0] == Catch::Approx(1.0 / G).epsilon(tolerance));

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
	MagnetoquasistaticSolver3D solver(mesh, DecodeConfig(config, "bar_field.h5"));
	solver.Setup();
	solver.Run();
	solver.SaveAnalysis();
	const std::complex<double> V = solver.GetPortVoltage("Bar");
	const auto losses = solver.ComputeRegionLosses();
	REQUIRE(losses.size() == 1);
	REQUIRE(losses[0].Name == "Bar");
	// Skin effect raises R above its DC value.
	REQUIRE(V.real() / 2.0 > 1.0001 / (sigma * 0.25));
	REQUIRE(losses[0].Power == Catch::Approx(0.5 * V.real() * 2.0).epsilon(1e-8));

	// The archive records the same losses with the scenario.
	std::vector<std::string> names;
	std::vector<double> power;
	{
		HighFive::File file("bar_field.h5", HighFive::File::ReadOnly);
		file.getDataSet("/scenarios/scenario_000000/losses/region_names").read(names);
		file.getDataSet("/scenarios/scenario_000000/losses/power_w").read(power);
	}
	fs::remove("bar_field.h5");
	REQUIRE(names == std::vector<std::string>{ "Bar" });
	REQUIRE(power == std::vector<double>{ losses[0].Power });
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

	// n x A = 0 is a perfect electrical contact. The bar touches the walls
	// only at its electrodes, which is intended; as a passive conductor with
	// no terminal the same faces short it to the box.
	SECTION("a conductor touching an n x A = 0 wall is reported") {
		auto warnings = [&](const json& c) {
			StderrCapture capture;
			MagnetoquasistaticSolver3D solver(mesh, DecodeConfig(c));
			solver.Setup();
			return capture.Text();
		};
		REQUIRE(warnings(config).find("perfect electrical contact") == std::string::npos);
		json passive = config;
		passive["terminals"] = json::array();
		passive["scenarios"][0]["excitations"] = json::array();
		REQUIRE_THAT(warnings(passive), ContainsSubstring("Conductor 'Bar' touches"));
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

// In the weak-induction limit (omega mu sigma a^2 << 1) the eddy-current loss
// of a region is proportional to its sigma: the loss of a sigma = 1 S/m block
// must be exactly 1/100 of the same block's at 100 S/m. A block with ends (a
// 90-degree sector, not a ring) needs surface charge to turn its current, so
// it exposes any error in charge conservation. The static regularization
// weight put exactly such an error into weak conductors (68% at 1 S/m);
// scaled to the weakest conductor it vanishes.
TEST_CASE("3D MQS loss in a weak conductor scales with sigma", "[solvers][mqs][3d][loss]") {
	auto sector_loss = [](double sigma) {
		AnnulusSpec spec;
		spec.sigma = sigma;
		spec.conductors = {
			{ 0.04, 0.06, 0.06, 0.08, ConductorRole::Stranded },
			{ 0.04, 0.08, 0.02, 0.05, ConductorRole::Passive } };
		mfem::Mesh mesh = MakeAnnulus3D(spec, 16);
		for (int e = 0; e < mesh.GetNE(); ++e) {
			if (mesh.GetAttribute(e) != 3) continue;
			mfem::Vector c;
			mesh.GetElementCenter(e, c);
			if (!(c(0) > 0.0 && c(1) > 0.0)) mesh.SetAttribute(e, 1);  // keep one quadrant
		}
		mesh.SetAttributes();
		json config = MakeAnnulusConfig(spec, true, 1, "magnetoquasistatics", "field");
		config["scenarios"] = json::array({{{"name", "50 Hz"}, {"frequency", 50.0},
			{"excitations", json::array({{{"terminal", "C1"}, {"value", 1.0}}})}}});
		MagnetoquasistaticSolver3D solver(mesh, DecodeConfig(config));
		solver.Setup();
		solver.Run();
		const auto losses = solver.ComputeRegionLosses();
		REQUIRE(losses.size() == 1);
		return losses[0].Power;
	};
	const double weak = sector_loss(1.0), stronger = sector_loss(100.0);
	REQUIRE(weak > 0.0);
	REQUIRE(100.0 * weak == Catch::Approx(stronger).epsilon(1e-6));
}

// A massive port's load is not projected, so its DC current is checked on
// its own: an azimuthal direction about an axis 5 mm off the ring's leaves
// part of the current unbalanced, which is reported.
TEST_CASE("3D MQS reports a massive current that does not balance", "[solvers][mqs][3d]") {
	AnnulusSpec spec;
	spec.sigma = 1e6;
	spec.conductors = { { 0.04, 0.06, 0.04, 0.06, ConductorRole::Massive } };
	auto warnings = [&](double origin_x) {
		mfem::Mesh mesh = MakeAnnulus3D(spec, 16);
		json config = MakeAnnulusConfig(spec, true, 1, "magnetoquasistatics", "field");
		config["terminals"][0]["direction"]["origin"] = { origin_x, 0.0, 0.0 };
		config["scenarios"] = FrequencyScenarios({ 50.0 });
		StderrCapture capture;
		MagnetoquasistaticSolver3D solver(mesh, DecodeConfig(config));
		solver.Setup();
		return capture.Text();
	};
	REQUIRE(warnings(0.0).find("does not stay balanced") == std::string::npos);
	REQUIRE(warnings(0.005).find("'azimuthal' direction fits only") != std::string::npos);
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
