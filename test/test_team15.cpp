// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// TEAM Workshop Problem 15 (examples/team15) against its measurements: the
// change in a coil's impedance due to a slot in a thick aluminium plate, at
// three positions of problem 1's scan. Run on demand ("[team15]"): about 25
// minutes with the iterative solver of the MPI build. The meshes are not
// committed: each position's is generated with Gmsh, and the test skips
// without it.
//
// As in examples/team15/scan.py, each position is solved twice on one mesh,
// with the slot filled and empty, so the difference carries no meshing noise.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

#include "csv_rows.hpp"
#include "config/input_parser.hpp"
#include "io/mesh_loader.hpp"
#include "parallel/mpi_runtime.hpp"
#include "solvers/magnetoquasistatic_solver_3d.hpp"

namespace fs = std::filesystem;

namespace {

const fs::path kExample = fs::path(MFEM_ELECTROMAG_EXAMPLES) / "team15";

// Problem 1's coil has 3790 turns; the model is its y >= 0 half driven with
// one ampere-turn.
constexpr double kScale = 2.0 * 3790.0 * 3790.0;

// The positions checked: the slot's centre, the peak of dL, and the tail
// beyond the slot's end near the peak of the positive dR.
const std::vector<double> kPositionsMm = { 0.0, 9.0, 17.0 };

// Bounds on |computed - measured|: the solution is 47, 73 and 67 uH and
// 0.02, 0.11 and 0.16 ohm from the measurements there (see
// examples/team15/README.md).
constexpr double kDlBoundUh = 100.0;
constexpr double kDrBoundOhm = 0.25;

// L and R of the coil's half model per ampere-turn squared, with the slot of
// the given material.
std::pair<double, double> Impedance(const fs::path& work, const fs::path& mesh_file, double x,
									const std::string& slot_material) {
	std::ifstream file(kExample / "config-1.json");
	json config = json::parse(file);
	config["simulation"]["mesh"] = mesh_file.string();
	const fs::path output = work / ("results-" + slot_material);
	config["output"] = {{"directory", output.string()}, {"hdf5", json::object()}};
	config["terminals"][0]["direction"]["origin"] = {x, 0.0, 0.0};
	for (json& region : config["regions"]) {
		if (region["name"] == "Slot") region["material"] = slot_material;
	}
	const auto mesh = mesh_io::LoadMesh(mesh_file.string());
	MagnetoquasistaticSolver3D solver(*mesh, InputParser(config).GetProblemConfig());
	solver.Setup();
	solver.Run();
	solver.SaveAnalysis();
	std::vector<std::vector<std::vector<double>>> L, R;
	HighFive::File archive((output / "results.h5").string(), HighFive::File::ReadOnly);
	archive.getDataSet("/coupling/Inductance/values").read(L);
	archive.getDataSet("/coupling/Resistance/values").read(R);
	return { L[0][0][0], R[0][0][0] };
}

} // namespace

TEST_CASE("TEAM 15 matches its measurements", "[.][team15][solvers][mqs][3d]") {
	if (!parallel::Enabled()) {
		SKIP("TEAM 15 needs the iterative solver of the MPI build.");
	}
#ifndef MFEM_ELECTROMAG_GMSH
	SKIP("TEAM 15 generates its meshes with Gmsh, which CMake did not find.");
#else
	const fs::path work = fs::temp_directory_path() / "mfem_em_team15";
	fs::remove_all(work);
	fs::create_directories(work);
	const std::vector<CsvRow> measured = ReadCsvRows(kExample / "measured.csv");

	for (const double x_mm : kPositionsMm) {
		INFO("x = " << x_mm << " mm");
		const double x = x_mm / 1000.0;
		const fs::path mesh_file = work / "team15.msh";
		const std::string command = std::string("\"") + MFEM_ELECTROMAG_GMSH
			+ "\" -3 -format msh2 -setnumber problem 1 -setnumber X " + std::to_string(x) + " \""
			+ (kExample / "team15.geo").string() + "\" -o \"" + mesh_file.string() + "\" > \""
			+ (work / "gmsh.log").string() + "\" 2>&1";
		REQUIRE(std::system(command.c_str()) == 0);

		const auto [L0, R0] = Impedance(work, mesh_file, x, "Aluminium");
		const auto [L1, R1] = Impedance(work, mesh_file, x, "Air");
		const double dl = 1e6 * kScale * (L1 - L0), dr = kScale * (R1 - R0);

		bool found = false;
		for (const CsvRow& row : measured) {
			if (std::stoi(row.at("problem")) != 1 || std::abs(std::stod(row.at("x_mm")) - x_mm) > 1e-9) continue;
			found = true;
			const double want_dl = std::stod(row.at("dL_uH")), want_dr = std::stod(row.at("dR_ohm"));
			INFO("dL " << dl << " uH (measured " << want_dl << "), dR " << dr << " ohm (measured " << want_dr << ")");
			CHECK(std::abs(dl - want_dl) < kDlBoundUh);
			CHECK(std::abs(dr - want_dr) < kDrBoundOhm);
		}
		REQUIRE(found);
	}
	fs::remove_all(work);
#endif
}
