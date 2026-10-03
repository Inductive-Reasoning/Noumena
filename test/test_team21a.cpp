// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// TEAM Workshop Problem 21a (examples/team21a) against its measurements: the
// eddy-current loss in a non-magnetic steel plate with 0-3 slits, and Bx
// beside the plate with two slits. Run on demand ("[team21a]"): about 15
// minutes with the iterative solver of the MPI build. The meshes are not
// committed: each variant's is generated with Gmsh, and the test skips
// without it.
//
// The bounds are the agreement the solution reaches plus a margin (see
// examples/team21a/README.md).

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

#include "csv_rows.hpp"
#include "config/input_parser.hpp"
#include "io/mesh_loader.hpp"
#include "parallel/mpi_runtime.hpp"
#include "solvers/magnetoquasistatic_solver_3d.hpp"

namespace fs = std::filesystem;

namespace {

const fs::path kExample = fs::path(MFEM_ELECTROMAG_EXAMPLES) / "team21a";

// Relative bound on |loss / measured - 1| per slit count (the converged
// solution is +2.4%, -0.9%, +1.4% and -19%: P21a-3 is low, as the problem's
// own calculation is, by 9%; see the README), and on the RMS difference of Bx
// over its line relative to the largest measured value (1.6% and 1.3%).
const std::map<int, double> kLossBound = { { 0, 0.035 }, { 1, 0.02 }, { 2, 0.025 }, { 3, 0.21 } };
constexpr double kBxBound = 0.025;

} // namespace

TEST_CASE("TEAM 21a matches its measurements", "[.][team21a][solvers][mqs][3d]") {
	if (!parallel::Enabled()) {
		SKIP("TEAM 21a needs the iterative solver of the MPI build.");
	}
#ifndef MFEM_ELECTROMAG_GMSH
	SKIP("TEAM 21a generates its meshes with Gmsh, which CMake did not find.");
#else
	const fs::path work = fs::temp_directory_path() / "mfem_em_team21a";
	fs::remove_all(work);
	fs::create_directories(work);
	const std::vector<CsvRow> measured = ReadCsvRows(kExample / "measured.csv");

	for (int slits = 0; slits <= 3; ++slits) {
		INFO("P21a-" << slits);
		const fs::path mesh_file = work / ("team21a-" + std::to_string(slits) + ".msh");
		const std::string command = std::string("\"") + MFEM_ELECTROMAG_GMSH + "\" -3 -format msh2 -setnumber slits "
			+ std::to_string(slits) + " \"" + (kExample / "team21a.geo").string() + "\" -o \""
			+ mesh_file.string() + "\" > \"" + (work / "gmsh.log").string() + "\" 2>&1";
		REQUIRE(std::system(command.c_str()) == 0);

		std::ifstream file(kExample / ("config-" + std::to_string(slits) + ".json"));
		json config = json::parse(file);
		config["simulation"]["mesh"] = mesh_file.string();
		const fs::path output = work / ("results-" + std::to_string(slits));
		config["output"] = {{"directory", output.string()}, {"probes", config["output"]["probes"]}};
		const auto mesh = mesh_io::LoadMesh(mesh_file.string());
		MagnetoquasistaticSolver3D solver(*mesh, InputParser(config).GetProblemConfig());
		solver.Setup();
		solver.Run();
		solver.SaveAnalysis();

		double plate = -1.0;
		for (const auto& loss : solver.ComputeRegionLosses()) {
			if (loss.Name == "Plate") plate = loss.Power;
		}
		for (const CsvRow& row : measured) {
			if (row.at("quantity") != "loss" || std::stoi(row.at("model")) != slits) continue;
			const double want = std::stod(row.at("measured_coil_side"));
			INFO("plate loss " << plate << " W, measured " << want << " W");
			CHECK(std::abs(plate / want - 1.0) < kLossBound.at(slits));
		}

		if (slits != 2) continue;
		// Bx is tabulated as rms; the solution's phasors are peak values.
		for (const auto& [probe, column] : { std::pair<std::string, std::string>{ "Bx_coil_side", "measured_coil_side" },
											 { "Bx_far_side", "measured_far_side" } }) {
			std::map<double, double> computed;
			for (const CsvRow& s : ReadCsvRows(output / "probes" / ("scenario_000000_50Hz_" + probe + ".csv"))) {
				computed[std::round(1e6 * std::stod(s.at("z"))) / 1e3] = 1e4 * std::stod(s.at("B_Real_x")) / std::sqrt(2.0);
			}
			double sum = 0.0, peak = 0.0;
			int count = 0;
			for (const CsvRow& row : measured) {
				if (row.at("quantity") != "Bx") continue;
				const auto at = computed.find(std::stod(row.at("z_mm")));
				REQUIRE(at != computed.end());
				const double want = std::stod(row.at(column));
				sum += (at->second - want) * (at->second - want);
				peak = std::max(peak, std::abs(want));
				++count;
			}
			const double error = std::sqrt(sum / count) / peak;
			INFO(probe << ": RMS difference " << 100 * error << "% of the largest measured value");
			CHECK(error < kBxBound);
		}
	}
	fs::remove_all(work);
#endif
}
