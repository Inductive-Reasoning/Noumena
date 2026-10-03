// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// TEAM Workshop Problem 7 (examples/team7) against its measurements: Bz above
// the plate and Jy on its surfaces, at 50 and 200 Hz. Run on demand
// ("[team7]"): about 15 minutes, with the iterative solver of the MPI build.
//
// Each column of the problem's tables (a line, frequency and phase) must stay
// within the agreement the converged solution reaches, as the RMS difference
// over the column relative to its largest measured magnitude. The bounds are
// the measured agreement plus a margin (see examples/team7/README.md for the
// numbers and for why the Jy tables are compared with the opposite surfaces
// to the ones their labels name).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "csv_rows.hpp"
#include "config/input_parser.hpp"
#include "io/mesh_loader.hpp"
#include "parallel/mpi_runtime.hpp"
#include "solvers/magnetoquasistatic_solver_3d.hpp"

namespace fs = std::filesystem;

namespace {

const fs::path kExample = fs::path(MFEM_ELECTROMAG_EXAMPLES) / "team7";

// A table's probe, the field component it compares, and the factor from SI
// to the table's unit.
struct Table {
	std::string probe, field, component;
	double factor;
};

} // namespace

TEST_CASE("TEAM 7 matches its measurements", "[.][team7][solvers][mqs][3d]") {
	if (!parallel::Enabled()) {
		SKIP("TEAM 7 needs the iterative solver of the MPI build.");
	}
	const fs::path output = fs::temp_directory_path() / "mfem_em_team7";
	fs::remove_all(output);

	std::ifstream file(kExample / "config.json");
	json config = json::parse(file);
	config["simulation"]["mesh"] = (kExample / "team7.msh").string();
	config["output"] = {{"directory", output.string()}, {"probes", config["output"]["probes"]}};
	const auto mesh = mesh_io::LoadMesh(config["simulation"]["mesh"].get<std::string>());
	MagnetoquasistaticSolver3D solver(*mesh, InputParser(config).GetProblemConfig());
	solver.Setup();
	solver.Run();
	solver.SaveAnalysis();

	const std::map<std::string, Table> tables = {
		{ "A1B1", { "A1B1", "B", "z", 1e4 } },  // gauss
		{ "A2B2", { "A2B2", "B", "z", 1e4 } },
		{ "A3B3", { "Bottom", "J", "y", 1e-6 } },  // 1e6 A/m^2
		{ "A4B4", { "Top", "J", "y", 1e-6 } } };
	const std::map<std::string, std::string> scenarios = {
		{ "f50", "scenario_000000_50Hz" }, { "f200", "scenario_000001_200Hz" } };
	// Bounds on RMS difference / max |measured|, by table, frequency, phase.
	const std::map<std::string, double> bound = {
		{ "A1B1 f50 wt0", 0.03 }, { "A1B1 f50 wt90", 0.04 },    // 1.4%, 2.2%
		{ "A1B1 f200 wt0", 0.03 }, { "A1B1 f200 wt90", 0.45 },  // 1.5%, 41%
		{ "A2B2 f50 wt0", 0.03 }, { "A2B2 f50 wt90", 0.04 },    // 1.6%, 2.6%
		{ "A2B2 f200 wt0", 0.03 }, { "A2B2 f200 wt90", 0.62 },  // 1.6%, 59%
		{ "A3B3 f50 wt0", 0.60 }, { "A3B3 f50 wt90", 0.55 },    // 56%, 51%
		{ "A3B3 f200 wt0", 0.17 }, { "A3B3 f200 wt90", 0.20 },  // 13%, 16%
		{ "A4B4 f50 wt0", 0.17 }, { "A4B4 f50 wt90", 0.21 },    // 13%, 17%
		{ "A4B4 f200 wt0", 0.17 }, { "A4B4 f200 wt90", 0.25 } };  // 14%, 22%

	const std::vector<CsvRow> measured = ReadCsvRows(kExample / "measured.csv");
	for (const auto& [name, table] : tables) {
		for (const auto& [frequency, scenario] : scenarios) {
			std::map<double, std::pair<double, double>> computed;  // x [mm] -> (wt0, wt90)
			for (const CsvRow& s : ReadCsvRows(output / "probes" / (scenario + "_" + table.probe + ".csv"))) {
				const double re = std::stod(s.at(table.field + "_Real_" + table.component));
				const double im = std::stod(s.at(table.field + "_Imag_" + table.component));
				const double x = std::round(1e6 * std::stod(s.at("x"))) / 1e3;
				computed[x] = { table.factor * re, -table.factor * im };  // Re(X e^{j wt})
			}
			for (const std::string phase : { "wt0", "wt90" }) {
				double sum = 0.0, peak = 0.0;
				int count = 0;
				for (const CsvRow& row : measured) {
					if (row.at("line") != name) continue;
					const auto at = computed.find(std::stod(row.at("x_mm")));
					REQUIRE(at != computed.end());
					const double want = std::stod(row.at(frequency + "_" + phase));
					const double got = phase == "wt0" ? at->second.first : at->second.second;
					sum += (got - want) * (got - want);
					peak = std::max(peak, std::abs(want));
					++count;
				}
				const std::string key = name + " " + frequency + " " + phase;
				const double error = std::sqrt(sum / count) / peak;
				INFO(key << ": RMS difference " << 100 * error << "% of the largest measured value");
				CHECK(error < bound.at(key));
			}
		}
	}
	fs::remove_all(output);
}
