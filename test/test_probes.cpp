// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// output.probes: parsing, validation, and sampling at points, including the
// side a point on a material interface is sampled from.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>

#include "config/config_validator.hpp"
#include "config/input_parser.hpp"
#include "solvers/electrostatic_solver.hpp"

namespace fs = std::filesystem;

ProblemConfig DecodeConfig(const json& config, const std::string& archive = {});

namespace {

// The unit square, y < 1/2 of permittivity 1 (attribute 1) and y > 1/2 of 3
// (attribute 2), at V = 0 on y = 0 and V = 1 on y = 1. D is uniform, so E_y
// is -3/2 below the interface and -1/2 above it, and V(1/2) = 3/4: all
// exactly representable at order 1.
mfem::Mesh MakeLayers() {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(4, 4, mfem::Element::QUADRILATERAL);
	for (int e = 0; e < mesh.GetNE(); ++e) {
		mfem::Vector c;
		mesh.GetElementCenter(e, c);
		mesh.SetAttribute(e, c(1) < 0.5 ? 1 : 2);
	}
	mesh.SetAttributes();
	return mesh;
}

json LayersConfig(const fs::path& directory, json probes) {
	return json{
		{"simulation", {
			{"physics_type", "electrostatics"}, {"mesh", "unused.mesh"}, {"order", 1},
			{"geometry_type", "planar"}, {"analysis_type", "field"}, {"linear_solver", "direct"}
		}},
		{"output", {{"directory", directory.string()}, {"probes", probes}, {"hdf5", json::object()}}},
		{"entity_groups", json::array({
			{{"name", "Lower"}, {"dim", 2}, {"attribute_ids", {1}}},
			{{"name", "Upper"}, {"dim", 2}, {"attribute_ids", {2}}},
			{{"name", "Bottom"}, {"dim", 1}, {"attribute_ids", {1}}},
			{{"name", "Top"}, {"dim", 1}, {"attribute_ids", {3}}}})},
		{"regions", json::array({
			{{"name", "Lower"}, {"entity_group", "Lower"}, {"material", "One"}},
			{{"name", "Upper"}, {"entity_group", "Upper"}, {"material", "Three"}}})},
		{"materials", json::array({
			{{"name", "One"}, {"properties", {{"epsilon_r", 1.0}}}},
			{{"name", "Three"}, {"properties", {{"epsilon_r", 3.0}}}}})},
		{"terminals", json::array()},
		{"boundary_conditions", json::array({
			{{"name", "Bottom"}, {"type", "dirichlet"}, {"entity_group", "Bottom"}, {"value", 0.0}},
			{{"name", "Top"}, {"type", "dirichlet"}, {"entity_group", "Top"}, {"value", 1.0}}})},
		{"scenarios", json::array({{{"name", "Layers"}, {"excitations", json::array()}}})}
	};
}

// The rows of a CSV file, split at commas; the first is the header.
std::vector<std::vector<std::string>> ReadCsv(const fs::path& path) {
	std::vector<std::vector<std::string>> rows;
	std::ifstream in(path);
	for (std::string line; std::getline(in, line);) {
		std::vector<std::string> cells;
		std::stringstream stream(line);
		for (std::string cell; std::getline(stream, cell, ',');) { cells.push_back(cell); }
		rows.push_back(cells);
	}
	return rows;
}

std::vector<std::string> ProbeErrors(const json& probes) {
	ConfigValidator validator;
	json config = LayersConfig("results", probes);
	validator.Validate(config);
	std::vector<std::string> fields;
	for (const auto& error : validator.GetErrors()) {
		if (error.field.rfind("output.probes", 0) == 0) { fields.push_back(error.field); }
	}
	return fields;
}

} // namespace

TEST_CASE("Probes are validated", "[probes][config_validator]") {
	const json line = {{"from", {0.0, 0.0}}, {"to", {1.0, 1.0}}, {"count", 3}};
	REQUIRE(ProbeErrors(json::array({{{"name", "ok"}, {"line", line}, {"entity_group", "Lower"}}})).empty());

	using List = std::vector<std::string>;
	REQUIRE(ProbeErrors(json::object()) == List{ "output.probes" });
	REQUIRE(ProbeErrors(json::array({{{"name", "a b"}, {"points", {{0.0, 0.0}}}}})) ==
			List{ "output.probes[0].name" });
	REQUIRE(ProbeErrors(json::array({{{"name", "p"}, {"points", {{0.0, 0.0}}}},
									  {{"name", "p"}, {"points", {{0.0, 0.0}}}}})) ==
			List{ "output.probes[1].name" });
	REQUIRE(ProbeErrors(json::array({{{"name", "p"}, {"points", {{0.0, 0.0}}}, {"line", line}}})) ==
			List{ "output.probes[0]" });
	REQUIRE(ProbeErrors(json::array({{{"name", "p"}, {"points", {{0.0, 0.0, 0.0}}}}})) ==
			List{ "output.probes[0].points[0]" });
	REQUIRE(ProbeErrors(json::array({{{"name", "p"}, {"line",
		{{"from", {0.0, 0.0}}, {"to", {1.0}}, {"count", 1}}}}})) ==
			List{ "output.probes[0].line.to", "output.probes[0].line.count" });
	REQUIRE(ProbeErrors(json::array({{{"name", "p"}, {"points", {{0.0, 0.0}}},
		{"entity_group", "Bottom"}, {"spacing", 1.0}}})) ==
			List{ "output.probes[0].spacing", "output.probes[0].entity_group" });
}

TEST_CASE("A probe line spans its ends evenly", "[probes][input_parser]") {
	const json config = LayersConfig("results", json::array({{{"name", "diagonal"},
		{"line", {{"from", {0.0, 1.0}}, {"to", {1.0, 0.0}}, {"count", 5}}}}}));
	const ProblemConfig decoded = InputParser(config).GetProblemConfig();
	REQUIRE(decoded.Output.Probes.size() == 1);
	const auto& points = decoded.Output.Probes[0].Points;
	REQUIRE(points.size() == 5);
	REQUIRE(points[1] == std::vector<double>{ 0.25, 0.75 });
	REQUIRE(points[4] == std::vector<double>{ 1.0, 0.0 });
}

// Probes report the exported fields exactly at their points, to CSV and to
// the archive; a point on the interface takes the values of the side its
// entity group names, and is rejected outside it.
TEST_CASE("Probes sample the exported fields at their points", "[probes][solvers]") {
	const fs::path directory = fs::temp_directory_path() / "mfem_em_probes";
	fs::remove_all(directory);
	const json probes = json::array({
		{{"name", "column"}, {"line", {{"from", {0.3, 0.1}}, {"to", {0.3, 0.9}}, {"count", 4}}}},
		{{"name", "below"}, {"points", {{0.7, 0.5}}}, {"entity_group", "Lower"}},
		{{"name", "above"}, {"points", {{0.7, 0.5}}}, {"entity_group", "Upper"}}});
	mfem::Mesh mesh = MakeLayers();
	ElectrostaticSolver solver(mesh, DecodeConfig(LayersConfig(directory, probes)));
	solver.Setup();
	solver.Run();
	solver.SaveAnalysis();

	const fs::path csv = directory / "probes";
	const auto column = ReadCsv(csv / "scenario_000000_Layers_column.csv");
	REQUIRE(column.front() == std::vector<std::string>{ "x", "y", "V", "E_x", "E_y", "Permittivity" });
	REQUIRE(column.size() == 5);
	for (size_t p = 1; p < column.size(); ++p) {
		const double y = std::stod(column[p][1]);
		const double V = y < 0.5 ? 1.5 * y : 0.75 + 0.5 * (y - 0.5);
		INFO("y = " << y);
		REQUIRE(std::stod(column[p][2]) == Catch::Approx(V).margin(1e-12));
		REQUIRE(std::stod(column[p][4]) == Catch::Approx(y < 0.5 ? -1.5 : -0.5).margin(1e-12));
	}
	REQUIRE(std::stod(ReadCsv(csv / "scenario_000000_Layers_below.csv")[1][4]) ==
			Catch::Approx(-1.5).margin(1e-12));
	REQUIRE(std::stod(ReadCsv(csv / "scenario_000000_Layers_above.csv")[1][4]) ==
			Catch::Approx(-0.5).margin(1e-12));

	HighFive::File file((directory / "results.h5").string(), HighFive::File::ReadOnly);
	std::vector<std::vector<double>> points, E;
	file.getDataSet("/scenarios/scenario_000000/probes/column/points").read(points);
	file.getDataSet("/scenarios/scenario_000000/probes/column/E").read(E);
	REQUIRE(points.size() == 4);
	REQUIRE(points[3] == std::vector<double>{ 0.3, 0.9 });
	REQUIRE(E[0][1] == Catch::Approx(-1.5).margin(1e-12));
	REQUIRE(E[3][1] == Catch::Approx(-0.5).margin(1e-12));
	fs::remove_all(directory);

	using Catch::Matchers::ContainsSubstring;
	const json outside = json::array({{{"name", "stray"}, {"points", {{0.7, 0.75}}},
									   {"entity_group", "Lower"}}});
	ElectrostaticSolver stray(mesh, DecodeConfig(LayersConfig(directory, outside)));
	stray.Setup();
	REQUIRE_THROWS_WITH(stray.Run(), ContainsSubstring("lies in no element of entity group 'Lower'"));
	fs::remove_all(directory);
}
