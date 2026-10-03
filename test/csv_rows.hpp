// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// Reading the small CSV files of the examples (measured data, probe output)
// in tests.

#pragma once

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using CsvRow = std::map<std::string, std::string>;

/// The rows of a CSV file with a header line, keyed by column name, skipping
/// '#' comment lines. Missing trailing cells read as empty.
inline std::vector<CsvRow> ReadCsvRows(const std::filesystem::path& path) {
	std::ifstream in(path);
	REQUIRE(in);
	std::vector<std::string> header;
	std::vector<CsvRow> rows;
	for (std::string line; std::getline(in, line);) {
		if (line.empty() || line[0] == '#') continue;
		std::vector<std::string> cells;
		std::stringstream stream(line);
		for (std::string cell; std::getline(stream, cell, ',');) { cells.push_back(cell); }
		if (header.empty()) { header = cells; continue; }
		CsvRow row;
		for (size_t c = 0; c < header.size(); ++c) { row[header[c]] = c < cells.size() ? cells[c] : ""; }
		rows.push_back(row);
	}
	return rows;
}
