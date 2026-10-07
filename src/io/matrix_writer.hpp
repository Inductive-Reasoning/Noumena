// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// Serializes a labeled dense matrix (e.g. a coupling matrix: capacitance,
// inductance, admittance, ...) to a formatted console table and/or a CSV file.
// Rows and columns carry names (typically terminal names) and the title holds
// the human-readable name plus unit. The writer computes nothing; callers hand
// it a finished matrix and its labels.

#pragma once

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "mfem.hpp"
#include "status_reporter.hpp"

namespace matrix_io {

class MatrixWriter {
	static constexpr int kColWidth = 18;

	std::string title_;
	std::vector<std::string> row_labels_;
	std::vector<std::string> col_labels_;

public:
	// Square matrix with identical row/column labels (the coupling-matrix case).
	MatrixWriter(std::string title, std::vector<std::string> labels)
		: title_(std::move(title)),
		  row_labels_(labels),
		  col_labels_(std::move(labels)) {}

	// Rectangular matrix with independent row/column labels.
	MatrixWriter(std::string title,
				 std::vector<std::string> row_labels,
				 std::vector<std::string> col_labels)
		: title_(std::move(title)),
		  row_labels_(std::move(row_labels)),
		  col_labels_(std::move(col_labels)) {}

	// Formatted table headed by the title and column labels, one labeled row per
	// matrix row. Values are printed in scientific notation.
	void PrintConsole(const mfem::DenseMatrix& M,
					  std::ostream& os = std::cout,
					  int precision = 6) const {
		CheckDimensions(M);
		// Restore the stream's format on return: os is usually std::cout.
		const struct FormatGuard {
			std::ostream& os;
			std::ios_base::fmtflags flags;
			std::streamsize precision;
			~FormatGuard() { os.flags(flags); os.precision(precision); }
		} guard{ os, os.flags(), os.precision() };

		os << "\n=== " << title_ << " ===\n";
		os << std::setw(kColWidth) << "";
		for (const auto& label : col_labels_) {
			os << std::setw(kColWidth) << label;
		}
		os << "\n";

		for (int r = 0; r < M.Height(); ++r) {
			os << std::setw(kColWidth) << row_labels_[r];
			for (int c = 0; c < M.Width(); ++c) {
				os << std::setw(kColWidth) << std::scientific
				   << std::setprecision(precision) << M(r, c);
			}
			os << "\n";
		}
	}

	// CSV with a header row (corner_label + column labels) and one labeled row
	// per matrix row. Returns false (and logs to stderr) if the file cannot be
	// opened.
	bool WriteCsv(const mfem::DenseMatrix& M,
				  const std::filesystem::path& path,
				  const std::string& corner_label = "Terminal",
				  int precision = 12) const {
		CheckDimensions(M);

		std::ofstream ofs(path);
		if (!ofs) {
			StatusReporter::Global().Error(
				"MatrixWriter: failed to open " + path.string());
			return false;
		}

		ofs << corner_label;
		for (const auto& label : col_labels_) { ofs << "," << label; }
		ofs << "\n";

		ofs << std::scientific << std::setprecision(precision);
		for (int r = 0; r < M.Height(); ++r) {
			ofs << row_labels_[r];
			for (int c = 0; c < M.Width(); ++c) { ofs << "," << M(r, c); }
			ofs << "\n";
		}
		ofs.close();
		if (ofs.fail()) {
			StatusReporter::Global().Error("MatrixWriter: writing " + path.string() + " failed");
			return false;
		}

		StatusReporter::Global().Diagnostic("Wrote " + path.string());
		return true;
	}

private:
	void CheckDimensions(const mfem::DenseMatrix& M) const {
		MFEM_VERIFY(M.Height() == static_cast<int>(row_labels_.size()),
			"MatrixWriter: row label count does not match matrix height.");
		MFEM_VERIFY(M.Width() == static_cast<int>(col_labels_.size()),
			"MatrixWriter: column label count does not match matrix width.");
	}
};

} // namespace matrix_io
