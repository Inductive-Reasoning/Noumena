// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// ProbeSampler: evaluates a FieldExportSet at the configured probe points.
//
// Points are located once per mesh: the element containing each point, and
// the point's reference coordinates in it. A probe restricted to an entity
// group searches only that group's elements, so a point on a material
// interface -- a conductor's surface -- takes the values of the chosen side,
// where the fields jump. Sampling then evaluates every exported field there,
// primaries and derived coefficients alike, so a probe reports exactly what
// the other formats write.

#pragma once

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "mfem.hpp"
#include "field_export.hpp"
#include "../core/problem_config.hpp"

/// One probe's samples for one scenario: per field, the values at every point
/// (VDim per point, point-major).
struct ProbeSamples {
	struct Field {
		std::string Name;
		int VDim = 1;
		std::vector<double> Values;
	};
	std::string Name;
	std::vector<std::vector<double>> Points;
	std::vector<Field> Fields;
};

class ProbeSampler {
public:
	ProbeSampler(mfem::Mesh& mesh, const ProblemConfig& config);

	bool Empty() const { return probes_.empty(); }

	std::vector<ProbeSamples> Sample(const FieldExportSet& fields) const;

	/// One CSV per probe, @p stem + "_" + probe name: a header row, then per
	/// point its coordinates and every field's components. Coordinates and
	/// components are named x, y, z, or r, z in an axisymmetric model.
	static void WriteCsv(const std::filesystem::path& directory, const std::string& stem,
						 const std::vector<ProbeSamples>& probes, bool axisymmetric);

private:
	struct Location {
		int Element = -1;
		mfem::IntegrationPoint Point;
	};
	struct Located {
		const ::Probe* Probe;
		std::vector<Location> At;
	};
	struct Box {
		mfem::Vector Low, High;
	};

	// A uniform grid over the element boxes, about one element per cell, so a
	// point is tested only against the boxes overlapping its cell instead of
	// every element of the mesh.
	struct BoxGrid {
		int dim = 0;
		int n[3] = { 1, 1, 1 };
		double low[3] = { 0.0, 0.0, 0.0 }, size[3] = { 1.0, 1.0, 1.0 };
		std::vector<std::vector<int>> cells;

		BoxGrid(const std::vector<Box>& boxes, int dim);

		/// The elements whose boxes may hold @p x (none outside the grid).
		const std::vector<int>& Candidates(const mfem::Vector& x) const;

	private:
		int Clamp(int c, double v) const {
			const int i = static_cast<int>(std::floor((v - low[c]) / size[c]));
			return std::min(std::max(i, 0), n[c] - 1);
		}
		size_t Index(int i, int j, int k) const {
			return (static_cast<size_t>(k) * n[1] + j) * n[0] + i;
		}
	};

	mfem::Mesh& mesh_;
	std::vector<Located> probes_;

	// Bounding box of every element, from its map sampled on a reference
	// lattice (exact for straight elements), widened by a tenth of its size
	// to cover most of a curved element's bulge between the samples. The
	// boxes only order the search (Locate): a point they miss is still looked
	// for in every other element.
	std::vector<Box> ElementBoxes() const;

	Location Locate(const ::Probe& probe, const std::vector<double>& point,
					const std::set<int>& attributes, const std::vector<Box>& boxes,
					const BoxGrid& grid) const;

	ProbeSamples::Field SampleField(const FieldExport& field, const std::vector<Location>& at) const;
};
