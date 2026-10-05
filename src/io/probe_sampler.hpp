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
	ProbeSampler(mfem::Mesh& mesh, const ProblemConfig& config) : mesh_(mesh) {
		const std::vector<Box> boxes = ElementBoxes();
		for (const Probe& probe : config.Output.Probes) {
			std::set<int> attributes;
			if (!probe.EntityGroupName.empty()) {
				const auto group = config.EntityGroups.find(probe.EntityGroupName);
				MFEM_VERIFY(group != config.EntityGroups.end(),
					"Probe '" << probe.Name << "' names an unknown entity group '"
					<< probe.EntityGroupName << "'.");
				attributes.insert(group->second.AttributeIds.begin(), group->second.AttributeIds.end());
			}
			Located located{ &probe, {} };
			for (const std::vector<double>& point : probe.Points) {
				located.At.push_back(Locate(probe, point, attributes, boxes));
			}
			probes_.push_back(std::move(located));
		}
	}

	bool Empty() const { return probes_.empty(); }

	std::vector<ProbeSamples> Sample(const FieldExportSet& fields) const {
		std::vector<ProbeSamples> result;
		for (const Located& probe : probes_) {
			ProbeSamples samples;
			samples.Name = probe.Probe->Name;
			samples.Points = probe.Probe->Points;
			for (const FieldExport& field : fields.Fields()) {
				samples.Fields.push_back(SampleField(field, probe.At));
			}
			result.push_back(std::move(samples));
		}
		return result;
	}

	/// One CSV per probe, @p stem + "_" + probe name: a header row, then per
	/// point its coordinates and every field's components. Coordinates and
	/// components are named x, y, z, or r, z in an axisymmetric model.
	static void WriteCsv(const std::filesystem::path& directory, const std::string& stem,
						 const std::vector<ProbeSamples>& probes, bool axisymmetric) {
		std::filesystem::create_directories(directory);
		const std::vector<std::string> axes = axisymmetric
			? std::vector<std::string>{ "r", "z" } : std::vector<std::string>{ "x", "y", "z" };
		for (const ProbeSamples& probe : probes) {
			const std::filesystem::path path = directory / (stem + "_" + probe.Name + ".csv");
			std::ofstream out(path);
			MFEM_VERIFY(out, "Cannot write probe file " << path.string());
			const size_t dim = probe.Points.front().size();
			for (size_t c = 0; c < dim; ++c) { out << (c ? "," : "") << axes[c]; }
			for (const ProbeSamples::Field& field : probe.Fields) {
				for (int c = 0; c < field.VDim; ++c) {
					out << "," << field.Name << (field.VDim > 1 ? "_" + axes[c] : "");
				}
			}
			out << "\n" << std::setprecision(std::numeric_limits<double>::max_digits10);
			for (size_t p = 0; p < probe.Points.size(); ++p) {
				for (size_t c = 0; c < dim; ++c) { out << (c ? "," : "") << probe.Points[p][c]; }
				for (const ProbeSamples::Field& field : probe.Fields) {
					for (int c = 0; c < field.VDim; ++c) { out << "," << field.Values[p * field.VDim + c]; }
				}
				out << "\n";
			}
		}
	}

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

	mfem::Mesh& mesh_;
	std::vector<Located> probes_;

	// Bounding box of every element, from its map sampled on a reference
	// lattice (exact for straight elements), widened by a tenth of its size
	// to cover most of a curved element's bulge between the samples. The
	// boxes only order the search (Locate): a point they miss is still looked
	// for in every other element.
	std::vector<Box> ElementBoxes() const {
		const int dim = mesh_.SpaceDimension();
		std::vector<Box> boxes(mesh_.GetNE());
		mfem::DenseMatrix x;
		for (int e = 0; e < mesh_.GetNE(); ++e) {
			const mfem::RefinedGeometry& lattice =
				*mfem::GlobGeometryRefiner.Refine(mesh_.GetElementBaseGeometry(e), 3);
			mesh_.GetElementTransformation(e)->Transform(lattice.RefPts, x);
			Box& box = boxes[e];
			box.Low.SetSize(dim);
			box.High.SetSize(dim);
			for (int c = 0; c < dim; ++c) {
				double low = x(c, 0), high = x(c, 0);
				for (int i = 1; i < x.Width(); ++i) {
					low = std::min(low, x(c, i));
					high = std::max(high, x(c, i));
				}
				const double margin = 0.1 * (high - low);
				box.Low(c) = low - margin;
				box.High(c) = high + margin;
			}
		}
		return boxes;
	}

	Location Locate(const ::Probe& probe, const std::vector<double>& point,
					const std::set<int>& attributes, const std::vector<Box>& boxes) const {
		const int dim = mesh_.SpaceDimension();
		MFEM_VERIFY(static_cast<int>(point.size()) == dim,
			"Probe '" << probe.Name << "' has a point with " << point.size()
			<< " coordinates in a mesh of space dimension " << dim << ".");
		mfem::Vector x(dim);
		for (int c = 0; c < dim; ++c) { x(c) = point[c]; }
		// First the elements whose box holds the point, then, since a box can
		// miss part of a curved element, all the others.
		for (const bool in_box_pass : { true, false }) {
			for (int e = 0; e < mesh_.GetNE(); ++e) {
				if (!attributes.empty() && !attributes.count(mesh_.GetAttribute(e))) continue;
				bool inside_box = true;
				for (int c = 0; c < dim && inside_box; ++c) {
					inside_box = x(c) >= boxes[e].Low(c) && x(c) <= boxes[e].High(c);
				}
				if (inside_box != in_box_pass) continue;
				mfem::InverseElementTransformation inverse(mesh_.GetElementTransformation(e));
				Location location{ e, {} };
				if (inverse.Transform(x, location.Point) == mfem::InverseElementTransformation::Inside) {
					return location;
				}
			}
		}
		std::ostringstream where;
		for (int c = 0; c < dim; ++c) { where << (c ? ", " : "") << point[c]; }
		MFEM_ABORT("Probe '" << probe.Name << "': the point (" << where.str() << ") lies in no "
			<< (attributes.empty() ? "element of the mesh"
				: "element of entity group '" + probe.EntityGroupName + "'") << ".");
		return {};
	}

	ProbeSamples::Field SampleField(const FieldExport& field, const std::vector<Location>& at) const {
		ProbeSamples::Field samples;
		samples.Name = field.name;
		switch (field.kind) {
			case FieldExport::Kind::Primary: samples.VDim = field.primary->VectorDim(); break;
			case FieldExport::Kind::DerivedScalar: samples.VDim = 1; break;
			case FieldExport::Kind::DerivedVector: samples.VDim = field.vector->GetVDim(); break;
		}
		mfem::Vector value;
		for (const Location& location : at) {
			mfem::ElementTransformation& T = *mesh_.GetElementTransformation(location.Element);
			T.SetIntPoint(&location.Point);
			switch (field.kind) {
				case FieldExport::Kind::Primary:
					if (samples.VDim == 1) {
						samples.Values.push_back(field.primary->GetValue(T, location.Point));
					} else {
						field.primary->GetVectorValue(T, location.Point, value);
						samples.Values.insert(samples.Values.end(), value.begin(), value.end());
					}
					break;
				case FieldExport::Kind::DerivedScalar:
					samples.Values.push_back(field.scalar->Eval(T, location.Point));
					break;
				case FieldExport::Kind::DerivedVector:
					field.vector->Eval(value, T, location.Point);
					samples.Values.insert(samples.Values.end(), value.begin(), value.end());
					break;
			}
		}
		return samples;
	}
};
