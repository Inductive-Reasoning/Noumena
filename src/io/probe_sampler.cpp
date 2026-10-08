// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include "probe_sampler.hpp"

ProbeSampler::ProbeSampler(mfem::Mesh& mesh, const ProblemConfig& config) : mesh_(mesh) {
	const std::vector<Box> boxes = ElementBoxes();
	const BoxGrid grid(boxes, mesh_.SpaceDimension());
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
			located.At.push_back(Locate(probe, point, attributes, boxes, grid));
		}
		probes_.push_back(std::move(located));
	}
}

std::vector<ProbeSamples> ProbeSampler::Sample(const FieldExportSet& fields) const {
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

 void ProbeSampler::WriteCsv(const std::filesystem::path& directory, const std::string& stem,
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
		out.close();
		MFEM_VERIFY(!out.fail(), "Writing probe file " << path.string() << " failed");
	}
}

ProbeSampler::BoxGrid::BoxGrid(const std::vector<Box>& boxes, int dim) : dim(dim) {
	if (boxes.empty()) { cells.resize(1); return; }
	double high[3] = { 0.0, 0.0, 0.0 };
	for (int c = 0; c < dim; ++c) {
		low[c] = std::numeric_limits<double>::max();
		high[c] = std::numeric_limits<double>::lowest();
		for (const Box& box : boxes) {
			low[c] = std::min(low[c], box.Low(c));
			high[c] = std::max(high[c], box.High(c));
		}
	}
	const int per_axis = std::max(1, static_cast<int>(
		std::round(std::pow(static_cast<double>(boxes.size()), 1.0 / dim))));
	for (int c = 0; c < dim; ++c) {
		n[c] = per_axis;
		size[c] = std::max(high[c] - low[c], std::numeric_limits<double>::min()) / n[c];
	}
	cells.resize(static_cast<size_t>(n[0]) * n[1] * n[2]);
	for (size_t e = 0; e < boxes.size(); ++e) {
		int from[3] = { 0, 0, 0 }, to[3] = { 0, 0, 0 };
		for (int c = 0; c < dim; ++c) {
			from[c] = Clamp(c, boxes[e].Low(c));
			to[c] = Clamp(c, boxes[e].High(c));
		}
		for (int i = from[0]; i <= to[0]; ++i)
			for (int j = from[1]; j <= to[1]; ++j)
				for (int k = from[2]; k <= to[2]; ++k)
					cells[Index(i, j, k)].push_back(static_cast<int>(e));
	}
}

const std::vector<int>& ProbeSampler::BoxGrid::Candidates(const mfem::Vector& x) const {
	static const std::vector<int> none;
	int at[3] = { 0, 0, 0 };
	for (int c = 0; c < dim; ++c) {
		const double t = (x(c) - low[c]) / size[c];
		if (t < 0.0 || t > n[c]) { return none; }
		at[c] = Clamp(c, x(c));
	}
	return cells[Index(at[0], at[1], at[2])];
}

std::vector<ProbeSampler::Box> ProbeSampler::ElementBoxes() const {
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

ProbeSampler::Location ProbeSampler::Locate(const ::Probe& probe, const std::vector<double>& point,
				const std::set<int>& attributes, const std::vector<Box>& boxes,
				const BoxGrid& grid) const {
	const int dim = mesh_.SpaceDimension();
	MFEM_VERIFY(static_cast<int>(point.size()) == dim,
		"Probe '" << probe.Name << "' has a point with " << point.size()
		<< " coordinates in a mesh of space dimension " << dim << ".");
	mfem::Vector x(dim);
	for (int c = 0; c < dim; ++c) { x(c) = point[c]; }
	const auto in_box = [&](int e) {
		for (int c = 0; c < dim; ++c) {
			if (x(c) < boxes[e].Low(c) || x(c) > boxes[e].High(c)) { return false; }
		}
		return true;
	};
	const auto contains = [&](int e, Location& location) {
		if (!attributes.empty() && !attributes.count(mesh_.GetAttribute(e))) { return false; }
		mfem::InverseElementTransformation inverse(mesh_.GetElementTransformation(e));
		location = { e, {} };
		return inverse.Transform(x, location.Point) == mfem::InverseElementTransformation::Inside;
	};
	// First the elements whose box holds the point (found through the
	// grid), then, since a box can miss part of a curved element, all the
	// others.
	Location location;
	for (const int e : grid.Candidates(x)) {
		if (in_box(e) && contains(e, location)) { return location; }
	}
	for (int e = 0; e < mesh_.GetNE(); ++e) {
		if (!in_box(e) && contains(e, location)) { return location; }
	}
	std::ostringstream where;
	for (int c = 0; c < dim; ++c) { where << (c ? ", " : "") << point[c]; }
	MFEM_ABORT("Probe '" << probe.Name << "': the point (" << where.str() << ") lies in no "
		<< (attributes.empty() ? "element of the mesh"
			: "element of entity group '" + probe.EntityGroupName + "'") << ".");
	return {};
}

ProbeSamples::Field ProbeSampler::SampleField(const FieldExport& field, const std::vector<Location>& at) const {
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
