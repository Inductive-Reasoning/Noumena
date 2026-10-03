#pragma once

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <highfive/H5File.hpp>
#include "field_export.hpp"
#include "probe_sampler.hpp"
#include "region_loss.hpp"
#include "../core/problem_config.hpp"

class Hdf5ResultsWriter {
public:
	Hdf5ResultsWriter(const std::filesystem::path& path, mfem::Mesh& mesh,
		const ProblemConfig& config)
		: file_(Open(path)), mesh_(mesh), order_(config.Order) {
		file_.createAttribute("schema_version", 2);
		file_.createAttribute("physics_type", std::string(ToString(config.PhysicsType)));
		file_.createAttribute("geometry_type", std::string(ToString(config.GeometryType)));
		file_.createAttribute("analysis_type", std::string(
			config.AnalysisType == AnalysisType::CouplingMatrix ? "coupling_matrix" : "field"));
		auto mesh_group = file_.createGroup("mesh");
		mesh_group.createAttribute("dimension", mesh.Dimension());
		mesh_group.createAttribute("space_dimension", mesh.SpaceDimension());
		mesh_group.createAttribute("coordinate_units", std::string("m"));
		std::ostringstream serialized;
		serialized << std::setprecision(std::numeric_limits<mfem::real_t>::max_digits10);
		mesh.Print(serialized);
		mesh_group.createDataSet("mfem", serialized.str());
		std::vector<double> vertices;
		for (int vertex = 0; vertex < mesh.GetNV(); ++vertex) {
			for (int component = 0; component < mesh.SpaceDimension(); ++component) {
				vertices.push_back(mesh.GetVertex(vertex)[component]);
			}
		}
		auto coordinates = mesh_group.createDataSet<double>("vertices", HighFive::DataSpace(
			{static_cast<std::size_t>(mesh.GetNV()), static_cast<std::size_t>(mesh.SpaceDimension())}));
		coordinates.write_raw(vertices.data());
		WriteElements(mesh_group.createGroup("elements"), false);
		WriteElements(mesh_group.createGroup("boundary"), true);
		file_.flush();
	}

	void WriteScenario(const std::string& id, const std::string& name,
		const Scenario& scenario, const FieldExportSet& fields,
		const std::vector<ProbeSamples>& probes, const std::string& driven_terminal = {},
		const std::vector<RegionLoss>& losses = {}) {
		if (!file_.exist("scenarios")) file_.createGroup("scenarios");
		auto group = file_.getGroup("scenarios").createGroup(id);
		group.createAttribute("name", name);
		group.createAttribute("mesh", std::string("/mesh"));
		if (scenario.Frequency > 0.0) group.createAttribute("frequency_hz", scenario.Frequency);
		if (!driven_terminal.empty()) group.createAttribute("driven_terminal", driven_terminal);
		std::vector<std::string> terminals;
		std::vector<double> values;
		for (const auto& excitation : scenario.Excitations) {
			terminals.push_back(excitation.TerminalName);
			values.push_back(excitation.Value);
		}
		auto excitations = group.createGroup("excitations");
		excitations.createDataSet("terminal_names", terminals);
		excitations.createDataSet("values", values);
		auto field_group = group.createGroup("fields");
		mfem::L2_FECollection collection(std::max(0, order_ - 1), mesh_.Dimension());
		for (const auto& field : fields.Fields()) {
			if (field.kind == FieldExport::Kind::Primary) {
				WriteField(field_group, field.name, "primary", *field.primary);
			} else {
				const bool vector = field.kind == FieldExport::Kind::DerivedVector;
				mfem::FiniteElementSpace space(&mesh_, &collection, vector ? field.vector->GetVDim() : 1);
				mfem::GridFunction projected(&space);
				if (vector) projected.ProjectCoefficient(*field.vector);
				else projected.ProjectCoefficient(*field.scalar);
				WriteField(field_group, field.name, vector ? "vector" : "scalar", projected);
			}
		}
		if (!probes.empty()) WriteProbes(group.createGroup("probes"), probes);
		if (!losses.empty()) {
			// Time-averaged Joule loss of every conducting region [W].
			std::vector<std::string> names;
			std::vector<double> power;
			for (const RegionLoss& loss : losses) {
				names.push_back(loss.Name);
				power.push_back(loss.Power);
			}
			auto loss_group = group.createGroup("losses");
			loss_group.createDataSet("region_names", names);
			loss_group.createDataSet("power_w", power);
		}
		file_.flush();
	}

	const HighFive::File& File() const { return file_; }

private:
	HighFive::File file_;
	mfem::Mesh& mesh_;
	int order_;

	static HighFive::File Open(const std::filesystem::path& path) {
		if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
		return HighFive::File(path.string(), HighFive::File::Truncate);
	}

	void WriteElements(HighFive::Group group, bool boundary) const {
		std::vector<int> offsets{0}, vertices, attributes, geometry;
		const int count = boundary ? mesh_.GetNBE() : mesh_.GetNE();
		for (int index = 0; index < count; ++index) {
			const auto* element = boundary ? mesh_.GetBdrElement(index) : mesh_.GetElement(index);
			vertices.insert(vertices.end(), element->GetVertices(),
				element->GetVertices() + element->GetNVertices());
			offsets.push_back(static_cast<int>(vertices.size()));
			attributes.push_back(element->GetAttribute());
			geometry.push_back(static_cast<int>(element->GetGeometryType()));
		}
		group.createDataSet("offsets", offsets);
		group.createDataSet("vertices", vertices);
		group.createDataSet("attributes", attributes);
		group.createDataSet("geometry", geometry);
	}

	// Per probe: "points" (count x space dimension) and one dataset per field
	// (count x components), sampled exactly at the points.
	static void WriteProbes(HighFive::Group parent, const std::vector<ProbeSamples>& probes) {
		for (const ProbeSamples& probe : probes) {
			auto group = parent.createGroup(probe.Name);
			const std::size_t count = probe.Points.size();
			std::vector<double> points;
			for (const auto& point : probe.Points) points.insert(points.end(), point.begin(), point.end());
			group.createDataSet<double>("points",
				HighFive::DataSpace({count, probe.Points.front().size()})).write_raw(points.data());
			for (const ProbeSamples::Field& field : probe.Fields) {
				group.createDataSet<double>(field.Name,
					HighFive::DataSpace({count, static_cast<std::size_t>(field.VDim)}))
					.write_raw(field.Values.data());
			}
		}
	}

	static void WriteField(HighFive::Group parent, const std::string& name,
		const std::string& kind, const mfem::GridFunction& field) {
		std::vector<double> values(field.Size());
		for (int index = 0; index < field.Size(); ++index) values[index] = field(index);
		auto group = parent.createGroup(name);
		group.createAttribute("kind", kind);
		auto dataset = group.createDataSet("values", values);
		const auto* space = field.FESpace();
		dataset.createAttribute("finite_element_collection", std::string(space->FEColl()->Name()));
		dataset.createAttribute("vector_dimension", space->GetVDim());
		dataset.createAttribute("ordering", static_cast<int>(space->GetOrdering()));
	}
};