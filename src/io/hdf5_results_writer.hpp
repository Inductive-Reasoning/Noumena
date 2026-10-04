#pragma once

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <highfive/H5File.hpp>
#include "field_export.hpp"
#include "../core/problem_config.hpp"

class Hdf5ResultsWriter {
public:
	Hdf5ResultsWriter(const std::filesystem::path& path, mfem::Mesh& mesh,
		const ProblemConfig& config)
		: file_(Open(path)), mesh_(mesh), order_(config.Order) {
		file_.createAttribute("schema_version", 2);
		file_.createAttribute("physics_type", std::string(ToString(config.PhysicsType)));
		file_.createAttribute("geometry_type", std::string(
			config.GeometryType == GeometryType::Axisymmetric ? "axisymmetric" : "planar"));
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
		// A curved (high-order) mesh carries its geometry as a nodal GridFunction;
		// export it like a field so readers can map reference points to space.
		if (const mfem::GridFunction* nodes = mesh.GetNodes()) {
			WriteField(mesh_group, "nodes", "geometry", *nodes);
		}
		file_.flush();
	}

	void WriteScenario(const std::string& id, const std::string& name,
		const Scenario& scenario, const FieldExportSet& fields,
		const std::string& driven_terminal = {}) {
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

	void WriteField(HighFive::Group parent, const std::string& name,
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
		dataset.createAttribute("space", WriteSpace(*space));
	}

	// Describes a field's scalar FE space for readers without MFEM: which scalar
	// DOFs each element uses, in element-local order, and where those local DOFs
	// sit on the reference element. Spaces depend only on the collection and the
	// mesh, so each is written once per archive and shared by every scenario.
	std::string WriteSpace(const mfem::FiniteElementSpace& space) {
		const std::string collection = space.FEColl()->Name();
		const std::string path = "/spaces/" + collection;
		if (!file_.exist("spaces")) file_.createGroup("spaces");
		auto spaces = file_.getGroup("spaces");
		if (spaces.exist(collection)) return path;

		std::vector<int> offsets{0}, dofs;
		std::map<int, const mfem::FiniteElement*> reference;
		mfem::Array<int> element_dofs;
		mfem::DofTransformation transformation;
		for (int element = 0; element < mesh_.GetNE(); ++element) {
			space.GetElementDofs(element, element_dofs, transformation);
			MFEM_VERIFY(transformation.IsIdentity(),
				"HDF5 export of space " + collection + " needs a DOF transformation.");
			dofs.insert(dofs.end(), element_dofs.begin(), element_dofs.end());
			offsets.push_back(static_cast<int>(dofs.size()));
			reference.emplace(static_cast<int>(mesh_.GetElementGeometry(element)), space.GetFE(element));
		}

		auto group = spaces.createGroup(collection);
		group.createAttribute("ndofs", space.GetNDofs());
		group.createDataSet("element_dof_offsets", offsets);
		group.createDataSet("element_dofs", dofs);
		auto nodes_group = group.createGroup("reference_nodes");
		for (const auto& [geometry, element] : reference) {
			// Only nodal, value-mapped elements have DOFs that are point values; a
			// reader cannot interpret any other basis from node positions alone.
			const auto* nodal = dynamic_cast<const mfem::NodalFiniteElement*>(element);
			if (!nodal || element->GetMapType() != mfem::FiniteElement::VALUE) continue;
			const mfem::IntegrationRule& nodes = element->GetNodes();
			const int dim = element->GetDim();
			std::vector<double> coordinates;
			for (int node = 0; node < nodes.GetNPoints(); ++node) {
				const mfem::IntegrationPoint& point = nodes.IntPoint(node);
				coordinates.push_back(point.x);
				if (dim > 1) coordinates.push_back(point.y);
				if (dim > 2) coordinates.push_back(point.z);
			}
			auto dataset = nodes_group.createDataSet<double>(std::to_string(geometry), HighFive::DataSpace(
				{static_cast<std::size_t>(nodes.GetNPoints()), static_cast<std::size_t>(dim)}));
			dataset.write_raw(coordinates.data());
			dataset.createAttribute("order", element->GetOrder());
		}
		return path;
	}
};