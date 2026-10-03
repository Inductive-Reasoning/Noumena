#pragma once

#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include "coupling_matrix_writer.hpp"
#include "hdf5_results_writer.hpp"
#include "probe_sampler.hpp"
#include "solver_field_writer.hpp"

class ResultWriter {
public:
	ResultWriter(mfem::Mesh& mesh, const ProblemConfig& config)
		: mesh_(mesh), config_(config), fields_(mesh, config.Order) {}

	void BeginMesh() {
		next_scenario_ = 0;
		hdf5_.reset();
		if (config_.Output.Hdf5File) {
			hdf5_ = std::make_unique<Hdf5ResultsWriter>(*config_.Output.Hdf5File, mesh_, config_);
		}
		// Located on every mesh: adaptive refinement replaces the elements.
		probes_ = std::make_unique<ProbeSampler>(mesh_, config_);
	}

	bool WantsFields() const {
		return (config_.AnalysisType != AnalysisType::CouplingMatrix || config_.Output.ExportFieldsForCouplingMatrix)
			&& (config_.Output.ParaviewDirectory || config_.Output.Gmsh || config_.Output.Hdf5File
				|| !config_.Output.Probes.empty());
	}

	void WriteScenario(const std::string& name, const Scenario& scenario,
		const FieldExportSet& fields, const std::string& driven_terminal = {},
		const std::vector<RegionLoss>& losses = {}) {
		if (!WantsFields()) return;
		std::ostringstream identifier;
		identifier << "scenario_" << std::setfill('0') << std::setw(6) << next_scenario_++;
		const std::string id = identifier.str();
		std::string label = name + (driven_terminal.empty() ? "" : "_" + driven_terminal);
		if (label.size() > 96) label.resize(96);
		for (char& character : label) {
			if (!((character >= 'a' && character <= 'z') ||
				(character >= 'A' && character <= 'Z') ||
				(character >= '0' && character <= '9') || character == '-' || character == '_')) {
				character = '_';
			}
		}
		const std::string artifact_name = id + "_" + label;
		if (config_.Output.ParaviewDirectory) {
			fields_.WriteParaview(*config_.Output.ParaviewDirectory, artifact_name, fields);
		}
		if (config_.Output.Gmsh) {
			fields_.WriteGmsh(config_.Output.Gmsh->Directory / (artifact_name + ".msh"), fields,
				gmsh_results::ParseMshVersion(config_.Output.Gmsh->Version));
		}
		std::vector<ProbeSamples> probes;
		if (probes_ && !probes_->Empty()) {
			probes = probes_->Sample(fields);
			ProbeSampler::WriteCsv(config_.Output.ProbeDirectory, artifact_name, probes,
				config_.GeometryType == GeometryType::Axisymmetric);
		}
		if (hdf5_) hdf5_->WriteScenario(id, name, scenario, fields, probes, driven_terminal, losses);
		StatusReporter::Global().Diagnostic("Wrote " + id + " for scenario '" + name + "'"
			+ (driven_terminal.empty() ? "" : ", terminal '" + driven_terminal + "'"));
	}

	std::optional<matrix_io::CouplingMatrixWriter> CouplingWriter(
		const std::vector<std::string>& terminals) {
		if (!hdf5_) return std::nullopt;
		return matrix_io::CouplingMatrixWriter(hdf5_->File(), terminals,
			ToString(config_.PhysicsType),
			ToString(config_.GeometryType));
	}

private:
	mfem::Mesh& mesh_;
	const ProblemConfig& config_;
	SolverFieldWriter fields_;
	std::unique_ptr<Hdf5ResultsWriter> hdf5_;
	std::unique_ptr<ProbeSampler> probes_;
	std::size_t next_scenario_ = 0;
};