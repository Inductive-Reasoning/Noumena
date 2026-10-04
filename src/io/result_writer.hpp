#pragma once

#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <fstream>
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
		accepted_scenarios_ = 0;
		artifacts_.clear();
		if (config_.Output.ParaviewDirectory) artifacts_[*config_.Output.ParaviewDirectory];
		if (config_.Output.Gmsh) artifacts_[config_.Output.Gmsh->Directory];
		if (!config_.Output.Probes.empty()) artifacts_[config_.Output.ProbeDirectory];
		WriteStatus(false);
		hdf5_.reset();
		if (config_.Output.Hdf5File) {
			hdf5_ = std::make_unique<Hdf5ResultsWriter>(*config_.Output.Hdf5File, mesh_, config_);
		}
		// Located on every mesh: adaptive refinement replaces the elements.
		probes_ = std::make_unique<ProbeSampler>(mesh_, config_);
	}

	void Complete() {
		if (hdf5_) hdf5_->Complete();
		WriteStatus(true);
	}

	bool WantsFields() const {
		return (config_.AnalysisType != AnalysisType::CouplingMatrix || config_.Output.ExportFieldsForCouplingMatrix)
			&& (config_.Output.ParaviewDirectory || config_.Output.Gmsh || config_.Output.Hdf5File
				|| !config_.Output.Probes.empty());
	}

	void WriteScenario(const std::string& name, const Scenario& scenario,
		const FieldExportSet& fields, const std::string& driven_terminal = {},
		const std::vector<RegionLoss>& losses = {}) {
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
		if (WantsFields() && config_.Output.ParaviewDirectory) {
			fields_.WriteParaview(*config_.Output.ParaviewDirectory, artifact_name, fields);
			artifacts_[*config_.Output.ParaviewDirectory].push_back(artifact_name);
		}
		if (WantsFields() && config_.Output.Gmsh) {
			fields_.WriteGmsh(config_.Output.Gmsh->Directory / (artifact_name + ".msh"), fields,
				gmsh_results::ParseMshVersion(config_.Output.Gmsh->Version));
			artifacts_[config_.Output.Gmsh->Directory].push_back(artifact_name + ".msh");
		}
		std::vector<ProbeSamples> probes;
		if (WantsFields() && probes_ && !probes_->Empty()) {
			probes = probes_->Sample(fields);
			ProbeSampler::WriteCsv(config_.Output.ProbeDirectory, artifact_name, probes,
				config_.GeometryType == GeometryType::Axisymmetric);
			for (const auto& probe : probes)
				artifacts_[config_.Output.ProbeDirectory].push_back(artifact_name + "_" + probe.Name + ".csv");
		}
		if (hdf5_) hdf5_->WriteScenario(id, name, scenario, fields, probes, driven_terminal, losses);
		++accepted_scenarios_;
		WriteStatus(false);
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
	void WriteStatus(bool complete) const {
		// Sidecars also invalidate artifacts left over from a preceding AMR pass
		// or run when no HDF5 archive was requested.
		for (const auto& [directory, artifacts] : artifacts_) {
			std::filesystem::create_directories(directory);
			std::ofstream status(directory / "run_status.json", std::ios::trunc);
			status << "{\"complete\":" << (complete ? "true" : "false")
				<< ",\"accepted_scenarios\":" << accepted_scenarios_ << ",\"artifacts\":[";
			for (size_t i = 0; i < artifacts.size(); ++i) {
				if (i) status << ',';
				status << '"' << artifacts[i] << '"';
			}
			status << "]}\n";
			status.close();
			if (!status) throw std::runtime_error("Cannot write run status in " + directory.string());
		}
	}
	mfem::Mesh& mesh_;
	const ProblemConfig& config_;
	SolverFieldWriter fields_;
	std::unique_ptr<Hdf5ResultsWriter> hdf5_;
	std::unique_ptr<ProbeSampler> probes_;
	std::size_t next_scenario_ = 0;
	std::size_t accepted_scenarios_ = 0;
	std::map<std::filesystem::path, std::vector<std::string>> artifacts_;
};