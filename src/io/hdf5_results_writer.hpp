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
		const ProblemConfig& config);

	void WriteScenario(const std::string& id, const std::string& name,
		const Scenario& scenario, const FieldExportSet& fields,
		const std::vector<ProbeSamples>& probes, const std::string& driven_terminal = {},
		const std::vector<RegionLoss>& losses = {});

	const HighFive::File& File() const { return file_; }

private:
	HighFive::File file_;
	mfem::Mesh& mesh_;
	int order_;

	static HighFive::File Open(const std::filesystem::path& path) {
		if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
		return HighFive::File(path.string(), HighFive::File::Truncate);
	}

	void WriteElements(HighFive::Group group, bool boundary) const;

	// Per probe: "points" (count x space dimension) and one dataset per field
	// (count x components), sampled exactly at the points.
	static void WriteProbes(HighFive::Group parent, const std::vector<ProbeSamples>& probes);

	static void WriteField(HighFive::Group parent, const std::string& name,
		const std::string& kind, const mfem::GridFunction& field);
};