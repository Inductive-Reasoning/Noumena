// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include "input_parser.hpp"

InputParser::InputParser(const std::string &filename)
    : owned_config(std::make_unique<json>()),
      config(*owned_config) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open config file: " + filename);
    }
    // Read the file once: the text is retained so error diagnostics can map
    // byte offsets to line/column without re-opening the file. Text mode
    // (no std::ios::binary) collapses CRLF to LF, matching the coordinate
    // system nlohmann uses while parsing; otherwise the column would be
    // skewed by one byte per preceding CRLF line ending on Windows.
    config_text.assign((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
    *owned_config = json::parse(config_text);

    // Store directory and path of config file (path used for diagnostics)
    config_path = filename;
    fs::path p(filename);
    config_dir = p.parent_path().string();
    if (config_dir.empty()) config_dir = ".";
}

const ProblemConfig InputParser::GetProblemConfig() const {
	try {
		return BuildProblemConfig();
	} catch (const json::exception& e) {
		// Re-throw with the config file path and, when available, a
		// resolved line/column so users can find the offending field.
		throw std::runtime_error(DescribeJsonError(e));
	}
}

const ProblemConfig InputParser::BuildProblemConfig() const {
	ProblemConfig prob_config;
	prob_config.Order = GetOrder();
	prob_config.PhysicsType = GetPhysicsType();
	prob_config.GeometryType = GetGeometryType();
	prob_config.AnalysisType = GetAnalysisType();
	prob_config.MeshPath = GetMeshPath();
        prob_config.Output = GetOutputSettings();

	prob_config.EntityGroups = GetEntityGroups();
	prob_config.Regions = GetRegions();
	prob_config.Materials = GetMaterials();
	prob_config.Terminals = GetTerminals();
	prob_config.BoundaryConditions = GetBoundaries();
	prob_config.Scenarios = GetScenarios();

	prob_config.SolverTolerance = GetSolverTolerance();
	prob_config.SolverMaxIter = GetSolverMaxIter();
	prob_config.SolverPrintLevel = GetSolverPrintLevel();
	prob_config.LinearSolver = GetLinearSolver();

	prob_config.Amr = GetAmrSettings();

	return prob_config;
}

std::string InputParser::DescribeJsonError(const json::exception& e) const {
	std::string what = e.what();
	std::string location;

	// Extract a "(bytes X-Y)" range emitted by JSON_DIAGNOSTIC_POSITIONS.
	const std::string marker = "(bytes ";
	const auto start = what.find(marker);
	if (start != std::string::npos) {
		const auto open = start + marker.size();
		const auto dash = what.find('-', open);
		if (dash != std::string::npos) {
			try {
				const std::size_t byte_off =
					static_cast<std::size_t>(std::stoull(what.substr(open, dash - open)));
				auto lc = ByteOffsetToLineCol(byte_off);
				if (lc.first > 0) {
					location = " (line " + std::to_string(lc.first) +
							   ", column " + std::to_string(lc.second) + ")";
				}
			} catch (const std::exception&) {
				// Leave location empty if the byte offset can't be parsed.
			}
		}
	}

	std::string prefix = "Failed to parse config";
	if (!config_path.empty()) {
		prefix += " '" + config_path + "'";
	}
	if (!location.empty()) {
		prefix += location;
	}
	return prefix + ": " + what;
}

std::pair<int, int> InputParser::ByteOffsetToLineCol(std::size_t byte_off) const {
	if (config_text.empty()) {
		return {0, 0};
	}
	if (byte_off >= config_text.size()) {
		byte_off = config_text.size() - 1;
	}
	int line = 1;
	int col = 1;
	for (std::size_t i = 0; i < byte_off; ++i) {
		if (config_text[i] == '\n') {
			++line;
			col = 1;
		} else {
			++col;
		}
	}
	return {line, col};
}

const json& InputParser::Sim() const {
    static const json empty = json::object();
    const auto it = config.find("simulation");
    return (it != config.end() && it->is_object()) ? *it : empty;
}

::PhysicsType InputParser::GetPhysicsType() const {
    return ParseEnum(Sim(), "physics_type", ::PhysicsType::Electrostatics,
                     {{"electrostatics",      ::PhysicsType::Electrostatics},
                      {"magnetostatics",      ::PhysicsType::Magnetostatics},
                      {"magnetoquasistatics", ::PhysicsType::Magnetoquasistatics}});
}

::GeometryType InputParser::GetGeometryType() const {
    return ParseEnum(Sim(), "geometry_type", ::GeometryType::Planar,
                     {{"axisymmetric", ::GeometryType::Axisymmetric},
                      {"planar",       ::GeometryType::Planar},
                      {"3d",           ::GeometryType::Cartesian3D}});
}

::AnalysisType InputParser::GetAnalysisType() const {
    return ParseEnum(Sim(), "analysis_type", ::AnalysisType::Field,
                     {{"field",           ::AnalysisType::Field},
                      {"coupling_matrix", ::AnalysisType::CouplingMatrix}});
}

OutputSettings InputParser::GetOutputSettings() const {
    OutputSettings output;
    const auto entry = config.find("output");
    const json empty = json::object();
    const json& settings = entry == config.end() ? empty : *entry;
    if (!settings.is_object()) {
        throw std::runtime_error("output must be an object");
    }
    const auto resolve = [](const fs::path& parent, const std::string& value) {
        const fs::path path(value);
        return (path.is_absolute() ? path : parent / path).lexically_normal();
    };
    output.Directory = resolve(config_dir, Get(settings, "directory", std::string("results")));
    output.ExportFieldsForCouplingMatrix = Get(settings, "export_fields_for_coupling_matrix", false);
    for (const char* format : {"paraview", "gmsh", "hdf5"}) {
        const auto target = settings.find(format);
        if (target == settings.end()) continue;
        if (!target->is_object()) {
            throw std::runtime_error(std::string("output.") + format + " must be an object");
        }
        if (std::string_view(format) == "paraview") {
            output.ParaviewDirectory = resolve(output.Directory,
                Get(*target, "directory", std::string("paraview")));
        } else if (std::string_view(format) == "gmsh") {
            output.Gmsh = GmshOutputSettings{resolve(output.Directory,
                Get(*target, "directory", std::string("gmsh"))),
                Get(*target, "version", std::string("2.2"))};
        } else {
            output.Hdf5File = resolve(output.Directory,
                Get(*target, "file", std::string("results.h5")));
        }
    }
    output.ProbeDirectory = resolve(output.Directory, "probes");
    if (const auto probes = settings.find("probes"); probes != settings.end()) {
        for (const json& entry : *probes) { output.Probes.push_back(GetProbe(entry)); }
    }
    return output;
}

 Probe InputParser::GetProbe(const json& entry) {
    Probe probe;
    probe.Name = entry.at("name").get<std::string>();
    probe.EntityGroupName = Get(entry, "entity_group", std::string{});
    if (const auto points = entry.find("points"); points != entry.end()) {
        for (const json& point : *points) {
            probe.Points.push_back(point.get<std::vector<double>>());
        }
    } else {
        const json& line = entry.at("line");
        const auto from = line.at("from").get<std::vector<double>>();
        const auto to = line.at("to").get<std::vector<double>>();
        const int count = line.at("count").get<int>();
        for (int i = 0; i < count; ++i) {
            const double t = static_cast<double>(i) / (count - 1);
            std::vector<double> point(from.size());
            for (size_t c = 0; c < from.size(); ++c) { point[c] = from[c] + t * (to[c] - from[c]); }
            probe.Points.push_back(point);
        }
    }
    return probe;
}

AmrSettings InputParser::GetAmrSettings() const {
    AmrSettings amr; // defaults (disabled)
    const auto it = Sim().find("amr");
    if (it == Sim().end() || !it->is_object()) return amr;

    const auto& a = *it;
    amr.Enabled        = Get(a, "enabled",         amr.Enabled);
    amr.MaxIterations  = Get(a, "max_iterations",  amr.MaxIterations);
    amr.MaxDofs        = Get(a, "max_dofs",        amr.MaxDofs);
    amr.ErrorFraction  = Get(a, "error_fraction",  amr.ErrorFraction);
    amr.ErrorTolerance = Get(a, "error_tolerance", amr.ErrorTolerance);
    amr.Conforming     = Get(a, "conforming",      amr.Conforming);
    return amr;
}

::LinearSolverType InputParser::GetLinearSolver() const {
    const bool three_d = GetGeometryType() == ::GeometryType::Cartesian3D;
    const bool magnetic = GetPhysicsType() != ::PhysicsType::Electrostatics;
    const bool iterative_available = !magnetic || parallel::Enabled();
    const ::LinearSolverType fallback = three_d && iterative_available
        ? ::LinearSolverType::Iterative : ::LinearSolverType::Direct;
    return ParseEnum(Sim(), "linear_solver", fallback,
                     {{"iterative", ::LinearSolverType::Iterative},
                      {"direct",    ::LinearSolverType::Direct}});
}

std::string InputParser::GetMeshPath() const {
    const std::string mesh = Get(Sim(), "mesh", std::string{});
    if (mesh.empty()) return "default.msh";
    const fs::path p(mesh);
    return p.is_absolute() ? p.string() : (fs::path(config_dir) / p).string();
}

std::unordered_map<std::string, EntityGroup> InputParser::GetEntityGroups() const {
	std::unordered_map<std::string, EntityGroup> groups;
	if (config.contains("entity_groups")) {
		for (const auto& g : config["entity_groups"]) {
			EntityGroup group;
			std::string name = g.value("name", std::string{});
			const std::string ctx = "entity group '" + name + "'";
			if (g.contains("kind")) {
				throw std::runtime_error(ctx + ": unsupported field 'kind'; use "
					"'dim' with the entity dimension (1 = curve, 2 = surface, "
					"3 = volume)");
			}
			if (!g.contains("dim")) {
				throw std::runtime_error(ctx + ": missing required field 'dim'");
			}
			if (!g["dim"].is_number_integer()) {
				throw std::runtime_error(ctx + ": field 'dim' must be an integer "
					"entity dimension (1 = curve, 2 = surface, 3 = volume)");
			}
			group.Dim = g["dim"].get<int>();
			if (group.Dim < 1 || group.Dim > 3) {
				throw std::runtime_error(ctx + ": invalid dim " +
					std::to_string(group.Dim) + "; must be 1, 2 or 3");
			}
			if (g.contains("attribute_ids")) {
				for (int attr : g["attribute_ids"]) {
					group.AttributeIds.push_back(attr);
				}
			}
			groups.emplace(std::move(name), std::move(group));
		}
	}
	return groups;
}

std::map<std::string, Terminal> InputParser::GetTerminals() const {
    std::map<std::string, Terminal> terminals;
    if (config.contains("terminals")) {
        for (auto& t : config["terminals"]) {
				Terminal terminal;
				std::string name = Get(t, "name", std::string{});
				terminal.DriveQuantity = ParseRequiredEnum(t, "quantity", Quantity::Voltage,
														   "terminal '" + name + "'",
														   {{"voltage", Quantity::Voltage},
															{"current", Quantity::Current}});
				terminal.Conductor = ParseEnum(t, "conductor_type", ConductorType::Massive,
											   {{"massive",  ConductorType::Massive},
												{"stranded", ConductorType::Stranded}});
				terminal.EntityGroupName = Get(t, "entity_group", std::string{});
				terminal.Turns = t.value("turns", 1.0);
				if (t.contains("direction")) {
					terminal.Direction = GetCurrentDirection(t["direction"]);
				}
				terminals.emplace(std::move(name), std::move(terminal));
        }
    }
    return terminals;
}

 CurrentDirection InputParser::GetCurrentDirection(const json& d) {
    CurrentDirection direction;
    direction.Type = ParseEnum(d, "type", CurrentDirection::Kind::Azimuthal,
                               {{"azimuthal",  CurrentDirection::Kind::Azimuthal},
                                {"electrodes", CurrentDirection::Kind::Electrodes},
                                {"cut",        CurrentDirection::Kind::Cut}});
    direction.Input = Get(d, "input", std::string{});
    direction.Output = Get(d, "output", std::string{});
    direction.Cut = Get(d, "cut", std::string{});
    auto read3 = [&d](const char* key, std::array<double, 3>& out) {
        if (!d.contains(key) || !d[key].is_array() || d[key].size() != 3) return;
        for (int c = 0; c < 3; ++c) out[c] = d[key][c].get<double>();
    };
    read3("origin", direction.Origin);
    read3("axis", direction.Axis);
    read3("normal", direction.Normal);
    return direction;
}

std::vector<Region> InputParser::GetRegions() const {
    std::vector<Region> regions;
    if (config.contains("regions")) {
        for (auto &region : config["regions"]) {
            Region _region;
            _region.EntityGroupName = Get(region, "entity_group", std::string{});
            _region.MaterialName = Get(region, "material", std::string{});
            _region.CurrentConstraint =
                ParseEnum(region, "current_constraint", RegionCurrentConstraint::None,
                          {{"none", RegionCurrentConstraint::None},
                           {"open", RegionCurrentConstraint::Open}});
            regions.push_back(_region);
        }
    }
    return regions;
}

std::map<std::string, Material> InputParser::GetMaterials() const {
    std::map<std::string, Material> materials;
    if (config.contains("materials")) {
        for (auto &material : config["materials"]) {
            Material _material;
            std::string name = Get(material, "name", std::string{});
            if (material.contains("properties")) {
                auto& props = material["properties"];
                if (props.contains("sigma")) {
                    _material.Conductivity = props["sigma"];
                }
                if (props.contains("epsilon_r")) {
                    _material.RelPermittivity = props["epsilon_r"];
                }
                if (props.contains("mu_r")) {
                    _material.RelPermeability = props["mu_r"];
                }
            }
            materials.emplace(std::move(name), std::move(_material));
        }
    }
    return materials;
}

std::vector<BoundaryCondition> InputParser::GetBoundaries() const {

    std::vector<BoundaryCondition> bcs;

    if (config.contains("boundary_conditions")) {
        for (auto &bc : config["boundary_conditions"]) {
            if (!bc.contains("type") || !bc.contains("value") || !bc.contains("entity_group")) {
                 continue; // Skip invalid entries, let validator handle reporting
            }
            std::string bc_type = bc["type"];
            std::string group_name = bc["entity_group"];
            double val = bc["value"];
            double robin_coeff = bc.value("robin_coefficient", 1.0);

            bcs.emplace_back(ParseBoundaryConditionType(bc_type), group_name,
                              val, robin_coeff);
        }
    }
    return bcs;
}

 BoundaryConditionType InputParser::ParseBoundaryConditionType(const std::string& type) {
    if (type == "dirichlet") return BoundaryConditionType::Dirichlet;
    if (type == "neumann") return BoundaryConditionType::Neumann;
    if (type == "robin") return BoundaryConditionType::Robin;
    throw std::invalid_argument("Unsupported boundary condition type: " + type);
}

 std::string InputParser::SweepScenarioName(const std::string& base_name,
                                     int point,
                                     double frequency) {
    std::ostringstream value;
    value << std::setprecision(12) << frequency;
    std::string token = value.str();
    for (char& c : token) {
        if (c == '.') c = 'p';
        else if (c == '+') c = '_';
        else if (c == '-') c = 'm';
    }
    return base_name + "_f" + std::to_string(point + 1) + "_" + token + "Hz";
}

 Scenario InputParser::ParseScenarioExcitations(const json& source) {
    Scenario scenario;
    if (source.contains("excitations")) {
        for (const auto& d : source["excitations"]) {
            Excitation excitation;
            if (d.contains("terminal")) {
                excitation.TerminalName = d["terminal"];
            }
            excitation.Value = d.value("value", 0.0);
            excitation.Phase = d.value("phase", 0.0);
            scenario.Excitations.push_back(excitation);
        }
    }
    return scenario;
}

std::vector<std::pair<std::string, Scenario>> InputParser::GetScenarios() const {
    std::vector<std::pair<std::string, Scenario>> scenarios;

    if (config.contains("scenarios")) {
        const bool is_mqs = GetPhysicsType() == PhysicsType::Magnetoquasistatics;
        for (const auto& sc : config["scenarios"]) {
            const std::string name = sc.value("name", "");
            Scenario scenario = ParseScenarioExcitations(sc);

            if (!is_mqs || !sc.contains("frequency")) {
                scenarios.emplace_back(name, std::move(scenario));
                continue;
            }

            const auto& frequency = sc["frequency"];
            if (frequency.is_number()) {
                scenario.Frequency = frequency.get<double>();
                scenarios.emplace_back(name, std::move(scenario));
                continue;
            }
            else if (frequency.is_array()) {
					for (size_t i = 0; i < frequency.size(); ++i) {
						Scenario point_scenario = scenario;
						point_scenario.Frequency = frequency[i].get<double>();
						scenarios.emplace_back(
							SweepScenarioName(name, static_cast<int>(i), point_scenario.Frequency),
							std::move(point_scenario));
					}
					continue;
            }

            const std::string scale = frequency.at("scale").get<std::string>();
            const double start = frequency.at("start").get<double>();
            const double stop = frequency.at("stop").get<double>();
            const int points = frequency.at("points").get<int>();
            const double log_start = std::log(start);
            const double log_stop = std::log(stop);

            for (int i = 0; i < points; ++i) {
                Scenario point_scenario = scenario;
                if (points == 1) {
                    point_scenario.Frequency = start;
                }
                else if (i == points - 1) {
                    point_scenario.Frequency = stop;
                }
                else {
                    const double fraction = static_cast<double>(i) / (points - 1);
                    point_scenario.Frequency = scale == "log"
                        ? std::exp(log_start + fraction * (log_stop - log_start))
                        : start + fraction * (stop - start);
                }
                scenarios.emplace_back(
                    SweepScenarioName(name, i, point_scenario.Frequency),
                    std::move(point_scenario));
            }
        }
    }
    return scenarios;
}
