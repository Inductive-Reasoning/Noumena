// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <functional>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "mfem.hpp"

using json = nlohmann::json;

/**
 * @brief Comprehensive configuration validator
 */
class ConfigValidator {
public:
    struct ValidationError {
        std::string field;
        std::string message;

        ValidationError(const std::string& f, const std::string& m)
            : field(f), message(m) {}
    };

private:
    std::vector<ValidationError> errors;

    // Entity group names declared in the "entity_groups" section, split by the
    // dimensionality the InputParser assigns (dim == 1 => boundary, otherwise
    // domain). Populated by ValidateEntityGroups and consulted when checking
    // that regions/terminals/boundaries reference a group of the right kind.
    std::set<std::string> entity_group_names_;
    std::set<std::string> boundary_group_names_;
    std::set<std::string> domain_group_names_;

    void AddError(const std::string& field, const std::string& message) {
        errors.emplace_back(field, message);
    }

    enum class ExpectedType { Object, Array, String, Number, Integer, Boolean };

    [[nodiscard]] static bool HasType(const json& value, ExpectedType type) {
        switch (type) {
            case ExpectedType::Object:  return value.is_object();
            case ExpectedType::Array:   return value.is_array();
            case ExpectedType::String:  return value.is_string();
            case ExpectedType::Number:  return value.is_number();
            case ExpectedType::Integer: return value.is_number_integer();
            case ExpectedType::Boolean: return value.is_boolean();
        }
        return false;
    }

    [[nodiscard]] static const char* TypeName(ExpectedType type) {
        switch (type) {
            case ExpectedType::Object:  return "an object";
            case ExpectedType::Array:   return "an array";
            case ExpectedType::String:  return "a string";
            case ExpectedType::Number:  return "a number";
            case ExpectedType::Integer: return "an integer";
            case ExpectedType::Boolean: return "a boolean";
        }
        return "the expected type";
    }

    void CheckFieldType(const json& object, const char* key,
                        const std::string& field, ExpectedType type) {
        if (object.contains(key) && !HasType(object[key], type)) {
            AddError(field, std::string("Must be ") + TypeName(type));
        }
    }

    void CheckObjectArrayTypes(
        const json& config, const char* key,
        const std::function<void(const json&, const std::string&)>& check_item) {
        if (!config.contains(key)) return;
        const auto& values = config[key];
        if (!values.is_array()) {
            AddError(key, "Must be an array");
            return;
        }
        for (size_t i = 0; i < values.size(); ++i) {
            const std::string prefix = std::string(key) + "[" + std::to_string(i) + "]";
            if (!values[i].is_object()) {
                AddError(prefix, "Must be an object");
                continue;
            }
            check_item(values[i], prefix);
        }
    }

    void ValidateOutput(const json& config) {
        if (!config.contains("output")) return;
        const auto& output = config["output"];
        if (!output.is_object()) {
            AddError("output", "Must be an object");
            return;
        }
        for (const auto& item : output.items()) {
            if (item.key() != "directory" && item.key() != "export_fields_for_coupling_matrix" &&
                item.key() != "paraview" && item.key() != "gmsh" && item.key() != "hdf5" &&
                item.key() != "probes") {
                AddError("output." + item.key(), "Unknown output setting");
            }
        }
        const auto check_path = [&](const json& object, const char* key,
                                    const std::string& field, bool file) {
            CheckFieldType(object, key, field, ExpectedType::String);
            if (!object.contains(key) || !object[key].is_string()) return;
            const std::string path = object[key];
            if (path.empty() || path.find('\0') != std::string::npos) {
                AddError(field, "Must be a nonempty path without null characters");
            } else if (file) {
                const auto filename = std::filesystem::path(path).filename();
                if (filename.empty() || filename == "." || filename == "..") {
                    AddError(field, "Must name a file, not a directory");
                }
            }
        };
        check_path(output, "directory", "output.directory", false);
        CheckFieldType(output, "export_fields_for_coupling_matrix", "output.export_fields_for_coupling_matrix", ExpectedType::Boolean);
        for (const std::string format : {"paraview", "gmsh", "hdf5"}) {
            if (!output.contains(format)) continue;
            const auto& target = output[format];
            const std::string prefix = "output." + format;
            if (!target.is_object()) {
                AddError(prefix, "Must be an object; omit the format to disable it");
                continue;
            }
            const char* path_key = format == "hdf5" ? "file" : "directory";
            check_path(target, path_key, prefix + "." + path_key, format == "hdf5");
            for (const auto& item : target.items()) {
                if (item.key() != path_key && !(format == "gmsh" && item.key() == "version")) {
                    AddError(prefix + "." + item.key(), "Unknown format setting");
                }
            }
            if (format == "gmsh") {
                CheckFieldType(target, "version", prefix + ".version", ExpectedType::String);
                if (target.contains("version") && target["version"].is_string() &&
                    target["version"] != "2.2" && target["version"] != "4.1") {
                    AddError(prefix + ".version", "Must be '2.2' or '4.1'");
                }
            }
        }
    }

    void ValidateDocumentTypes(const json& config) {
        if (!config.is_object()) {
            AddError("config", "Configuration root must be an object");
            return;
        }

        ValidateOutput(config);
        if (config.contains("simulation") && !config["simulation"].is_object()) {
            AddError("simulation", "Must be an object");
        } else if (config.contains("simulation")) {
            const auto& sim = config["simulation"];
            CheckFieldType(sim, "physics_type", "simulation.physics_type", ExpectedType::String);
            CheckFieldType(sim, "geometry_type", "simulation.geometry_type", ExpectedType::String);
            CheckFieldType(sim, "analysis_type", "simulation.analysis_type", ExpectedType::String);
            CheckFieldType(sim, "linear_solver", "simulation.linear_solver", ExpectedType::String);
            CheckFieldType(sim, "mesh", "simulation.mesh", ExpectedType::String);
            CheckFieldType(sim, "order", "simulation.order", ExpectedType::Integer);
            CheckFieldType(sim, "solver_tolerance", "simulation.solver_tolerance", ExpectedType::Number);
            CheckFieldType(sim, "solver_max_iter", "simulation.solver_max_iter", ExpectedType::Integer);
            CheckFieldType(sim, "solver_print_level", "simulation.solver_print_level", ExpectedType::Integer);
            CheckFieldType(sim, "amr", "simulation.amr", ExpectedType::Object);

            if (sim.contains("physics")) {
                AddError("simulation.physics", "Unsupported field; use 'physics_type'");
            }
            if (sim.contains("geometry")) {
                AddError("simulation.geometry", "Unsupported field; use 'geometry_type'");
            }
            if (sim.contains("type")) {
                AddError("simulation.type", "Unsupported field; use 'physics_type'");
            }
            for (const std::string key : {"output_paraview", "output_gmsh", "output_hdf5",
                                         "gmsh_format", "results_path", "results_file", "results_filename", "output"}) {
                if (sim.contains(key)) {
                    AddError("simulation." + key, "Unsupported field; configure the top-level 'output' object");
                }
            }

            if (sim.contains("amr") && sim["amr"].is_object()) {
                const auto& amr = sim["amr"];
                CheckFieldType(amr, "enabled", "simulation.amr.enabled", ExpectedType::Boolean);
                CheckFieldType(amr, "max_iterations", "simulation.amr.max_iterations", ExpectedType::Integer);
                CheckFieldType(amr, "max_dofs", "simulation.amr.max_dofs", ExpectedType::Integer);
                CheckFieldType(amr, "error_fraction", "simulation.amr.error_fraction", ExpectedType::Number);
                CheckFieldType(amr, "error_tolerance", "simulation.amr.error_tolerance", ExpectedType::Number);
                CheckFieldType(amr, "conforming", "simulation.amr.conforming", ExpectedType::Boolean);
            }
        }

        if (config.contains("boundaries")) {
            AddError("boundaries", "Unsupported section; use 'boundary_conditions'");
        }

        CheckObjectArrayTypes(config, "entity_groups", [&](const json& group, const std::string& prefix) {
            CheckFieldType(group, "name", prefix + ".name", ExpectedType::String);
            CheckFieldType(group, "dim", prefix + ".dim", ExpectedType::Integer);
            if (group.contains("kind")) {
                AddError(prefix + ".kind", "Unsupported field; use 'dim' with the entity "
                        "dimension (1 = curve, 2 = surface, 3 = volume)");
            }
            CheckFieldType(group, "attribute_ids", prefix + ".attribute_ids", ExpectedType::Array);
            if (group.contains("attribute_ids") && group["attribute_ids"].is_array()) {
                for (size_t i = 0; i < group["attribute_ids"].size(); ++i) {
                    if (!group["attribute_ids"][i].is_number_integer()) {
                        AddError(prefix + ".attribute_ids[" + std::to_string(i) + "]", "Must be an integer");
                    }
                }
            }
        });

        CheckObjectArrayTypes(config, "regions", [&](const json& region, const std::string& prefix) {
            CheckFieldType(region, "entity_group", prefix + ".entity_group", ExpectedType::String);
            CheckFieldType(region, "material", prefix + ".material", ExpectedType::String);
            CheckFieldType(region, "current_constraint", prefix + ".current_constraint", ExpectedType::String);
        });

        CheckObjectArrayTypes(config, "materials", [&](const json& material, const std::string& prefix) {
            CheckFieldType(material, "name", prefix + ".name", ExpectedType::String);
            CheckFieldType(material, "properties", prefix + ".properties", ExpectedType::Object);
            if (material.contains("properties") && material["properties"].is_object()) {
                const auto& props = material["properties"];
                CheckFieldType(props, "sigma", prefix + ".properties.sigma", ExpectedType::Number);
                CheckFieldType(props, "epsilon_r", prefix + ".properties.epsilon_r", ExpectedType::Number);
                CheckFieldType(props, "mu_r", prefix + ".properties.mu_r", ExpectedType::Number);
            }
        });

        CheckObjectArrayTypes(config, "terminals", [&](const json& terminal, const std::string& prefix) {
            CheckFieldType(terminal, "name", prefix + ".name", ExpectedType::String);
            CheckFieldType(terminal, "quantity", prefix + ".quantity", ExpectedType::String);
            if (terminal.contains("excitation_type")) {
                AddError(prefix + ".excitation_type", "Unsupported field; use 'quantity'");
            }
            if (terminal.contains("excitation")) {
                AddError(prefix + ".excitation", "Unsupported field; use 'quantity'");
            }
            CheckFieldType(terminal, "conductor_type", prefix + ".conductor_type", ExpectedType::String);
            CheckFieldType(terminal, "entity_group", prefix + ".entity_group", ExpectedType::String);
            CheckFieldType(terminal, "direction", prefix + ".direction", ExpectedType::Object);
        });

        CheckObjectArrayTypes(config, "boundary_conditions", [&](const json& boundary, const std::string& prefix) {
            CheckFieldType(boundary, "type", prefix + ".type", ExpectedType::String);
            CheckFieldType(boundary, "entity_group", prefix + ".entity_group", ExpectedType::String);
            CheckFieldType(boundary, "value", prefix + ".value", ExpectedType::Number);
            CheckFieldType(boundary, "robin_coefficient", prefix + ".robin_coefficient", ExpectedType::Number);
        });

        CheckObjectArrayTypes(config, "scenarios", [&](const json& scenario, const std::string& prefix) {
            CheckFieldType(scenario, "name", prefix + ".name", ExpectedType::String);
            CheckFieldType(scenario, "excitations", prefix + ".excitations", ExpectedType::Array);
            if (scenario.contains("frequency")) {
                const auto& frequency = scenario["frequency"];
                if (frequency.is_object()) {
                    CheckFieldType(frequency, "scale", prefix + ".frequency.scale", ExpectedType::String);
                    CheckFieldType(frequency, "start", prefix + ".frequency.start", ExpectedType::Number);
                    CheckFieldType(frequency, "stop", prefix + ".frequency.stop", ExpectedType::Number);
                    CheckFieldType(frequency, "points", prefix + ".frequency.points", ExpectedType::Integer);
                }
				else if (frequency.is_array()) {
					for (size_t i = 0; i < frequency.size(); ++i) {
						if (!frequency[i].is_number()) {
							AddError(prefix + ".frequency[" + std::to_string(i) + "]", "Must be a number");
						}
					}
				}
                else if (!frequency.is_number()) {
                    AddError(prefix + ".frequency", "Must be a number or a sweep object");
                }
            }
            if (scenario.contains("excitations") && scenario["excitations"].is_array()) {
                for (size_t i = 0; i < scenario["excitations"].size(); ++i) {
                    const auto& excitation = scenario["excitations"][i];
                    const std::string eprefix = prefix + ".excitations[" + std::to_string(i) + "]";
                    if (!excitation.is_object()) {
                        AddError(eprefix, "Must be an object");
                        continue;
                    }
                    CheckFieldType(excitation, "terminal", eprefix + ".terminal", ExpectedType::String);
                    CheckFieldType(excitation, "value", eprefix + ".value", ExpectedType::Number);
                }
            }
        });
    }

    // "simulation.physics_type" as a string, or "" when absent.
    [[nodiscard]] std::string PhysicsType(const json& config) const {
        if (config.contains("simulation") && config["simulation"].is_object()) {
            return config["simulation"].value("physics_type", std::string{});
        }
        return {};
    }

    // Which kind of entity group a reference is required to name.
    enum class RequiredGroupKind { Any, Boundary, Domain };

    // Verify that an "entity_group" reference points at a declared group and,
    // when required is not Any, that the group is of the matching kind. Mirrors
    // how the solvers resolve EntityGroupName -> EntityGroup via
    // config.EntityGroups.at(name).
    void CheckEntityGroupRef(const json& node, const std::string& field, RequiredGroupKind required) {
        if (!node.is_string()) {
            AddError(field, "Entity group reference must be a string");
            return;
        }
        const std::string ref = node.get<std::string>();
        if (entity_group_names_.find(ref) == entity_group_names_.end()) {
            AddError(field, "Unknown entity group '" + ref +
                    "'. No entity group with that name is declared in 'entity_groups'");
            return;
        }
        if (required == RequiredGroupKind::Boundary && boundary_group_names_.count(ref) == 0) {
            AddError(field, "Entity group '" + ref + "' must be a boundary group (kind = 'boundary')");
        } else if (required == RequiredGroupKind::Domain && domain_group_names_.count(ref) == 0) {
            AddError(field, "Entity group '" + ref + "' must be a domain group (kind = 'domain')");
        }
    }

    void ValidateSimulation(const json& config) {
        if (!config.contains("simulation")) {
            AddError("simulation", "Missing required 'simulation' section");
            return;
        }

        const auto& sim = config["simulation"];

        // Required fields
        if (!sim.contains("physics_type")) {
            AddError("simulation.physics_type", "Missing required field 'physics_type'");
        } else {
            std::string type = sim["physics_type"];
            if (type != "electrostatics" && type != "magnetostatics" && type != "magnetoquasistatics") {
                AddError("simulation.physics_type", "Invalid physics_type '" + type + "'. Must be 'electrostatics', 'magnetostatics', or 'magnetoquasistatics'");
            }

            if (type == "magnetoquasistatics" && sim.contains("frequency")) {
                AddError("simulation.frequency",
                    "Unsupported for magnetoquasistatics; define 'frequency' on each scenario");
            }
        }

        if (!sim.contains("mesh")) {
            AddError("simulation.mesh", "Missing required field 'mesh'");
        }

        // Optional enumerated fields
        if (sim.contains("geometry_type")) {
            std::string g = sim["geometry_type"];
            if (g != "axisymmetric" && g != "planar" && g != "3d") {
                AddError("simulation.geometry_type", "Invalid geometry_type '" + g + "'. Must be 'axisymmetric', 'planar', or '3d'");
            }
        }

        if (sim.contains("linear_solver")) {
            std::string solver = sim["linear_solver"];
            if (solver != "direct" && solver != "iterative") {
                AddError("simulation.linear_solver", "Invalid linear_solver '" + solver +
                    "'. Must be 'direct' or 'iterative'");
            }
        }

        if (sim.contains("analysis_type")) {
            std::string a = sim["analysis_type"];
            if (a != "field" && a != "coupling_matrix") {
                AddError("simulation.analysis_type", "Invalid analysis_type '" + a + "'. Must be 'field' or 'coupling_matrix'");
            }
        }

        // Optional fields with validation
        if (sim.contains("order")) {
            int order = sim["order"];
            if (order < 1 || order > 10) {
                AddError("simulation.order", "Order must be between 1 and 10");
            }
        }

        if (sim.contains("solver_tolerance")) {
            double tol = sim["solver_tolerance"];
            if (tol <= 0 || tol >= 1) {
                AddError("simulation.solver_tolerance", "Solver tolerance must be between 0 and 1");
            }
        }

        if (sim.contains("solver_max_iter")) {
            int max_iter = sim["solver_max_iter"];
            if (max_iter < 1) {
                AddError("simulation.solver_max_iter", "Max iterations must be at least 1");
            }
        }

        if (sim.contains("amr")) {
            const auto& amr = sim["amr"];
            if (amr.contains("max_iterations") && amr["max_iterations"].get<int>() < 0) {
                AddError("simulation.amr.max_iterations", "Maximum iterations cannot be negative");
            }
            if (amr.contains("error_fraction")) {
                const double fraction = amr["error_fraction"];
                if (fraction <= 0.0 || fraction > 1.0) {
                    AddError("simulation.amr.error_fraction", "Error fraction must be in (0, 1]");
                }
            }
            if (amr.contains("error_tolerance") && amr["error_tolerance"].get<double>() < 0.0) {
                AddError("simulation.amr.error_tolerance", "Error tolerance cannot be negative");
            }
        }
    }

    // output.probes: named point sets, each "points" (coordinate arrays) or a
    // "line" {"from", "to", "count" >= 2}, optionally restricted to a domain
    // entity group. Coordinates have the model's space dimension. Names become
    // file names, so they are restricted to [A-Za-z0-9_-].
    void ValidateProbes(const json& config, const mfem::Mesh* mesh) {
        if (!config.contains("output") || !config["output"].contains("probes")) return;
        const auto& probes = config["output"]["probes"];
        if (!probes.is_array()) {
            AddError("output.probes", "Must be an array of probes");
            return;
        }
        int dim = 2;
        if (mesh) {
            dim = mesh->SpaceDimension();
        } else if (config.contains("simulation") && config["simulation"].value("geometry_type", "") == "3d") {
            dim = 3;
        }
        const auto check_point = [&](const json& point, const std::string& field) {
            if (!point.is_array() || static_cast<int>(point.size()) != dim ||
                !std::all_of(point.begin(), point.end(), [](const json& x) { return x.is_number(); })) {
                AddError(field, "Must be an array of " + std::to_string(dim) + " coordinates");
            }
        };
        std::set<std::string> names;
        for (size_t i = 0; i < probes.size(); ++i) {
            const auto& probe = probes[i];
            const std::string prefix = "output.probes[" + std::to_string(i) + "]";
            if (!probe.is_object()) {
                AddError(prefix, "Must be an object");
                continue;
            }
            for (const auto& item : probe.items()) {
                if (item.key() != "name" && item.key() != "points" && item.key() != "line" &&
                    item.key() != "entity_group") {
                    AddError(prefix + "." + item.key(), "Unknown probe setting");
                }
            }
            const std::string name = probe.contains("name") && probe["name"].is_string()
                ? probe["name"].get<std::string>() : std::string{};
            if (name.empty() || !std::all_of(name.begin(), name.end(), [](char c) {
                    return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-'; })) {
                AddError(prefix + ".name", "Required: a nonempty name of letters, digits, '_' and '-'");
            } else if (!names.insert(name).second) {
                AddError(prefix + ".name", "Duplicate probe name '" + name + "'");
            }
            if (probe.contains("points") == probe.contains("line")) {
                AddError(prefix, "Give exactly one of 'points' or 'line'");
            } else if (probe.contains("points")) {
                const auto& points = probe["points"];
                if (!points.is_array() || points.empty()) {
                    AddError(prefix + ".points", "Must be a nonempty array of points");
                } else {
                    for (size_t k = 0; k < points.size(); ++k) {
                        check_point(points[k], prefix + ".points[" + std::to_string(k) + "]");
                    }
                }
            } else {
                const auto& line = probe["line"];
                if (!line.is_object()) {
                    AddError(prefix + ".line", "Must be an object with 'from', 'to' and 'count'");
                } else {
                    for (const auto& item : line.items()) {
                        if (item.key() != "from" && item.key() != "to" && item.key() != "count") {
                            AddError(prefix + ".line." + item.key(), "Unknown line setting");
                        }
                    }
                    check_point(line.value("from", json()), prefix + ".line.from");
                    check_point(line.value("to", json()), prefix + ".line.to");
                    if (!line.contains("count") || !line["count"].is_number_integer() ||
                        line["count"].get<int>() < 2) {
                        AddError(prefix + ".line.count", "Must be an integer of at least 2");
                    }
                }
            }
            if (probe.contains("entity_group")) {
                if (!probe["entity_group"].is_string()) {
                    AddError(prefix + ".entity_group", "Must be a string");
                } else if (!domain_group_names_.count(probe["entity_group"].get<std::string>())) {
                    AddError(prefix + ".entity_group",
                        "Must name a domain entity group (of the mesh's dimension)");
                }
            }
        }
    }

    void ValidateEntityGroups(const json& config, const mfem::Mesh* mesh = nullptr) {
        entity_group_names_.clear();
        boundary_group_names_.clear();
        domain_group_names_.clear();

        if (!config.contains("entity_groups")) {
            return;
        }

        const auto& groups = config["entity_groups"];
        if (!groups.is_array()) {
            AddError("entity_groups", "Entity groups must be an array");
            return;
        }

        const int max_dom = mesh ? mesh->attributes.Max() : 0;
        const int max_bdr = mesh ? mesh->bdr_attributes.Max() : 0;

        // The role of a group (boundary vs. domain) is derived by comparing its
        // entity dimension to the mesh dimension, so a reference dimension is
        // needed before any group can be classified. With a mesh loaded that is
        // simply its dimension. Without one, the highest dimension any group
        // declares is used: domain groups carry the mesh's own dimension, so the
        // maximum over all groups equals it in any well-formed file.
        int mesh_dim = mesh ? mesh->Dimension() : 0;
        if (!mesh) {
            for (const auto& g : groups) {
                if (g.contains("dim") && g["dim"].is_number_integer()) {
                    mesh_dim = std::max(mesh_dim, g["dim"].get<int>());
                }
            }
        }

        for (size_t i = 0; i < groups.size(); ++i) {
            const auto& g = groups[i];
            std::string prefix = "entity_groups[" + std::to_string(i) + "]";

            std::string name = g.value("name", std::string{});
            if (name.empty()) {
                AddError(prefix + ".name", "Missing required field 'name'");
            } else if (!entity_group_names_.insert(name).second) {
                AddError(prefix + ".name", "Duplicate entity group name '" + name + "'");
            }

            // 'dim' is the topological dimension of the entities the ids name,
            // which is what selects the attribute namespace. Boundary/domain is
            // derived from it, not prescribed (matches InputParser).
            bool is_boundary = false;
            if (g.contains("kind")) {
                AddError(prefix + ".kind", "Unsupported field 'kind'; use 'dim' with the "
                        "entity dimension (1 = curve, 2 = surface, 3 = volume)");
            } else if (!g.contains("dim")) {
                AddError(prefix + ".dim", "Missing required field 'dim'");
            } else if (!g["dim"].is_number_integer()) {
                AddError(prefix + ".dim", "Must be an integer entity dimension "
                        "(1 = curve, 2 = surface, 3 = volume)");
            } else {
                const int dim = g["dim"];
                if (dim < 1 || dim > 3) {
                    AddError(prefix + ".dim", "Invalid dim " + std::to_string(dim) +
                            ". Must be 1, 2 or 3");
                } else if (mesh_dim > 0 && dim != mesh_dim && dim != mesh_dim - 1) {
                    // Anything other than the mesh dimension or one below it
                    // names entities this solver cannot use: too low and it is a
                    // corner/edge of the boundary, too high and it does not exist.
                    AddError(prefix + ".dim", "Entity dimension " + std::to_string(dim) +
                            " is unusable in a " + std::to_string(mesh_dim) +
                            "D mesh. Must be " + std::to_string(mesh_dim) +
                            " (domain) or " + std::to_string(mesh_dim - 1) + " (boundary)");
                }
                is_boundary = (dim == mesh_dim - 1);
            }

            if (!name.empty()) {
                if (is_boundary) boundary_group_names_.insert(name);
                else             domain_group_names_.insert(name);
            }

            if (!g.contains("attribute_ids")) {
                AddError(prefix + ".attribute_ids", "Missing required field 'attribute_ids'");
            } else if (!g["attribute_ids"].is_array()) {
                AddError(prefix + ".attribute_ids", "Attribute ids must be an array");
            } else {
                const int max_attr = mesh ? (is_boundary ? max_bdr : max_dom) : 0;
                for (int attr : g["attribute_ids"]) {
                    if (attr <= 0) {
                        AddError(prefix + ".attribute_ids", "Attributes must be positive integers");
                    } else if (mesh && attr > max_attr) {
                        AddError(prefix + ".attribute_ids", "Attribute " + std::to_string(attr) +
                                " is out of range [1, " + std::to_string(max_attr) + "]");
                    }
                }
            }
        }
    }

    void ValidateRegions(const json& config, const mfem::Mesh* mesh = nullptr) {
        if (!config.contains("regions")) {
            return;
        }

        const auto& regions = config["regions"];
        if (!regions.is_array()) {
            AddError("regions", "Regions must be an array");
            return;
        }

        // Regions bind to materials by name, so the declared names (not an
        // array position) are what a region reference must resolve against.
        std::set<std::string> material_names;
        if (config.contains("materials") && config["materials"].is_array()) {
            for (const auto& material : config["materials"]) {
                const std::string name = material.value("name", std::string{});
                if (!name.empty()) material_names.insert(name);
            }
        }

        auto group_attributes = [&](const std::string& name) {
            std::set<int> attributes;
            if (!config.contains("entity_groups") || !config["entity_groups"].is_array()) {
                return attributes;
            }
            for (const auto& group : config["entity_groups"]) {
                if (group.value("name", std::string{}) != name) continue;
                for (int attribute : group["attribute_ids"]) attributes.insert(attribute);
                break;
            }
            return attributes;
        };

        std::set<int> massive_terminal_attributes;
        if (config.contains("terminals") && config["terminals"].is_array()) {
            for (const auto& terminal : config["terminals"]) {
                if (terminal.value("conductor_type", "massive") != "massive" ||
                    !terminal.contains("entity_group")) continue;
                const auto attributes = group_attributes(terminal["entity_group"]);
                massive_terminal_attributes.insert(attributes.begin(), attributes.end());
            }
        }
        std::set<int> constrained_attributes;

        // attribute -> entity group of the region that first claimed it. A domain
        // attribute resolves to exactly one material (MaterialForAttr returns the
        // first match), so a second claim would silently make the material depend
        // on region declaration order.
        std::map<int, std::string> attribute_owner;

        for (size_t i = 0; i < regions.size(); ++i) {
            const auto& reg = regions[i];
            std::string prefix = "regions[" + std::to_string(i) + "]";

            if (!reg.contains("entity_group")) {
                AddError(prefix + ".entity_group", "Missing required field 'entity_group'");
            } else {
                CheckEntityGroupRef(reg["entity_group"], prefix + ".entity_group", RequiredGroupKind::Domain);
            }

            if (!reg.contains("material")) {
                AddError(prefix + ".material", "Missing required field 'material'");
            } else if (reg["material"].is_string()) {
                const std::string material_name = reg["material"];
                if (material_names.count(material_name) == 0) {
                    AddError(prefix + ".material", "Unknown material '" + material_name +
                             "'. No material with that name is declared");
                }
            }

            if (reg.contains("entity_group") && reg["entity_group"].is_string()) {
                const std::string group_name = reg["entity_group"];
                for (int attribute : group_attributes(reg["entity_group"])) {
                    auto [it, inserted] =
                        attribute_owner.emplace(attribute, group_name);
                    if (!inserted) {
                        AddError(prefix + ".entity_group",
                            "Domain attribute " + std::to_string(attribute) +
                            " is already claimed by entity group '" + it->second +
                            "'. Every domain attribute must have exactly one region, "
                            "otherwise its material depends on region declaration order.");
                    }
                }
            }

            if (!reg.contains("current_constraint")) continue;
            const std::string constraint = reg["current_constraint"];
            if (constraint != "open") {
                AddError(prefix + ".current_constraint",
                    "Invalid current constraint '" + constraint + "'. Must be 'open'");
                continue;
            }
            if (PhysicsType(config) != "magnetoquasistatics") {
                AddError(prefix + ".current_constraint",
                    "Region current constraints are supported only for magnetoquasistatics");
            }

            const std::string material_name =
                reg.contains("material") && reg["material"].is_string()
                    ? reg["material"].get<std::string>() : std::string{};
            static const json no_materials = json::array();
            const json& declared_materials =
                (config.contains("materials") && config["materials"].is_array())
                    ? config["materials"] : no_materials;
            for (const auto& material : declared_materials) {
                if (material.value("name", std::string{}) != material_name) continue;
                if (!material.contains("properties") || !material["properties"].is_object()) break;
                const auto& properties = material["properties"];
                const double conductivity = properties.value("sigma", 0.0);
                if (!std::isfinite(conductivity) || conductivity <= 0.0) {
                    AddError(prefix + ".current_constraint",
                        "An open-current region requires a material with positive conductivity");
                }
                break;
            }

            const auto attributes = reg.contains("entity_group")
                ? group_attributes(reg["entity_group"]) : std::set<int>{};
            for (int attribute : attributes) {
                if (massive_terminal_attributes.count(attribute) != 0) {
                    AddError(prefix + ".current_constraint",
                        "Open-current region overlaps an explicit massive terminal");
                    break;
                }
                if (constrained_attributes.count(attribute) != 0) {
                    AddError(prefix + ".current_constraint",
                        "Open-current region overlaps another open-current region");
                    break;
                }
            }
            constrained_attributes.insert(attributes.begin(), attributes.end());
        }

        ValidateDomainCoverage(config, mesh, group_attributes);
    }

    // Every domain attribute actually present in the mesh must be claimed by some
    // region, because an unclaimed attribute silently gets the zero-material
    // default: its elements contribute no stiffness, leaving their interior DOFs
    // unconstrained and the assembled system singular. An iterative solver hides
    // this (it simply stalls on the null space and returns whatever it reached at
    // the iteration cap, producing plausible-looking but wrong fields), so without
    // this check the error only surfaces as bad results. A direct factorization
    // fails outright on the same system, which is how this was found.
    //
    // Mesh-dependent, so it is skipped in the schema-only (mesh == nullptr) pass.
    void ValidateDomainCoverage(
        const json& config, const mfem::Mesh* mesh,
        const std::function<std::set<int>(const std::string&)>& group_attributes) {
        if (!mesh) return;

        std::set<int> claimed;
        if (config.contains("regions") && config["regions"].is_array()) {
            for (const auto& reg : config["regions"]) {
                if (!reg.contains("entity_group") || !reg["entity_group"].is_string()) continue;
                const auto attributes = group_attributes(reg["entity_group"]);
                claimed.insert(attributes.begin(), attributes.end());
            }
        }

        // Count elements per attribute so the diagnostic can quantify the gap and
        // so attributes that exist only in the numbering (no elements) are ignored.
        std::map<int, int> element_counts;
        for (int e = 0; e < mesh->GetNE(); ++e) element_counts[mesh->GetAttribute(e)]++;

        for (const auto& [attribute, count] : element_counts) {
            if (claimed.count(attribute) != 0) continue;
            AddError("regions",
                "Mesh domain attribute " + std::to_string(attribute) + " (" +
                std::to_string(count) + " elements) is not covered by any region, so "
                "it has no material. Every meshed domain attribute must belong to a "
                "region; add one, or remove the attribute from the mesh.");
        }
    }

    void ValidateMaterials(const json& config, const mfem::Mesh* mesh = nullptr) {
        if (!config.contains("materials")) {
            return; // Materials are optional for some physics types
        }

        const auto& materials = config["materials"];
        if (!materials.is_array()) {
            AddError("materials", "Materials must be an array");
            return;
        }

        std::string type = PhysicsType(config);

        // Regions bind to materials by name, so a missing or duplicated name
        // makes the reference either unresolvable or silently ambiguous.
        std::set<std::string> material_names;

        for (size_t i = 0; i < materials.size(); ++i) {
            const auto& mat = materials[i];
            std::string prefix = "materials[" + std::to_string(i) + "]";

            if (!mat.contains("name") || mat["name"].get<std::string>().empty()) {
                AddError(prefix + ".name", "Missing required field 'name'");
            } else {
                const std::string name = mat["name"];
                if (!material_names.insert(name).second) {
                    AddError(prefix + ".name", "Duplicate material name '" + name + "'");
                }
            }

            if (!mat.contains("properties")) {
                AddError(prefix + ".properties", "Missing required field 'properties'");
                continue;
            }

            const auto& props = mat["properties"];

            // Type-specific property validation
            if (type == "electrostatics") {
                if (!props.contains("epsilon_r")) {
                    AddError(prefix + ".properties.epsilon_r", "Electrostatic materials require 'epsilon_r'");
                } else {
                    double eps_r = props["epsilon_r"];
                    if (eps_r <= 0) {
                        AddError(prefix + ".properties.epsilon_r", "Permittivity must be positive");
                    }
                }
            } else if (type == "magnetostatics" || type == "magnetoquasistatics") {
                if (!props.contains("mu_r")) {
                    AddError(prefix + ".properties.mu_r", "Magnetic materials require 'mu_r'");
                } else {
                    double mu_r = props["mu_r"];
                    if (mu_r <= 0) {
                        AddError(prefix + ".properties.mu_r", "Permeability must be positive");
                    }
                }

                if (type == "magnetoquasistatics" && !props.contains("sigma")) {
                    // Sigma is optional (default 0), but warn if missing
                }
            }

            if (props.contains("sigma") && props["sigma"].get<double>() < 0.0) {
                AddError(prefix + ".properties.sigma", "Conductivity cannot be negative");
            }
        }
    }

    void ValidateBoundaries(const json& config, const mfem::Mesh* mesh = nullptr) {
        if (!config.contains("boundary_conditions")) {
            return; // Boundary conditions are optional; absent means homogeneous Neumann
        }

        const auto& boundaries = config["boundary_conditions"];
        if (!boundaries.is_array()) {
            AddError("boundary_conditions", "Boundary conditions must be an array");
            return;
        }

        for (size_t i = 0; i < boundaries.size(); ++i) {
            const auto& bc = boundaries[i];
            std::string prefix = "boundary_conditions[" + std::to_string(i) + "]";

            if (!bc.contains("type")) {
                AddError(prefix + ".type", "Missing required field 'type'");
            } else if (bc["type"].is_string()) {
                std::string bc_type = bc["type"];
                if (bc_type != "dirichlet" && bc_type != "neumann" && bc_type != "robin") {
                    std::string lowered = bc_type;
                    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                    const bool is_case_slip = (lowered == "dirichlet" || lowered == "neumann" || lowered == "robin");
                    AddError(prefix + ".type", is_case_slip
                        ? "Boundary condition type '" + bc_type + "' must be lowercase; use '" + lowered + "'"
                        : "Invalid boundary condition type '" + bc_type + "'. Must be 'dirichlet', 'neumann', or 'robin'");
                }

                if (bc_type == "robin" && !bc.contains("robin_coefficient")) {
                    AddError(prefix + ".robin_coefficient",
                             "Robin boundary conditions require 'robin_coefficient'");
                }
                if (bc_type != "robin" && bc.contains("robin_coefficient")) {
                    AddError(prefix + ".robin_coefficient",
                             "'robin_coefficient' is only valid for Robin boundary conditions");
                }
            }

            if (!bc.contains("entity_group")) {
                AddError(prefix + ".entity_group", "Missing required field 'entity_group'");
            } else {
                CheckEntityGroupRef(bc["entity_group"], prefix + ".entity_group", RequiredGroupKind::Boundary);
            }

            if (!bc.contains("value")) {
                AddError(prefix + ".value", "Missing required field 'value'");
            } else if (bc["value"].is_number()) {
                const double value = bc["value"].get<double>();
                if (!std::isfinite(value)) {
                    AddError(prefix + ".value", "Boundary value must be finite");
                }
            }

            if (bc.contains("robin_coefficient") &&
                bc["robin_coefficient"].is_number()) {
                const double coefficient = bc["robin_coefficient"].get<double>();
                if (!std::isfinite(coefficient)) {
                    AddError(prefix + ".robin_coefficient",
                             "Robin coefficient must be finite");
                }
            }
        }
    }

    // "direction" is how a 3D conductor states where its current flows, so it is
    // required on every 3D magnetic current terminal and meaningless anywhere
    // else (a 2D model's current direction is fixed by the geometry).
    void ValidateCurrentDirection(const json& config, const json& t, const std::string& prefix,
                               const std::string& physics, const std::string& quantity) {
        std::string geometry = "planar";
        if (config.contains("simulation") && config["simulation"].is_object() &&
            config["simulation"].contains("geometry_type") &&
            config["simulation"]["geometry_type"].is_string()) {
            geometry = config["simulation"]["geometry_type"].get<std::string>();
        }
        const bool magnetic = physics == "magnetostatics" || physics == "magnetoquasistatics";
        const bool wants_direction = geometry == "3d" && magnetic && quantity == "current";
        const std::string field = prefix + ".direction";

        if (!t.contains("direction")) {
            if (wants_direction) {
                AddError(field, "3D magnetic current terminals require 'direction', e.g. "
                    "{\"type\": \"azimuthal\", \"origin\": [0, 0, 0], \"axis\": [0, 0, 1]}");
            }
            return;
        }
        if (!wants_direction) {
            AddError(field, "'direction' applies only to current terminals of a 3D "
                "magnetic model; 2D current directions are fixed by the geometry");
            return;
        }
        const auto& d = t["direction"];
        if (!d.is_object()) return;  // reported by the type check
        const std::string kind = d.contains("type") && d["type"].is_string()
            ? d["type"].get<std::string>() : std::string();
        const std::map<std::string, std::set<std::string>> keys = {
            {"azimuthal",  {"type", "origin", "axis"}},
            {"electrodes", {"type", "input", "output"}},
            {"cut",        {"type", "cut", "normal"}}};
        const auto allowed = keys.find(kind);
        if (allowed == keys.end()) {
            AddError(field + ".type", "Must be 'azimuthal', 'electrodes' or 'cut'");
            return;
        }
        for (const auto& item : d.items()) {
            if (!allowed->second.count(item.key())) {
                AddError(field + "." + item.key(), "Unknown setting for a '" + kind + "' direction");
            }
        }
        auto vector3 = [&](const char* key, bool required) -> bool {
            const std::string where = field + "." + key;
            if (!d.contains(key)) {
                if (required) AddError(where, "Missing required field '" + std::string(key) + "'");
                return false;
            }
            const auto& v = d[key];
            if (!v.is_array() || v.size() != 3) {
                AddError(where, "Must be an array of three numbers");
                return false;
            }
            for (const auto& c : v) {
                if (!c.is_number() || !std::isfinite(c.get<double>())) {
                    AddError(where, "Must be an array of three finite numbers");
                    return false;
                }
            }
            return true;
        };
        auto nonzero = [&](const char* key) {
            if (!vector3(key, true)) return;
            const auto& a = d[key];
            const double n2 = a[0].get<double>() * a[0].get<double>() +
                              a[1].get<double>() * a[1].get<double>() +
                              a[2].get<double>() * a[2].get<double>();
            if (!(n2 > 0.0)) AddError(field + "." + key, "Must be a nonzero vector");
        };
        auto boundary_group = [&](const char* key) {
            const std::string where = field + "." + key;
            if (!d.contains(key)) {
                AddError(where, "Missing required field '" + std::string(key) + "'");
            } else {
                CheckEntityGroupRef(d[key], where, RequiredGroupKind::Boundary);
            }
        };
        if (kind == "azimuthal") {
            vector3("origin", false);
            nonzero("axis");
        } else if (kind == "electrodes") {
            boundary_group("input");
            boundary_group("output");
        } else {
            boundary_group("cut");
            nonzero("normal");
        }
    }

    void ValidateTerminals(const json& config, const mfem::Mesh* mesh = nullptr) {
        std::string type = PhysicsType(config);

        // Magnetic simulations need at least one current terminal to excite the field.
        if ((type == "magnetostatics" || type == "magnetoquasistatics") && !config.contains("terminals")) {
            AddError("terminals", "Magnetic simulations require at least one terminal");
            return;
        }

        if (!config.contains("terminals")) {
            return;
        }

        const auto& terminals = config["terminals"];
        if (!terminals.is_array()) {
            AddError("terminals", "Terminals must be an array");
            return;
        }

        std::set<std::string> terminal_names;
        for (size_t i = 0; i < terminals.size(); ++i) {
            const auto& t = terminals[i];
            std::string prefix = "terminals[" + std::to_string(i) + "]";

            if (!t.contains("name") || t["name"].get<std::string>().empty()) {
                AddError(prefix + ".name", "Missing required field 'name'");
            } else {
                const std::string name = t["name"];
                if (!terminal_names.insert(name).second) {
                    AddError(prefix + ".name", "Duplicate terminal name '" + name + "'");
                }
            }

            // 'quantity' is required rather than defaulting to voltage: a
            // silent default turns a renamed or misspelled key into a confusing
            // downstream "must be a boundary group" error instead of naming the
            // real problem. A superseded spelling already got the actionable
            // rename message, so the generic error is suppressed there.
            const bool has_legacy_spelling =
                t.contains("excitation_type") || t.contains("excitation");
            if (!t.contains("quantity") && !has_legacy_spelling) {
                AddError(prefix + ".quantity", "Missing required field 'quantity'");
            }

            std::string excitation = t.value("quantity", "voltage");
            if (excitation != "voltage" && excitation != "current") {
                AddError(prefix + ".quantity", "Invalid quantity '" + excitation + "'. Must be 'voltage' or 'current'");
            }
            else if (type == "electrostatics" && excitation != "voltage") {
                AddError(prefix + ".quantity",
                    "Electrostatic terminals must use quantity 'voltage'");
            }
            else if ((type == "magnetostatics" || type == "magnetoquasistatics") &&
                     excitation != "current") {
                AddError(prefix + ".quantity",
                    "Magnetic terminals must use quantity 'current'; "
                    "massive MQS terminal voltage is a solved output");
            }

            const std::string conductor = t.value("conductor_type", "massive");
            if (conductor != "massive" && conductor != "stranded") {
                AddError(prefix + ".conductor_type", "Invalid conductor_type '" + conductor + "'. Must be 'massive' or 'stranded'");
            }

            ValidateCurrentDirection(config, t, prefix, type, excitation);

            // Voltage terminals bind to boundary groups; current terminals to domain groups.
            const bool is_current = (excitation == "current");
            if (!t.contains("entity_group")) {
                AddError(prefix + ".entity_group", "Missing required field 'entity_group'");
            } else {
                CheckEntityGroupRef(t["entity_group"], prefix + ".entity_group",
                                    is_current ? RequiredGroupKind::Domain
                                               : RequiredGroupKind::Boundary);
            }
        }
    }

    void ValidateScenarios(const json& config, const mfem::Mesh* mesh = nullptr) {
        if (!config.contains("scenarios")) {
            if (PhysicsType(config) == "magnetoquasistatics") {
                AddError("scenarios", "Magnetoquasistatic simulations require at least one frequency scenario");
            }
            return;
        }

        const auto& scenarios = config["scenarios"];
        if (!scenarios.is_array()) {
            AddError("scenarios", "Scenarios must be an array");
            return;
        }
        const bool is_mqs = PhysicsType(config) == "magnetoquasistatics";
        if (is_mqs && scenarios.empty()) {
            AddError("scenarios", "Magnetoquasistatic simulations require at least one frequency scenario");
            return;
        }

        // Build the set of declared terminal names so excitations can be
        // cross-checked (catches typo'd terminal references).
        std::set<std::string> terminal_names;
        if (config.contains("terminals") && config["terminals"].is_array()) {
            for (const auto& t : config["terminals"]) {
                std::string name = t.value("name", "");
                if (!name.empty()) {
                    terminal_names.insert(name);
                }
            }
        }

        std::set<std::string> scenario_names;
        for (size_t i = 0; i < scenarios.size(); ++i) {
            const auto& sc = scenarios[i];
            std::string prefix = "scenarios[" + std::to_string(i) + "]";

            if (!sc.contains("name") || sc["name"].get<std::string>().empty()) {
                AddError(prefix + ".name", "Missing required field 'name'");
            } else {
                const std::string name = sc["name"];
                if (!scenario_names.insert(name).second) {
                    AddError(prefix + ".name", "Duplicate scenario name '" + name + "'");
                }
            }

            if (is_mqs) {
                if (!sc.contains("frequency")) {
                    AddError(prefix + ".frequency",
                        "Missing required MQS scenario field 'frequency'");
                }
                else if (sc["frequency"].is_number()) {
                    const double frequency = sc["frequency"].get<double>();
                    if (!std::isfinite(frequency) || frequency <= 0.0) {
                        AddError(prefix + ".frequency", "Frequency must be finite and positive");
                    }
                }
				else if (sc["frequency"].is_array()) {
					// Frequency sweep as an array of numbers
					const auto& freq_array = sc["frequency"];
					if (freq_array.empty()) {
						AddError(prefix + ".frequency", "Frequency array must not be empty");
					}
					else {
						for (size_t j = 0; j < freq_array.size(); ++j) {
							const auto& freq = freq_array[j];
							if (!freq.is_number()) {
								AddError(prefix + ".frequency[" + std::to_string(j) + "]", "Frequency must be a number");
							}
							else {
								const double frequency = freq.get<double>();
								if (!std::isfinite(frequency) || frequency <= 0.0) {
									AddError(prefix + ".frequency[" + std::to_string(j) + "]", "Frequency must be finite and positive");
								}
							}
						}
					}
				}
                else if (sc["frequency"].is_object()) {
                    const auto& sweep = sc["frequency"];
                    const bool complete = sweep.contains("scale") && sweep.contains("start")
                        && sweep.contains("stop") && sweep.contains("points");
                    if (!complete) {
                        AddError(prefix + ".frequency",
                            "Sweep requires 'scale', 'start', 'stop', and 'points'");
                    }
                    else {
                        const std::string scale = sweep["scale"];
                        const double start = sweep["start"];
                        const double stop = sweep["stop"];
                        const int points = sweep["points"];
                        if (scale != "linear" && scale != "log") {
                            AddError(prefix + ".frequency.scale",
                                "Must be 'linear' or 'log'");
                        }
                        if (!std::isfinite(start) || start <= 0.0) {
                            AddError(prefix + ".frequency.start", "Must be finite and positive");
                        }
                        if (!std::isfinite(stop) || stop <= 0.0) {
                            AddError(prefix + ".frequency.stop", "Must be finite and positive");
                        }
                        if (std::isfinite(start) && std::isfinite(stop) && stop < start) {
                            AddError(prefix + ".frequency.stop", "Must be greater than or equal to start");
                        }
                        if (points < 1) {
                            AddError(prefix + ".frequency.points", "Must be at least 1");
                        }
                    }
                }
            }

            if (!sc.contains("excitations")) {
                continue;
            }
            if (!sc["excitations"].is_array()) {
                AddError(prefix + ".excitations", "Excitations must be an array");
                continue;
            }

            for (size_t j = 0; j < sc["excitations"].size(); ++j) {
                const auto& d = sc["excitations"][j];
                std::string dprefix = prefix + ".excitations[" + std::to_string(j) + "]";

                if (!d.contains("terminal")) {
                    AddError(dprefix + ".terminal", "Missing required field 'terminal'");
                    continue;
                }

                std::string tname = d["terminal"];
                if (terminal_names.find(tname) == terminal_names.end()) {
                    AddError(dprefix + ".terminal", "Unknown terminal '" + tname +
                            "'. No terminal with that name is declared");
                }
            }
        }
    }

    // Cross-checks between geometry_type and the rest of the run: the mesh
    // dimension it requires, and the settings that have no 3D meaning. Runs
    // after ValidateSimulation(), so an invalid
    // geometry_type string has already been reported and is skipped here.
    //
    // The dimension check matters because nothing downstream would catch the
    // mismatch: the planar default on a 3D mesh assembles and solves an
    // ordinary 3D Laplacian, then labels an absolute capacitance in F/m.
    void ValidateGeometryCompatibility(const json& config, const mfem::Mesh* mesh) {
        if (!config.contains("simulation") || !config["simulation"].is_object()) return;
        const auto& sim = config["simulation"];

        std::string geometry = "planar";  // InputParser's default
        if (sim.contains("geometry_type")) {
            if (!sim["geometry_type"].is_string()) return;
            geometry = sim["geometry_type"].get<std::string>();
        }
        if (geometry != "axisymmetric" && geometry != "planar" && geometry != "3d") return;
        const bool three_d = (geometry == "3d");

        if (mesh) {
            const int required = three_d ? 3 : 2;
            if (mesh->Dimension() != required) {
                AddError("simulation.geometry_type",
                    "geometry_type '" + geometry + "' requires a " +
                    std::to_string(required) + "D mesh, but the mesh is " +
                    std::to_string(mesh->Dimension()) + "D" +
                    (mesh->Dimension() == 3
                        ? ". Set geometry_type to '3d' for a 3D mesh"
                        : (three_d ? ". Use 'planar' or 'axisymmetric' for a 2D mesh" : "")));
            }
        }

        if (!three_d) return;

        // A 2D "open" region forces a conductor's net current to zero because
        // the planar model cannot represent where its ends are. In 3D the
        // ends are part of the mesh, so the constraint has no meaning.
        if (config.contains("regions") && config["regions"].is_array()) {
            const auto& regions = config["regions"];
            for (size_t i = 0; i < regions.size(); ++i) {
                if (regions[i].is_object() && regions[i].contains("current_constraint")) {
                    AddError("regions[" + std::to_string(i) + "].current_constraint",
                        "Current constraints are not available for geometry_type "
                        "'3d': a 3D conductor with no terminal carries only "
                        "induced current, so model its actual extent instead");
                }
            }
        }
    }

public:
    /**
     * @brief Validate a configuration against a mesh
     * @param config The JSON configuration
     * @param mesh Optional mesh pointer for attribute range checking
     * @return true if configuration is valid
     */
    bool Validate(const json& config, const mfem::Mesh* mesh = nullptr) {
        errors.clear();

        ValidateDocumentTypes(config);
        if (!errors.empty()) {
            return false;
        }

        ValidateSimulation(config);
        ValidateGeometryCompatibility(config, mesh);
        ValidateEntityGroups(config, mesh);
        ValidateProbes(config, mesh);
        ValidateMaterials(config, mesh);
        ValidateRegions(config, mesh);
        ValidateBoundaries(config, mesh);
        ValidateTerminals(config, mesh);
        ValidateScenarios(config, mesh);

        return errors.empty();
    }

    /**
     * @brief Get all validation errors
     */
    [[nodiscard]] const std::vector<ValidationError>& GetErrors() const {
        return errors;
    }

    /**
     * @brief Get formatted error message
     */
    [[nodiscard]] std::string GetErrorMessage() const {
        if (errors.empty()) {
            return "No errors";
        }

        std::string msg = "Configuration validation failed:\n";
        for (const auto& err : errors) {
            msg += "  - " + err.field + ": " + err.message + "\n";
        }
        return msg;
    }

    /**
     * @brief Validate and throw if invalid
     * @throws std::runtime_error with detailed error message
     */
    void ValidateOrThrow(const json& config, const mfem::Mesh* mesh = nullptr) {
        if (!Validate(config, mesh)) {
            throw std::runtime_error(GetErrorMessage());
        }
    }
};
