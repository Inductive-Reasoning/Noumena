// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "mfem.hpp"

/**
 * @brief Resolves entity groups against a loaded mesh.
 *
 * An entity group is a named set of mesh attributes. A mesh file may define
 * some itself -- Gmsh $PhysicalNames, Netgen .vol material and boundary names,
 * MFEM v1.3 attribute_sets -- which the loader keeps as the mesh's named
 * attribute sets: domain groups in attribute_sets, boundary groups in
 * bdr_attribute_sets. The configuration's "entity_groups" section defines
 * more, each from attribute ids, from other entity groups, or both:
 *
 *   {"name": "Coil",  "dim": 2, "attribute_ids": [7, 8]}
 *   {"name": "Walls", "dim": 2, "entity_groups": ["outer", "symmetry"]}
 *
 * The "entity_group" of a region, terminal, boundary condition or probe may
 * name a group from either source. A domain reference (a region, a current
 * terminal, a probe) looks among domain groups, a boundary reference (a
 * boundary condition, a voltage terminal) among boundary groups.
 *
 * ResolveEntityGroups() rewrites the configuration so that every group a
 * reference names is a config group with explicit "attribute_ids":
 *   - each config group's "entity_groups" is replaced by the union of their
 *     ids (merged with its own "attribute_ids");
 *   - each reference to a mesh-defined group adds that group to
 *     "entity_groups" with the mesh's ids.
 * Downstream code then sees only attribute ids. It is an error for a name to
 * be defined nowhere, for a config group to reuse a mesh group's name with
 * different ids, and for groups to include each other in a cycle.
 */
namespace entity_group_resolution {

using json = nlohmann::json;

class Resolver {
public:
	Resolver(json& config, const mfem::Mesh& mesh) : config_(config), mesh_(mesh) {
		if (config_.contains("entity_groups") && config_["entity_groups"].is_array()) {
			for (auto& group : config_["entity_groups"]) {
				if (group.is_object() && group.contains("name") && group["name"].is_string()) {
					config_groups_.emplace(group["name"].get<std::string>(), &group);
				}
			}
		}
	}

	void Run() {
		for (const auto& [name, group] : config_groups_) { Ids(name); }
		for (const auto& [name, group] : config_groups_) { CheckNoShadowing(name, *group); }

		std::vector<json> added;
		std::set<std::string> added_names;
		for (const auto& [name, boundary] : References()) {
			if (config_groups_.count(name) || added_names.count(name)) { continue; }
			const mfem::AttributeSets& sets = Sets(boundary);
			if (!sets.AttributeSetExists(name)) {
				throw std::runtime_error("Unknown entity group '" + name + "': not defined in "
					"'entity_groups' and not named by the mesh. " + Known(boundary));
			}
			added.push_back({{"name", name}, {"dim", Dim(boundary)},
							 {"attribute_ids", Sorted(sets.GetAttributeSet(name))}});
			added_names.insert(name);
		}
		if (!added.empty() && !config_.contains("entity_groups")) {
			config_["entity_groups"] = json::array();
		}
		for (auto& group : added) { config_["entity_groups"].push_back(std::move(group)); }
	}

private:
	json& config_;
	const mfem::Mesh& mesh_;
	std::map<std::string, json*> config_groups_;
	std::map<std::string, std::vector<int>> resolved_;
	std::set<std::string> in_progress_;

	int Dim(bool boundary) const { return mesh_.Dimension() - (boundary ? 1 : 0); }

	const mfem::AttributeSets& Sets(bool boundary) const {
		return boundary ? mesh_.bdr_attribute_sets : mesh_.attribute_sets;
	}

	static json Sorted(const mfem::Array<int>& ids) {
		std::set<int> unique(ids.begin(), ids.end());
		return json(std::vector<int>(unique.begin(), unique.end()));
	}

	std::string Known(bool boundary) const {
		std::set<std::string> names;
		for (const auto& [name, group] : config_groups_) {
			if (group->value("dim", 0) == Dim(boundary)) { names.insert(name); }
		}
		for (const auto& name : Sets(boundary).GetAttributeSetNames()) { names.insert(name); }
		std::string list;
		for (const auto& name : names) { list += (list.empty() ? "" : ", ") + ("'" + name + "'"); }
		return std::string("Known ") + (boundary ? "boundary" : "domain") + " groups: " +
			(list.empty() ? "none" : list) + ".";
	}

	// The ids of config group `name`, resolving its "entity_groups" (once).
	const std::vector<int>& Ids(const std::string& name) {
		if (const auto done = resolved_.find(name); done != resolved_.end()) { return done->second; }
		json& group = *config_groups_.at(name);
		const std::string where = "entity group '" + name + "'";
		if (!in_progress_.insert(name).second) {
			throw std::runtime_error(where + " includes itself through 'entity_groups'");
		}
		const int dim = group.value("dim", 0);
		const bool boundary = dim == mesh_.Dimension() - 1;

		std::set<int> ids;
		if (group.contains("attribute_ids") && group["attribute_ids"].is_array()) {
			for (const auto& id : group["attribute_ids"]) {
				if (id.is_number_integer()) { ids.insert(id.get<int>()); }
			}
		}
		if (group.contains("entity_groups")) {
			if (!group["entity_groups"].is_array()) {
				throw std::runtime_error(where + ": 'entity_groups' must be an array of names");
			}
			for (const auto& member : group["entity_groups"]) {
				if (!member.is_string()) {
					throw std::runtime_error(where + ": 'entity_groups' must be an array of names");
				}
				const std::string member_name = member.get<std::string>();
				if (config_groups_.count(member_name)) {
					if (config_groups_.at(member_name)->value("dim", 0) != dim) {
						throw std::runtime_error(where + " (dim " + std::to_string(dim) +
							") includes '" + member_name + "' of a different dim");
					}
					for (int id : Ids(member_name)) { ids.insert(id); }
				}
				else if (Sets(boundary).AttributeSetExists(member_name)) {
					for (int id : Sets(boundary).GetAttributeSet(member_name)) { ids.insert(id); }
				}
				else {
					throw std::runtime_error(where + " includes unknown entity group '" +
						member_name + "'. " + Known(boundary));
				}
			}
			group.erase("entity_groups");
			group["attribute_ids"] = std::vector<int>(ids.begin(), ids.end());
		}
		in_progress_.erase(name);
		return resolved_[name] = std::vector<int>(ids.begin(), ids.end());
	}

	// A config group may share a mesh group's name only if it means the same
	// attributes; otherwise one would silently hide the other.
	void CheckNoShadowing(const std::string& name, const json& group) {
		const int dim = group.value("dim", 0);
		if (dim != mesh_.Dimension() && dim != mesh_.Dimension() - 1) { return; }
		const mfem::AttributeSets& sets = Sets(dim == mesh_.Dimension() - 1);
		if (!sets.AttributeSetExists(name)) { return; }
		const json mesh_ids = Sorted(sets.GetAttributeSet(name));
		if (mesh_ids != json(resolved_.at(name))) {
			throw std::runtime_error("entity group '" + name + "' has attribute ids " +
				json(resolved_.at(name)).dump() + ", but the mesh names " + mesh_ids.dump() +
				" '" + name + "'; rename the configuration's group");
		}
	}

	// Every group reference, with whether it needs a boundary group.
	std::vector<std::pair<std::string, bool>> References() const {
		std::vector<std::pair<std::string, bool>> refs;
		auto collect = [&](const char* section, auto needs_boundary) {
			if (!config_.contains(section) || !config_[section].is_array()) { return; }
			for (const auto& item : config_[section]) {
				if (item.is_object() && item.contains("entity_group") && item["entity_group"].is_string()) {
					refs.emplace_back(item["entity_group"].get<std::string>(), needs_boundary(item));
				}
			}
		};
		collect("regions", [](const json&) { return false; });
		collect("boundary_conditions", [](const json&) { return true; });
		collect("terminals", [](const json& t) { return t.value("quantity", "voltage") != "current"; });
		if (config_.contains("output") && config_["output"].is_object() &&
			config_["output"].contains("probes") && config_["output"]["probes"].is_array()) {
			for (const auto& probe : config_["output"]["probes"]) {
				if (probe.is_object() && probe.contains("entity_group") && probe["entity_group"].is_string()) {
					refs.emplace_back(probe["entity_group"].get<std::string>(), false);
				}
			}
		}
		return refs;
	}
};

} // namespace entity_group_resolution

inline void ResolveEntityGroups(nlohmann::json& config, const mfem::Mesh& mesh) {
	entity_group_resolution::Resolver(config, mesh).Run();
}
