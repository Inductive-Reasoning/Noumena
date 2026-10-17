// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include "mfem.hpp"

/**
 * @brief Resolves entity groups against a loaded mesh.
 *
 * An entity group is a named set of mesh attributes, identified by its name
 * and dimension: a domain and a boundary group may share a name. A mesh file
 * may define groups itself -- Gmsh $PhysicalNames, Netgen .vol material and
 * boundary names, MFEM v1.3 attribute_sets -- which the loader keeps as the
 * mesh's named attribute sets: domain groups in attribute_sets, boundary
 * groups in bdr_attribute_sets. The configuration's "entity_groups" section
 * defines more, each from attribute ids, from other groups of its dimension,
 * or both:
 *
 *   {"name": "Coil",  "dim": 2, "attribute_ids": [7, 8]}
 *   {"name": "Walls", "dim": 1, "entity_groups": ["outer", "symmetry"]}
 *
 * The "entity_group" of a region, terminal, boundary condition or probe, and
 * the electrodes or cut of a 3D current terminal's direction, may name a group
 * from either source. A domain reference (a region, a current terminal, a
 * probe) names a group of the mesh's dimension, a boundary reference (a
 * boundary condition, a voltage terminal, a direction) one of a dimension
 * less.
 *
 * ResolveEntityGroups() rewrites the configuration so that every group a
 * reference names is a config group with explicit "attribute_ids":
 *   - each config group's "entity_groups" is replaced by the union of their
 *     ids (merged with its own "attribute_ids");
 *   - each reference to a mesh-defined group adds that group, with its dim,
 *     to "entity_groups" with the mesh's ids.
 * Downstream code then sees only attribute ids. It is an error for a name to
 * be defined nowhere in the role a reference needs, for a config group to
 * reuse a mesh group's name and dim with different ids, and for groups to
 * include each other in a cycle.
 */
namespace entity_group_resolution {

using json = nlohmann::json;

class Resolver {
public:
	Resolver(json& config, const mfem::Mesh& mesh) : config_(config), mesh_(mesh) {
		if (config_.contains("entity_groups") && config_["entity_groups"].is_array()) {
			for (auto& group : config_["entity_groups"]) {
				if (group.is_object() && group.contains("name") && group["name"].is_string()) {
					config_groups_.emplace(Key{group["name"].get<std::string>(), group.value("dim", 0)}, &group);
				}
			}
		}
	}

	void Run() {
		for (const auto& [key, group] : config_groups_) { Ids(key); }
		for (const auto& [key, group] : config_groups_) { CheckNoShadowing(key); }

		std::vector<json> added;
		std::set<Key> materialized;
		for (const auto& [name, boundary] : References()) {
			const Key key{name, Dim(boundary)};
			if (config_groups_.count(key) || materialized.count(key)) { continue; }
			const mfem::AttributeSets& sets = Sets(boundary);
			if (!sets.AttributeSetExists(name)) {
				throw std::runtime_error("Unknown " + Role(boundary) + " entity group '" + name +
					"': not defined in 'entity_groups' and not named by the mesh. " + Known(boundary));
			}
			added.push_back({{"name", name}, {"dim", key.second},
							 {"attribute_ids", Sorted(sets.GetAttributeSet(name))}});
			materialized.insert(key);
		}
		if (!added.empty() && !config_.contains("entity_groups")) {
			config_["entity_groups"] = json::array();
		}
		for (auto& group : added) { config_["entity_groups"].push_back(std::move(group)); }
	}

private:
	using Key = std::pair<std::string, int>;  // (name, dim)

	json& config_;
	const mfem::Mesh& mesh_;
	std::map<Key, json*> config_groups_;
	std::map<Key, std::vector<int>> resolved_;
	std::set<Key> in_progress_;

	int Dim(bool boundary) const { return mesh_.Dimension() - (boundary ? 1 : 0); }
	static std::string Role(bool boundary) { return boundary ? "boundary" : "domain"; }

	const mfem::AttributeSets& Sets(bool boundary) const {
		return boundary ? mesh_.bdr_attribute_sets : mesh_.attribute_sets;
	}

	static json Sorted(const mfem::Array<int>& ids) {
		std::set<int> unique(ids.begin(), ids.end());
		return json(std::vector<int>(unique.begin(), unique.end()));
	}

	std::string Known(bool boundary) const {
		std::set<std::string> names;
		for (const auto& [key, group] : config_groups_) {
			if (key.second == Dim(boundary)) { names.insert(key.first); }
		}
		for (const auto& name : Sets(boundary).GetAttributeSetNames()) { names.insert(name); }
		std::string list;
		for (const auto& name : names) { list += (list.empty() ? "" : ", ") + ("'" + name + "'"); }
		return "Known " + Role(boundary) + " groups: " + (list.empty() ? "none" : list) + ".";
	}

	// The ids of a config group, resolving its "entity_groups" (once). A
	// member is a group of the same dim: the configuration's, else the mesh's.
	const std::vector<int>& Ids(const Key& key) {
		if (const auto done = resolved_.find(key); done != resolved_.end()) { return done->second; }
		json& group = *config_groups_.at(key);
		const std::string where = "entity group '" + key.first + "' (dim " + std::to_string(key.second) + ")";
		if (!in_progress_.insert(key).second) {
			throw std::runtime_error(where + " includes itself through 'entity_groups'");
		}
		const bool boundary = key.second == mesh_.Dimension() - 1;

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
				const Key member_key{member.get<std::string>(), key.second};
				if (config_groups_.count(member_key)) {
					for (int id : Ids(member_key)) { ids.insert(id); }
				}
				else if (Sets(boundary).AttributeSetExists(member_key.first)) {
					for (int id : Sets(boundary).GetAttributeSet(member_key.first)) { ids.insert(id); }
				}
				else {
					throw std::runtime_error(where + " includes unknown entity group '" +
						member_key.first + "' of its dim. " + Known(boundary));
				}
			}
			group.erase("entity_groups");
			group["attribute_ids"] = std::vector<int>(ids.begin(), ids.end());
		}
		in_progress_.erase(key);
		return resolved_[key] = std::vector<int>(ids.begin(), ids.end());
	}

	// A config group may share a mesh group's name and dim only if it means
	// the same attributes; otherwise one would silently hide the other.
	void CheckNoShadowing(const Key& key) {
		const auto& [name, dim] = key;
		if (dim != mesh_.Dimension() && dim != mesh_.Dimension() - 1) { return; }
		const mfem::AttributeSets& sets = Sets(dim == mesh_.Dimension() - 1);
		if (!sets.AttributeSetExists(name)) { return; }
		const json mesh_ids = Sorted(sets.GetAttributeSet(name));
		if (mesh_ids != json(resolved_.at(key))) {
			throw std::runtime_error("entity group '" + name + "' (dim " + std::to_string(dim) +
				") has attribute ids " + json(resolved_.at(key)).dump() + ", but the mesh names " +
				mesh_ids.dump() + " '" + name + "'; rename the configuration's group");
		}
	}

	// Every group reference, with whether it needs a boundary group: regions,
	// boundary conditions, terminals and their 3D current directions, probes.
	std::vector<std::pair<std::string, bool>> References() const {
		std::vector<std::pair<std::string, bool>> refs;
		auto add = [&](const json& item, const char* key, bool boundary) {
			if (item.is_object() && item.contains(key) && item[key].is_string()) {
				refs.emplace_back(item[key].get<std::string>(), boundary);
			}
		};
		auto each = [&](const json& parent, const char* section, auto visit) {
			if (parent.is_object() && parent.contains(section) && parent[section].is_array()) {
				for (const auto& item : parent[section]) { visit(item); }
			}
		};
		each(config_, "regions", [&](const json& r) { add(r, "entity_group", false); });
		each(config_, "boundary_conditions", [&](const json& bc) { add(bc, "entity_group", true); });
		each(config_, "terminals", [&](const json& t) {
			if (!t.is_object()) { return; }
			add(t, "entity_group", t.value("quantity", "voltage") != "current");
			if (t.contains("direction")) {
				// Electrodes and cuts name boundary groups.
				for (const char* key : {"input", "output", "cut"}) { add(t["direction"], key, true); }
			}
		});
		if (config_.contains("output")) {
			each(config_["output"], "probes", [&](const json& p) { add(p, "entity_group", false); });
		}
		return refs;
	}
};

} // namespace entity_group_resolution

inline void ResolveEntityGroups(nlohmann::json& config, const mfem::Mesh& mesh) {
	entity_group_resolution::Resolver(config, mesh).Run();
}
