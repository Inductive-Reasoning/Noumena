// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>
#include "mfem.hpp"

/**
 * @brief Resolves entity groups that name a mesh attribute set.
 *
 * An entity group selects its mesh attributes either by number,
 * `"attribute_ids": [2, 3]`, or by the name the mesh file gives them,
 * `"attribute_set": "outer"`. Names come from the mesh's named attribute sets:
 * Gmsh $PhysicalNames, Netgen .vol material and boundary names and MFEM v1.3
 * attribute_sets. A domain group (dim equal to
 * the mesh dimension) looks in the domain sets, a boundary group in the
 * boundary sets.
 *
 * ResolveAttributeSets() replaces each group's "attribute_set" with the
 * "attribute_ids" it names, after which the group is indistinguishable from
 * one written with ids: everything downstream works on attribute numbers. A name the mesh
 * does not define is an error, never an empty group, so a configuration
 * cannot silently select nothing on a mesh format that carries no names.
 */
inline void ResolveAttributeSets(nlohmann::json& config, const mfem::Mesh& mesh) {
	if (!config.contains("entity_groups") || !config["entity_groups"].is_array()) { return; }
	const int mesh_dim = mesh.Dimension();
	for (auto& group : config["entity_groups"]) {
		if (!group.is_object() || !group.contains("attribute_set")) { continue; }
		const std::string group_name = group.value("name", std::string("(unnamed)"));
		const std::string where = "entity group '" + group_name + "'";
		if (!group["attribute_set"].is_string()) {
			throw std::runtime_error(where + ": 'attribute_set' must be a string");
		}
		if (group.contains("attribute_ids")) {
			throw std::runtime_error(where + ": give either 'attribute_ids' or "
				"'attribute_set', not both");
		}
		const std::string set_name = group["attribute_set"].get<std::string>();
		const bool boundary = group.value("dim", 0) == mesh_dim - 1;
		const mfem::AttributeSets& sets = boundary ? mesh.bdr_attribute_sets : mesh.attribute_sets;
		if (!sets.AttributeSetExists(set_name)) {
			std::string known;
			for (const auto& name : sets.GetAttributeSetNames()) {
				known += (known.empty() ? "" : ", ") + ("'" + name + "'");
			}
			throw std::runtime_error(where + ": the mesh has no " +
				(boundary ? "boundary" : "domain") + " attribute set named '" + set_name +
				"' (" + (known.empty() ? "it names none" : "it names " + known) + ")");
		}
		const mfem::Array<int>& ids = sets.GetAttributeSet(set_name);
		nlohmann::json attribute_ids = nlohmann::json::array();
		for (int i = 0; i < ids.Size(); ++i) { attribute_ids.push_back(ids[i]); }
		group["attribute_ids"] = std::move(attribute_ids);
		group.erase("attribute_set");
	}
}

