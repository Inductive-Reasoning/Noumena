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

    [[nodiscard]] static bool HasType(const json& value, ExpectedType type);

    [[nodiscard]] static const char* TypeName(ExpectedType type);

    void CheckFieldType(const json& object, const char* key,
                        const std::string& field, ExpectedType type);

    void CheckObjectArrayTypes(
        const json& config, const char* key,
        const std::function<void(const json&, const std::string&)>& check_item);

    void ValidateOutput(const json& config);

    void ValidateDocumentTypes(const json& config);

    // "simulation.physics_type" as a string, or "" when absent.
    [[nodiscard]] std::string PhysicsType(const json& config) const;

    // Which kind of entity group a reference is required to name.
    enum class RequiredGroupKind { Any, Boundary, Domain };

    // Verify that an "entity_group" reference points at a declared group and,
    // when required is not Any, that the group is of the matching kind. Mirrors
    // how the solvers resolve EntityGroupName -> EntityGroup via
    // config.EntityGroups.at(name).
    void CheckEntityGroupRef(const json& node, const std::string& field, RequiredGroupKind required);

    void ValidateSimulation(const json& config);

    // output.probes: named point sets, each "points" (coordinate arrays) or a
    // "line" {"from", "to", "count" >= 2}, optionally restricted to a domain
    // entity group. Coordinates have the model's space dimension. Names become
    // file names, so they are restricted to [A-Za-z0-9_-].
    void ValidateProbes(const json& config, const mfem::Mesh* mesh);

    void ValidateEntityGroups(const json& config, const mfem::Mesh* mesh = nullptr);

    void ValidateRegions(const json& config, const mfem::Mesh* mesh = nullptr);

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
        const std::function<std::set<int>(const std::string&)>& group_attributes);

    void ValidateMaterials(const json& config, const mfem::Mesh* mesh = nullptr);

    void ValidateBoundaries(const json& config, const mfem::Mesh* mesh = nullptr);

    // "direction" is how a 3D conductor states where its current flows, so it is
    // required on every 3D magnetic current terminal and meaningless anywhere
    // else (a 2D model's current direction is fixed by the geometry).
    void ValidateCurrentDirection(const json& config, const json& t, const std::string& prefix,
                               const std::string& physics, const std::string& quantity);

    void ValidateTerminals(const json& config, const mfem::Mesh* mesh = nullptr);

    void ValidateScenarios(const json& config, const mfem::Mesh* mesh = nullptr);

    // Cross-checks between geometry_type and the rest of the run: the mesh
    // dimension it requires, and the settings that have no 3D meaning. Runs
    // after ValidateSimulation(), so an invalid
    // geometry_type string has already been reported and is skipped here.
    //
    // The dimension check matters because nothing downstream would catch the
    // mismatch: the planar default on a 3D mesh assembles and solves an
    // ordinary 3D Laplacian, then labels an absolute capacitance in F/m.
    void ValidateGeometryCompatibility(const json& config, const mfem::Mesh* mesh);

public:
    /**
     * @brief Validate a configuration against a mesh
     * @param config The JSON configuration
     * @param mesh Optional mesh pointer for attribute range checking
     * @return true if configuration is valid
     */
    bool Validate(const json& config, const mfem::Mesh* mesh = nullptr);

    /**
     * @brief Get all validation errors
     */
    [[nodiscard]] const std::vector<ValidationError>& GetErrors() const {
        return errors;
    }

    /**
     * @brief Get formatted error message
     */
    [[nodiscard]] std::string GetErrorMessage() const;

    /**
     * @brief Validate and throw if invalid
     * @throws std::runtime_error with detailed error message
     */
    void ValidateOrThrow(const json& config, const mfem::Mesh* mesh = nullptr);
};
