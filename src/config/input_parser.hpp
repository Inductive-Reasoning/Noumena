// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once
#include <nlohmann/json.hpp>
#include "../core/constants.hpp"
#include "../core/problem_config.hpp"
#include "../parallel/mpi_runtime.hpp"
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <optional>
#include <memory>
#include <filesystem> // C++17
#include <algorithm>
#include <initializer_list>
#include <string_view>
#include <utility>
#include <iterator>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;
namespace fs = std::filesystem;

class InputParser {
    std::unique_ptr<json> owned_config;
    std::string config_text;   // Source text, retained for error diagnostics

public:
    const json& config;
    std::string config_dir;
    std::string config_path;   // Source file (empty when built from in-memory json)

    InputParser(const std::string &filename);

    // Allow construction from existing json
    InputParser(const json& c) : config(c), config_dir(".") {}
    InputParser(json&&) = delete;

	const ProblemConfig GetProblemConfig() const;

private:
	const ProblemConfig BuildProblemConfig() const;

	// Build a human-friendly message from a nlohmann json exception.
	// With JSON_DIAGNOSTICS the message carries a JSON pointer like
	// "(/terminals/0/entity_group)" and with JSON_DIAGNOSTIC_POSITIONS it also
	// carries "(bytes X-Y)"; the byte range is translated into a line/column
	// against the source config file when one is available.
	[[nodiscard]] std::string DescribeJsonError(const json::exception& e) const;

	// Translate a 0-based byte offset in the config source into a 1-based
	// (line, column). Returns {0, 0} when no source text was retained (e.g. for
	// in-memory configs with no backing file). The text was read in text mode by
	// the constructor, so CRLF is already collapsed to LF and the offsets match
	// the coordinate system nlohmann used while parsing.
	[[nodiscard]] std::pair<int, int> ByteOffsetToLineCol(std::size_t byte_off) const;

    // --------------------------------------------------------
    // Lookup primitives
    // --------------------------------------------------------

    // The "simulation" object, or a shared empty object when the section is
    // absent or malformed. Lets the getters below index a single node instead
    // of repeating a contains/is_object guard chain.
    [[nodiscard]] const json& Sim() const;

    // Typed lookup with a fallback. A missing or null key yields the fallback;
    // a key that is present but of the wrong type throws json::type_error,
    // which GetProblemConfig() turns into a located error message. This is
    // deliberately stricter than json::value(), which silently absorbs some
    // mismatches and would let a typo like "order": "2" pass unnoticed.
    template <typename T>
    [[nodiscard]] static T Get(const json& obj, const char* key, T fallback) {
        const auto it = obj.find(key);
        if (it == obj.end() || it->is_null()) return fallback;
        return it->get<T>();
    }

    // String-keyed enum lookup. Unknown values fall back rather than throwing,
    // because ConfigValidator reports them with the full list of valid options.
    template <typename E>
    [[nodiscard]] static E ParseEnum(const json& obj, const char* key, E fallback,
                                     std::initializer_list<std::pair<std::string_view, E>> table) {
        const auto it = obj.find(key);
        if (it == obj.end() || it->is_null()) return fallback;
        const auto value = it->template get<std::string>();
        for (const auto& [name, mapped] : table) {
            if (value == name) return mapped;
        }
        return fallback;
    }

    // As ParseEnum, but the key must be present. Used where no default is
    // defensible: silently substituting one would turn a renamed or misspelled
    // key into a physically different problem that still solves, producing
    // plausible-looking but wrong results. Unknown *values* still fall back so
    // ConfigValidator can report them with the full list of valid options.
    template <typename E>
    [[nodiscard]] static E ParseRequiredEnum(const json& obj, const char* key, E fallback,
                                            const std::string& context,
                                            std::initializer_list<std::pair<std::string_view, E>> table) {
        const auto it = obj.find(key);
        if (it == obj.end() || it->is_null()) {
            throw std::runtime_error(context + ": missing required field '" + key + "'");
        }
        return ParseEnum(obj, key, fallback, table);
    }

    // "simulation.physics_type": electrostatics | magnetostatics | magnetoquasistatics.
    // Tolerant default; ConfigValidator enforces presence and validity.
    [[nodiscard]] ::PhysicsType GetPhysicsType() const;

    // "simulation.geometry_type": axisymmetric | planar | 3d.
    [[nodiscard]] ::GeometryType GetGeometryType() const;

    // "simulation.analysis_type": field | coupling_matrix.
    [[nodiscard]] ::AnalysisType GetAnalysisType() const;

    // Range checking lives in ConfigValidator; the parser only reads the value.
    [[nodiscard]] int GetOrder() const {
        return Get(Sim(), "order", 1);
    }

    [[nodiscard]] OutputSettings GetOutputSettings() const;

    // {"name", "points": [[x, y(, z)], ...]} or {"name", "line": {"from", "to",
    // "count"}} (count points evenly spaced, both ends included), with an
    // optional "entity_group". Shapes are checked by ConfigValidator.
    [[nodiscard]] static Probe GetProbe(const json& entry);

    // Parse the optional "simulation.amr" block. Missing keys fall back to the
    // AmrSettings defaults and unknown future keys are ignored, so the two sides
    // (C# requester / solver) can evolve independently. Numbers parse with the
    // invariant '.' separator regardless of locale.
    [[nodiscard]] AmrSettings GetAmrSettings() const;

    // Solver parameters, defaulted from Constants.
    [[nodiscard]] double GetSolverTolerance() const {
        return Get(Sim(), "solver_tolerance", Constants::DEFAULT_SOLVER_TOLERANCE);
    }

    [[nodiscard]] int GetSolverMaxIter() const {
        return Get(Sim(), "solver_max_iter", Constants::DEFAULT_SOLVER_MAX_ITER);
    }

    [[nodiscard]] int GetSolverPrintLevel() const {
        return Get(Sim(), "solver_print_level", Constants::DEFAULT_SOLVER_PRINT_LEVEL);
    }

    // "simulation.linear_solver": iterative | direct.
    //
    // The default depends on geometry_type. For the 2D models it is direct: it
    // factors once per mesh and reuses the factors across scenarios, and its
    // accuracy does not depend on a residual tolerance. For '3d' it is
    // iterative (multigrid-preconditioned CG), because the direct solver's
    // fill-in makes a 3D factorization of even ~100k unknowns take minutes.
    // The one exception is 3D magnetics in a serial build: its iterative
    // solver needs hypre's AMS, which only the MPI build has, so the default
    // there is the best solver the build actually provides.
    [[nodiscard]] ::LinearSolverType GetLinearSolver() const;

    // Relative mesh paths resolve against the directory holding the config file.
    [[nodiscard]] std::string GetMeshPath() const;

	std::unordered_map<std::string, EntityGroup> GetEntityGroups() const;

    std::map<std::string, Terminal> GetTerminals() const;

    // "terminals[].direction": {"type": "azimuthal", "origin", "axis"} |
    // {"type": "electrodes", "input", "output"} | {"type": "cut", "cut",
    // "normal"}. Tolerant defaults; the validator enforces a well-formed block.
    static CurrentDirection GetCurrentDirection(const json& d);

    std::vector<Region> GetRegions() const;

    std::map<std::string, Material> GetMaterials() const;

    // --------------------------------------------------------
    // Boundary conditions (far-field, symmetry, prescribed flux)
    // --------------------------------------------------------

    std::vector<BoundaryCondition> GetBoundaries() const;

    static BoundaryConditionType ParseBoundaryConditionType(const std::string& type);

    // --------------------------------------------------------
    // Scenarios (one solve each: parameters + per-terminal excitations)
    // --------------------------------------------------------
    static std::string SweepScenarioName(const std::string& base_name,
                                         int point,
                                         double frequency);

    // Excitation values are taken verbatim. For time-harmonic runs they are
    // PEAK amplitudes, with a phase in degrees; there is deliberately no
    // rms/peak selector, so an RMS value prescribed here propagates
    // unconverted. See Excitation in problem_config.hpp.
    static Scenario ParseScenarioExcitations(const json& source);

    std::vector<std::pair<std::string, Scenario>> GetScenarios() const;
};