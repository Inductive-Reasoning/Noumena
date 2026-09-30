// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "config/input_parser.hpp"
#include "io/mesh_loader.hpp"
#include "solvers/electrostatic_solver.hpp"
#include "solvers/magnetostatic_solver.hpp"
#include "solvers/magnetoquasistatic_solver.hpp"
#include "solvers/solver_factory.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <array>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <iomanip>

namespace fs = std::filesystem;

ProblemConfig DecodeConfig(const json& config, const std::string& archive = {});

namespace {
void CreatePlanarStripMesh(const std::string& filename,
                           double length, double height, int nx, int ny);
json MakePlanarStripConfig(const std::string& physics,
                           const std::string& mesh_file,
                           int order,
                           const json& material_properties,
                           double left_value,
                           double right_value);
const FieldExport& FindField(const FieldExportSet& fields,
                             const std::string& name);
mfem::IntegrationPoint TriangleCenter();
mfem::Vector PhysicalPoint(const mfem::GridFunction& field, int element,
                           const mfem::IntegrationPoint& point);
double SamplePrimaryScalar(const FieldExportSet& fields,
                           const std::string& name, int element,
                           const mfem::IntegrationPoint& point);
mfem::Vector SampleDerivedVector(const FieldExportSet& fields,
                                 const std::string& name, int element,
                                 const mfem::IntegrationPoint& point);
} // namespace

// Helper function to create a minimal test mesh file
void CreateTestMesh(const std::string& filename) {
    std::ofstream mesh_file(filename);
    // Create a simple 2D axisymmetric mesh (single triangle element)
    mesh_file << "MFEM mesh v1.0\n\n";
    mesh_file << "dimension\n2\n\n";
    mesh_file << "elements\n1\n";
    mesh_file << "1 2 0 1 2\n\n";  // Triangle with attribute 1
    mesh_file << "boundary\n3\n";
    mesh_file << "1 1 0 1\n";       // Boundary segment with attribute 1
    mesh_file << "1 1 1 2\n";       // Boundary segment with attribute 1
    mesh_file << "1 1 2 0\n\n";     // Boundary segment with attribute 1
    mesh_file << "vertices\n3\n2\n";
    mesh_file << "0.0 0.0\n";
    mesh_file << "1.0 0.0\n";
    mesh_file << "0.5 1.0\n";
    mesh_file.close();
}

TEST_CASE("Electrostatic solver applies natural-flux Neumann data",
          "[solvers][analytic][electrostatic][neumann]") {
    const std::string mesh_file = "test_electrostatic_neumann.mesh";
    constexpr double length = 0.2;
    constexpr double height = 0.05;
    constexpr double relative_permittivity = 2.5;
    constexpr double gradient = 3.0;
    constexpr int nx = 8;
    constexpr int ny = 2;
    CreatePlanarStripMesh(mesh_file, length, height, nx, ny);

    const double permittivity = Constants::EPSILON_0 * relative_permittivity;
    json config = MakePlanarStripConfig(
        "electrostatics", mesh_file, 1,
        {{"epsilon_r", relative_permittivity}}, 0.0, 0.0);
    config["boundary_conditions"][1]["type"] = "neumann";
    config["boundary_conditions"][1]["value"] = permittivity * gradient;

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    ElectrostaticSolver solver(mesh, DecodeConfig(config));
    solver.Setup();
    solver.Run();

    const FieldExportSet fields = solver.CollectExportFields();
    const mfem::IntegrationPoint point = TriangleCenter();
    const int element = 2 * (nx / 2);
    const FieldExport& potential_field = FindField(fields, "V");
    const mfem::Vector physical =
        PhysicalPoint(*potential_field.primary, element, point);

    REQUIRE(SamplePrimaryScalar(fields, "V", element, point) ==
        Catch::Approx(gradient * physical(0)).epsilon(1.0e-7));
    const mfem::Vector electric_field =
        SampleDerivedVector(fields, "E", element, point);
    REQUIRE(electric_field(0) == Catch::Approx(-gradient).epsilon(1.0e-7));
    REQUIRE(electric_field(1) == Catch::Approx(0.0).margin(1.0e-6));

    fs::remove(mesh_file);
}

// Robin condition eps dV/dn + alpha V = g on the right end of a strip grounded
// on the left. The exact solution is linear, V = c x with
// eps c + alpha c L = g, so P1 reproduces it exactly: this checks both the
// Robin operator term and its data load, and the sign convention (outward
// natural flux, the same as Neumann).
TEST_CASE("Electrostatic solver applies Robin data and coefficient",
          "[solvers][analytic][electrostatic][robin]") {
    const std::string mesh_file = "test_electrostatic_robin.mesh";
    constexpr double length = 0.2, height = 0.05, eps_r = 2.5, g = 4.0e-11;
    constexpr int nx = 8, ny = 2;
    CreatePlanarStripMesh(mesh_file, length, height, nx, ny);

    const double eps = Constants::EPSILON_0 * eps_r;
    const double alpha = 3.0 * eps / length;
    json config = MakePlanarStripConfig(
        "electrostatics", mesh_file, 1, {{"epsilon_r", eps_r}}, 0.0, 0.0);
    config["boundary_conditions"][1]["type"] = "robin";
    config["boundary_conditions"][1]["value"] = g;
    config["boundary_conditions"][1]["robin_coefficient"] = alpha;

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    ElectrostaticSolver solver(mesh, DecodeConfig(config));
    solver.Setup();
    solver.Run();

    const double slope = g / (eps + alpha * length);
    const FieldExportSet fields = solver.CollectExportFields();
    const mfem::IntegrationPoint point = TriangleCenter();
    const int element = 2 * (nx - 1);  // next to the Robin end
    const mfem::Vector physical =
        PhysicalPoint(*FindField(fields, "V").primary, element, point);
    REQUIRE(SamplePrimaryScalar(fields, "V", element, point) ==
        Catch::Approx(slope * physical(0)).epsilon(1.0e-7));

    SECTION("a negative coefficient is rejected") {
        config["boundary_conditions"][1]["robin_coefficient"] = -alpha;
        ElectrostaticSolver bad(mesh, DecodeConfig(config));
        REQUIRE_THROWS_WITH(bad.Setup(),
            Catch::Matchers::ContainsSubstring("negative robin_coefficient"));
    }

    SECTION("magnetics still rejects Robin") {
        json magnetic = MakePlanarStripConfig(
            "magnetostatics", mesh_file, 1, {{"mu_r", 1.0}}, 0.0, 0.0);
        magnetic["boundary_conditions"][1]["type"] = "robin";
        magnetic["boundary_conditions"][1]["robin_coefficient"] = 1.0;
        MagnetostaticSolver ms(mesh, DecodeConfig(magnetic));
        REQUIRE_THROWS_WITH(ms.Setup(),
            Catch::Matchers::ContainsSubstring("not implemented for magnetostatics"));
    }

    fs::remove(mesh_file);
}

TEST_CASE("Boundary closures and voltage terminals have distinct ownership",
          "[solvers][boundaries][overlap]") {
    const std::string mesh_file = "test_boundary_ownership.mesh";
    CreatePlanarStripMesh(mesh_file, 0.2, 0.05, 2, 1);

    auto terminal_config = [&]() {
        json config = MakePlanarStripConfig(
            "electrostatics", mesh_file, 1, {{"epsilon_r", 1.0}}, 0.0, 0.0);
        config["terminals"] = json::array({
            {{"name", "LeftTerminal"}, {"quantity", "voltage"},
             {"entity_group", "Left"}}
        });
        config["boundary_conditions"].erase(config["boundary_conditions"].begin());
        return config;
    };

    SECTION("same boundary attribute is rejected") {
        json config = terminal_config();
        config["boundary_conditions"].push_back(
            {{"name", "LeftFlux"}, {"type", "neumann"},
             {"entity_group", "Left"}, {"value", 1.0}});

        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        ElectrostaticSolver solver(mesh, DecodeConfig(config));
        REQUIRE_THROWS_WITH(solver.Setup(),
            Catch::Matchers::ContainsSubstring("one physical role"));
    }

    SECTION("different attributes may meet at a corner") {
        json config = terminal_config();
        config["entity_groups"].push_back(
            {{"name", "Horizontal"}, {"dim", 1}, {"attribute_ids", {3}}});
        config["boundary_conditions"].push_back(
            {{"name", "HorizontalFlux"}, {"type", "neumann"},
             {"entity_group", "Horizontal"}, {"value", 0.0}});

        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        ElectrostaticSolver solver(mesh, DecodeConfig(config));
        REQUIRE_NOTHROW(solver.Setup());
    }

    fs::remove(mesh_file);
}

TEST_CASE("Axisymmetric magnetic solvers enforce zero A_phi on the axis",
          "[solvers][axisymmetric][axis][boundaries]") {
    const std::string mesh_file = "test_magnetic_axis_boundary.mesh";
    CreatePlanarStripMesh(mesh_file, 0.2, 0.05, 2, 1);

    auto magnetic_config = [&](const std::string& physics, double axis_value) {
        const json material = physics == "magnetoquasistatics"
            ? json{{"mu_r", 1.0}, {"sigma", 0.0}}
            : json{{"mu_r", 1.0}};
        json config = MakePlanarStripConfig(
            physics, mesh_file, 1, material, axis_value, 0.0);
        config["simulation"]["geometry_type"] = "axisymmetric";
        if (physics == "magnetoquasistatics") {
            config["scenarios"][0]["frequency"] = 60.0;
        }
        return config;
    };

    auto add_horizontal_dirichlet = [](json& config, double value) {
        config["entity_groups"].push_back(
            {{"name", "Horizontal"}, {"dim", 1}, {"attribute_ids", {3}}});
        config["boundary_conditions"].push_back(
            {{"name", "Horizontal"}, {"type", "dirichlet"},
             {"entity_group", "Horizontal"}, {"value", value}});
    };

    SECTION("magnetostatics rejects a nonzero axis value") {
        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        MagnetostaticSolver solver(
            mesh, DecodeConfig(magnetic_config("magnetostatics", 1.0)));
        REQUIRE_THROWS_WITH(solver.Setup(),
            Catch::Matchers::ContainsSubstring("Axis regularity requires A_phi = 0"));
    }

    SECTION("magnetoquasistatics rejects a nonzero axis value") {
        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        MagnetoquasistaticSolver solver(
            mesh, DecodeConfig(magnetic_config("magnetoquasistatics", 1.0)));
        REQUIRE_THROWS_WITH(solver.Setup(),
            Catch::Matchers::ContainsSubstring("Axis regularity requires A_phi = 0"));
    }

    SECTION("an explicit zero axis value is accepted") {
        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        MagnetostaticSolver solver(
            mesh, DecodeConfig(magnetic_config("magnetostatics", 0.0)));
        REQUIRE_NOTHROW(solver.Setup());
    }

    SECTION("magnetostatics rejects a nonzero value at an axis corner") {
        json config = magnetic_config("magnetostatics", 0.0);
        add_horizontal_dirichlet(config, 1.0);
        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        MagnetostaticSolver solver(mesh, DecodeConfig(config));
        REQUIRE_THROWS_WITH(solver.Setup(),
            Catch::Matchers::ContainsSubstring("on the magnetic symmetry axis"));
    }

    SECTION("magnetoquasistatics rejects a nonzero value at an axis corner") {
        json config = magnetic_config("magnetoquasistatics", 0.0);
        add_horizontal_dirichlet(config, 1.0);
        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        MagnetoquasistaticSolver solver(mesh, DecodeConfig(config));
        REQUIRE_THROWS_WITH(solver.Setup(),
            Catch::Matchers::ContainsSubstring("on the magnetic symmetry axis"));
    }

    SECTION("a zero value on a boundary adjacent to the axis is accepted") {
        json config = magnetic_config("magnetostatics", 0.0);
        add_horizontal_dirichlet(config, 0.0);
        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        MagnetostaticSolver solver(mesh, DecodeConfig(config));
        REQUIRE_NOTHROW(solver.Setup());
    }

    SECTION("planar problems do not apply magnetic axis regularity") {
        json config = magnetic_config("magnetostatics", 1.0);
        config["simulation"]["geometry_type"] = "planar";
        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        MagnetostaticSolver solver(mesh, DecodeConfig(config));
        REQUIRE_NOTHROW(solver.Setup());
    }

    SECTION("annular domains may use a nonzero inner-boundary value") {
        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        for (int v = 0; v < mesh.GetNV(); ++v) {
            mesh.GetVertex(v)[0] += 1.0;
        }
        MagnetostaticSolver solver(
            mesh, DecodeConfig(magnetic_config("magnetostatics", 1.0)));
        REQUIRE_NOTHROW(solver.Setup());
    }

    fs::remove(mesh_file);
}

ProblemConfig DecodeConfig(const json& config, const std::string& archive) {
    json source = config;
    if (!archive.empty()) source["output"]["hdf5"]["file"] = fs::absolute(archive).string();
    return InputParser(source).GetProblemConfig();
}

TEST_CASE("Magnetoquasistatic Neumann data loads only the real field",
          "[solvers][analytic][mqs][neumann]") {
    const std::string mesh_file = "test_mqs_neumann.mesh";
    constexpr double length = 0.1;
    constexpr double height = 0.02;
    constexpr double gradient = 0.75;
    constexpr int nx = 8;
    constexpr int ny = 2;
    CreatePlanarStripMesh(mesh_file, length, height, nx, ny);

    const double reluctivity = 1.0 / Constants::MU_0;
    json config = MakePlanarStripConfig(
        "magnetoquasistatics", mesh_file, 1,
        {{"mu_r", 1.0}, {"sigma", 0.0}}, 0.0, 0.0);
    config["boundary_conditions"][1]["type"] = "neumann";
    config["boundary_conditions"][1]["value"] = reluctivity * gradient;
    config["scenarios"][0]["frequency"] = 60.0;

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    MagnetoquasistaticSolver solver(mesh, DecodeConfig(config));
    solver.Setup();
    solver.Run();

    const FieldExportSet fields = solver.CollectExportFields();
    const mfem::IntegrationPoint point = TriangleCenter();
    const int element = 2 * (nx / 2);
    const FieldExport& real_field = FindField(fields, "A_Real");
    const mfem::Vector physical = PhysicalPoint(*real_field.primary, element, point);

    REQUIRE(SamplePrimaryScalar(fields, "A_Real", element, point) ==
        Catch::Approx(gradient * physical(0)).epsilon(1.0e-8));
    REQUIRE(SamplePrimaryScalar(fields, "A_Imag", element, point) ==
        Catch::Approx(0.0).margin(1.0e-10));

    fs::remove(mesh_file);
}

TEST_CASE("Only electrostatics accepts Robin boundary conditions",
          "[solvers][boundaries][robin]") {
    const std::string mesh_file = "test_robin_rejection.mesh";
    CreatePlanarStripMesh(mesh_file, 0.1, 0.02, 2, 1);

    auto robin_config = [&](const std::string& physics, const json& material) {
        json config = MakePlanarStripConfig(
            physics, mesh_file, 1, material, 0.0, 0.0);
        config["boundary_conditions"][1]["type"] = "robin";
        config["boundary_conditions"][1]["robin_coefficient"] = 1.0;
        if (physics == "magnetoquasistatics") {
            config["scenarios"][0]["frequency"] = 60.0;
        }
        return config;
    };

    SECTION("electrostatics assembles Robin") {
        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        ElectrostaticSolver solver(mesh, DecodeConfig(
            robin_config("electrostatics", {{"epsilon_r", 1.0}})));
        REQUIRE_NOTHROW(solver.Setup());
    }

    SECTION("magnetostatics") {
        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        MagnetostaticSolver solver(mesh, DecodeConfig(
            robin_config("magnetostatics", {{"mu_r", 1.0}})));
        REQUIRE_THROWS_WITH(solver.Setup(),
            Catch::Matchers::ContainsSubstring("not implemented for"));
    }

    SECTION("magnetoquasistatics") {
        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        MagnetoquasistaticSolver solver(mesh, DecodeConfig(
            robin_config("magnetoquasistatics",
                         {{"mu_r", 1.0}, {"sigma", 0.0}})));
        REQUIRE_THROWS_WITH(solver.Setup(),
            Catch::Matchers::ContainsSubstring("not implemented for"));
    }

    fs::remove(mesh_file);
}

class ScenarioOrderProbe : public PhysicsSolver {
public:
    ScenarioOrderProbe(mfem::Mesh& mesh, const ProblemConfig& config)
        : PhysicsSolver(mesh, config) {}

    std::vector<std::string> ScenarioNames() const {
        std::vector<std::string> names;
        for (const auto& [name, scenario] : BuildSolveScenarios()) {
            names.push_back(name);
        }
        return names;
    }

    void Setup() override {}
    void SaveAnalysisResults() override {}
    FieldExportSet CollectExportFields() const override { return {}; }

protected:
    void BuildOperators() override {}
    void RunOnCurrentMesh() override {}
    void EstimateCurrentSolutionError(mfem::Vector& errors) override {
        errors.SetSize(mesh.GetNE());
        errors = 0.0;
    }
    double ComputePeakFieldMagnitude() const override { return 0.0; }
};

class AmrLifecycleProbe : public PhysicsSolver {
public:
    AmrLifecycleProbe(mfem::Mesh& mesh, const ProblemConfig& config)
        : PhysicsSolver(mesh, config) {}

    void Setup() override {
        fec = std::make_unique<mfem::H1_FECollection>(1, mesh.Dimension());
        BuildOperators();
    }

    void SaveAnalysisResults() override {}
    FieldExportSet CollectExportFields() const override { return {}; }

    int OperatorBuilds() const { return operator_builds; }
    int ErrorEstimates() const { return error_estimates; }
    int SolvePasses() const { return solve_passes; }
    const std::vector<amr::AmrIterationInfo>& AmrHistory() const {
        return GetAmrHistory();
    }

protected:
    void BuildOperators() override {
        ++operator_builds;
        fespace = std::make_unique<mfem::FiniteElementSpace>(&mesh, fec.get());
    }

    // One scenario per pass, folding its error indicator in as production
    // solvers do.
    void RunOnCurrentMesh() override {
        ++solve_passes;
        AccumulateScenarioError();
    }

    void EstimateCurrentSolutionError(mfem::Vector& errors) override {
        ++error_estimates;
        errors.SetSize(mesh.GetNE());
        errors = 1.0;
    }

    double ComputePeakFieldMagnitude() const override { return 2.5; }

private:
    int operator_builds = 0;
    int error_estimates = 0;
    int solve_passes = 0;
};

class ElectrostaticAmrProbe : public ElectrostaticSolver {
public:
    using ElectrostaticSolver::ElectrostaticSolver;

    const std::vector<amr::AmrIterationInfo>& AmrHistory() const {
        return GetAmrHistory();
    }

    mfem::Vector LocalErrors() {
        mfem::Vector errors;
        EstimateCurrentSolutionError(errors);
        return errors;
    }
};

class MagnetostaticAmrProbe : public MagnetostaticSolver {
public:
    using MagnetostaticSolver::MagnetostaticSolver;

    mfem::Vector LocalErrors() {
        mfem::Vector errors;
        EstimateCurrentSolutionError(errors);
        return errors;
    }
};

TEST_CASE("ElectrostaticSolver can be constructed", "[solvers]") {
    std::string mesh_file = "test_electrostatic.mesh";
    CreateTestMesh(mesh_file);

    json test_config = {
        {"simulation", {
            {"physics_type", "electrostatics"},
            {"mesh", mesh_file},
            {"order", 1},
            {"geometry_type", "axisymmetric"}
        }},
        {"entity_groups", json::array({
            {{"name", "Domain"}, {"dim", 2}, {"attribute_ids", {1}}},
            {{"name", "Ground"}, {"dim", 1}, {"attribute_ids", {1}}}
        })},
        {"regions", json::array({
            {{"entity_group", "Domain"}, {"material", "dielectric"}}
        })},
        {"materials", json::array({
            {
                {"name", "dielectric"},
                {"properties", {{"epsilon_r", 2.0}}}
            }
        })},
        {"boundary_conditions", json::array({
            {
                {"name", "ground"},
                {"entity_group", "Ground"},
                {"type", "dirichlet"},
                {"value", 0.0}
            }
        })}
    };

    InputParser parser(test_config);
    ProblemConfig config = parser.GetProblemConfig();
    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);

    REQUIRE_NOTHROW(ElectrostaticSolver(mesh, config));

    // Cleanup
    fs::remove(mesh_file);
}

TEST_CASE("PhysicsSolver owns the adaptive lifecycle", "[solvers][amr]") {
    const std::string mesh_file = "test_amr_lifecycle.mesh";
    CreateTestMesh(mesh_file);
    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);

    ProblemConfig config;
    config.Amr.Enabled = true;
    config.Amr.MaxIterations = 3;
    config.Amr.MaxDofs = 0;
    config.Amr.ErrorFraction = 1.0;

    AmrLifecycleProbe probe(mesh, config);
    probe.Setup();
    probe.Run();

    REQUIRE(probe.OperatorBuilds() == 3);
    REQUIRE(probe.ErrorEstimates() == 3);
    // The AMR loop now runs exactly one solve pass per iteration; the last one
    // is on the converged mesh, so there is no extra pass afterwards.
    REQUIRE(probe.SolvePasses() == 3);
    REQUIRE(probe.AmrHistory().size() == 3);
    REQUIRE(probe.AmrHistory().back().peak_field_magnitude == 2.5);
    REQUIRE(mesh.GetNE() > 1);
    REQUIRE_FALSE(mesh.Nonconforming());

    fs::remove(mesh_file);
}

TEST_CASE("Field scenarios preserve declaration order", "[solvers][scenarios]") {
    const std::string mesh_file = "test_scenario_order.mesh";
    CreateTestMesh(mesh_file);
    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);

    ProblemConfig config;
    config.AnalysisType = AnalysisType::Field;
    config.Scenarios.emplace_back("Zulu", Scenario{});
    config.Scenarios.emplace_back("Alpha", Scenario{});
    config.Scenarios.emplace_back("Middle", Scenario{});

    ScenarioOrderProbe probe(mesh, config);
    REQUIRE((probe.ScenarioNames() == std::vector<std::string>{"Zulu", "Alpha", "Middle"}));

    fs::remove(mesh_file);
}

TEST_CASE("MagnetostaticSolver can be constructed", "[solvers]") {
    std::string mesh_file = "test_magnetostatic.mesh";
    CreateTestMesh(mesh_file);

    json test_config = {
        {"simulation", {
            {"physics_type", "magnetostatics"},
            {"mesh", mesh_file},
            {"order", 1},
            {"geometry_type", "axisymmetric"}
        }},
        {"entity_groups", json::array({
            {{"name", "Domain"}, {"dim", 2}, {"attribute_ids", {1}}},
            {{"name", "FarField"}, {"dim", 1}, {"attribute_ids", {1}}}
        })},
        {"regions", json::array({
            {{"entity_group", "Domain"}, {"material", "iron"}}
        })},
        {"materials", json::array({
            {
                {"name", "iron"},
                {"properties", {{"mu_r", 1000.0}}}
            }
        })},
        {"boundary_conditions", json::array({
            {
                {"name", "far_field"},
                {"entity_group", "FarField"},
                {"type", "dirichlet"},
                {"value", 0.0}
            }
        })}
    };

    InputParser parser(test_config);
    ProblemConfig config = parser.GetProblemConfig();
    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);

    REQUIRE_NOTHROW(MagnetostaticSolver(mesh, config));

    // Cleanup
    fs::remove(mesh_file);
}

TEST_CASE("MagnetoquasistaticSolver can be constructed", "[solvers]") {
    std::string mesh_file = "test_mqs.mesh";
    CreateTestMesh(mesh_file);

    json test_config = {
        {"simulation", {
            {"physics_type", "magnetoquasistatics"},
            {"mesh", mesh_file},
            {"order", 1},
            {"geometry_type", "axisymmetric"}
        }},
        {"entity_groups", json::array({
            {{"name", "Domain"}, {"dim", 2}, {"attribute_ids", {1}}},
            {{"name", "FarField"}, {"dim", 1}, {"attribute_ids", {1}}}
        })},
        {"regions", json::array({
            {{"entity_group", "Domain"}, {"material", "conductor"}}
        })},
        {"materials", json::array({
            {
                {"name", "conductor"},
                {"properties", {
                    {"mu_r", 1.0},
                    {"sigma", 5.8e7}
                }}
            }
        })},
        {"boundary_conditions", json::array({
            {
                {"name", "far_field"},
                {"entity_group", "FarField"},
                {"type", "dirichlet"},
                {"value", 0.0}
            }
        })},
        {"scenarios", json::array({
            {{"name", "default"}, {"frequency", 60.0}, {"excitations", json::array()}}
        })}
    };

    InputParser parser(test_config);
    ProblemConfig config = parser.GetProblemConfig();
    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);

    REQUIRE_NOTHROW(MagnetoquasistaticSolver(mesh, config));

    // Cleanup
    fs::remove(mesh_file);
}

TEST_CASE("Solver factory logic works correctly", "[solvers]") {
    std::string mesh_file = "test_factory.mesh";
    CreateTestMesh(mesh_file);
    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);

    SECTION("electrostatics") {
        ProblemConfig config;
        config.PhysicsType = PhysicsType::Electrostatics;
        auto solver = SolverFactory::Instance().Create(mesh, config);
        REQUIRE(dynamic_cast<ElectrostaticSolver*>(solver.get()) != nullptr);
    }

    SECTION("magnetostatics") {
        ProblemConfig config;
        config.PhysicsType = PhysicsType::Magnetostatics;
        auto solver = SolverFactory::Instance().Create(mesh, config);
        REQUIRE(dynamic_cast<MagnetostaticSolver*>(solver.get()) != nullptr);
    }

    SECTION("magnetoquasistatics") {
        ProblemConfig config;
        config.PhysicsType = PhysicsType::Magnetoquasistatics;
        auto solver = SolverFactory::Instance().Create(mesh, config);
        REQUIRE(dynamic_cast<MagnetoquasistaticSolver*>(solver.get()) != nullptr);
    }

    // Cleanup
    fs::remove(mesh_file);
}

// ===========================================================================
// Adaptive Mesh Refinement (AMR) regression tests
// ===========================================================================

namespace {

void CreatePlanarStripMesh(const std::string& filename,
                           double length, double height,
                           int nx, int ny) {
    const int nvx = nx + 1;
    const int nvy = ny + 1;
    auto vertex = [nvx](int i, int j) { return j * nvx + i; };

    std::ofstream mesh_file(filename);
    mesh_file << "MFEM mesh v1.0\n\n";
    mesh_file << "dimension\n2\n\n";
    mesh_file << "elements\n" << 2 * nx * ny << "\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int v00 = vertex(i, j);
            const int v10 = vertex(i + 1, j);
            const int v11 = vertex(i + 1, j + 1);
            const int v01 = vertex(i, j + 1);
            mesh_file << "1 2 " << v00 << " " << v10 << " " << v11 << "\n";
            mesh_file << "1 2 " << v00 << " " << v11 << " " << v01 << "\n";
        }
    }

    mesh_file << "\nboundary\n" << 2 * ny + 2 * nx << "\n";
    for (int j = 0; j < ny; ++j) {
        mesh_file << "1 1 " << vertex(0, j) << " " << vertex(0, j + 1) << "\n";
        mesh_file << "2 1 " << vertex(nx, j) << " " << vertex(nx, j + 1) << "\n";
    }
    for (int i = 0; i < nx; ++i) {
        mesh_file << "3 1 " << vertex(i, 0) << " " << vertex(i + 1, 0) << "\n";
        mesh_file << "3 1 " << vertex(i, ny) << " " << vertex(i + 1, ny) << "\n";
    }

    mesh_file << "\nvertices\n" << nvx * nvy << "\n2\n";
    for (int j = 0; j < nvy; ++j) {
        const double y = height * static_cast<double>(j) / ny;
        for (int i = 0; i < nvx; ++i) {
            const double x = length * static_cast<double>(i) / nx;
            mesh_file << x << " " << y << "\n";
        }
    }
}

void CreateLayeredStripMesh(const std::string& filename,
                            double length, double height,
                            int nx, int ny, int interface_column) {
    REQUIRE(interface_column > 0);
    REQUIRE(interface_column < nx);

    const int nvx = nx + 1;
    const int nvy = ny + 1;
    auto vertex = [nvx](int i, int j) { return j * nvx + i; };

    std::ofstream mesh_file(filename);
    mesh_file << "MFEM mesh v1.0\n\n";
    mesh_file << "dimension\n2\n\n";
    mesh_file << "elements\n" << 2 * nx * ny << "\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int attribute = i < interface_column ? 1 : 2;
            const int v00 = vertex(i, j);
            const int v10 = vertex(i + 1, j);
            const int v11 = vertex(i + 1, j + 1);
            const int v01 = vertex(i, j + 1);
            mesh_file << attribute << " 2 " << v00 << " " << v10 << " " << v11 << "\n";
            mesh_file << attribute << " 2 " << v00 << " " << v11 << " " << v01 << "\n";
        }
    }

    mesh_file << "\nboundary\n" << 2 * ny + 2 * nx << "\n";
    for (int j = 0; j < ny; ++j) {
        mesh_file << "1 1 " << vertex(0, j) << " " << vertex(0, j + 1) << "\n";
        mesh_file << "2 1 " << vertex(nx, j) << " " << vertex(nx, j + 1) << "\n";
    }
    for (int i = 0; i < nx; ++i) {
        mesh_file << "3 1 " << vertex(i, 0) << " " << vertex(i + 1, 0) << "\n";
        mesh_file << "3 1 " << vertex(i, ny) << " " << vertex(i + 1, ny) << "\n";
    }

    mesh_file << "\nvertices\n" << nvx * nvy << "\n2\n";
    for (int j = 0; j < nvy; ++j) {
        const double y = height * static_cast<double>(j) / ny;
        for (int i = 0; i < nvx; ++i) {
            const double x = length * static_cast<double>(i) / nx;
            mesh_file << x << " " << y << "\n";
        }
    }
}

json MakePlanarStripConfig(const std::string& physics,
                           const std::string& mesh_file,
                           int order,
                           const json& material_properties,
                           double left_value,
                           double right_value) {
    return json{
        {"simulation", {
            {"physics_type", physics},
            {"mesh", mesh_file},
            {"order", order},
            {"geometry_type", "planar"},
            {"analysis_type", "field"},
            {"solver_tolerance", 1e-12},
            {"solver_max_iter", 4000},
            {"solver_print_level", 0}
        }},
        {"entity_groups", json::array({
            {{"name", "Domain"}, {"dim", 2}, {"attribute_ids", {1}}},
            {{"name", "Left"},   {"dim", 1}, {"attribute_ids", {1}}},
            {{"name", "Right"},  {"dim", 1}, {"attribute_ids", {2}}}
        })},
        {"regions", json::array({
            {{"name", "Domain"}, {"entity_group", "Domain"}, {"material", "Material"}}
        })},
        {"materials", json::array({
            {{"name", "Material"}, {"properties", material_properties}}
        })},
        {"boundary_conditions", json::array({
            {{"name", "Left"},  {"type", "dirichlet"}, {"entity_group", "Left"},  {"value", left_value}},
            {{"name", "Right"}, {"type", "dirichlet"}, {"entity_group", "Right"}, {"value", right_value}}
        })},
        {"scenarios", json::array({
            {{"name", "analytic"}, {"excitations", json::array()}}
        })}
    };
}

const FieldExport& FindField(const FieldExportSet& fields, const std::string& name) {
    const auto& exported = fields.Fields();
    const auto field = std::find_if(exported.begin(), exported.end(),
        [&name](const FieldExport& candidate) { return candidate.name == name; });
    REQUIRE(field != exported.end());
    return *field;
}

mfem::IntegrationPoint TriangleCenter() {
    mfem::IntegrationPoint point;
    point.Set2(1.0 / 3.0, 1.0 / 3.0);
    return point;
}

mfem::Vector PhysicalPoint(const mfem::GridFunction& field, int element,
                           const mfem::IntegrationPoint& point) {
    mfem::Vector physical(2);
    field.FESpace()->GetElementTransformation(element)->Transform(point, physical);
    return physical;
}

double SamplePrimaryScalar(const FieldExportSet& fields, const std::string& name,
                           int element, const mfem::IntegrationPoint& point) {
    const FieldExport& field = FindField(fields, name);
    REQUIRE(field.kind == FieldExport::Kind::Primary);
    return field.primary->GetValue(element, point);
}

mfem::Vector SampleDerivedVector(const FieldExportSet& fields, const std::string& name,
                                 int element, const mfem::IntegrationPoint& point) {
    const FieldExport& field = FindField(fields, name);
    REQUIRE(field.kind == FieldExport::Kind::DerivedVector);
    const auto& exported = fields.Fields();
    const auto primary = std::find_if(exported.begin(), exported.end(),
        [](const FieldExport& candidate) { return candidate.primary != nullptr; });
    REQUIRE(primary != exported.end());
    mfem::ElementTransformation* transformation =
        primary->primary->FESpace()->GetElementTransformation(element);
    mfem::Vector value(field.vector->GetVDim());
    field.vector->Eval(value, *transformation, point);
    return value;
}

double IntegrateVectorMagnitudeSquared(const FieldExportSet& fields,
                                       const std::string& name) {
    const FieldExport& field = FindField(fields, name);
    REQUIRE(field.kind == FieldExport::Kind::DerivedVector);
    const auto& exported = fields.Fields();
    const auto primary = std::find_if(exported.begin(), exported.end(),
        [](const FieldExport& candidate) { return candidate.primary != nullptr; });
    REQUIRE(primary != exported.end());

    const mfem::FiniteElementSpace* space = primary->primary->FESpace();
    double integral = 0.0;
    for (int element = 0; element < space->GetNE(); ++element) {
        const mfem::FiniteElement* finite_element = space->GetFE(element);
        mfem::ElementTransformation* transformation =
            space->GetElementTransformation(element);
        const mfem::IntegrationRule& rule = mfem::IntRules.Get(
            finite_element->GetGeomType(), 2 * finite_element->GetOrder() + 2);
        for (int q = 0; q < rule.GetNPoints(); ++q) {
            const mfem::IntegrationPoint& point = rule.IntPoint(q);
            transformation->SetIntPoint(&point);
            mfem::Vector value(field.vector->GetVDim());
            field.vector->Eval(value, *transformation, point);
            integral += point.weight * transformation->Weight() * (value * value);
        }
    }
    return integral;
}

double RelativeComplexL2Error(
    const FieldExportSet& fields,
    const std::function<std::complex<double>(const mfem::Vector&)>& exact) {
    const FieldExport& real_field = FindField(fields, "A_Real");
    const FieldExport& imag_field = FindField(fields, "A_Imag");
    REQUIRE(real_field.kind == FieldExport::Kind::Primary);
    REQUIRE(imag_field.kind == FieldExport::Kind::Primary);

    const mfem::FiniteElementSpace* space = real_field.primary->FESpace();
    double error_squared = 0.0;
    double exact_squared = 0.0;
    for (int element = 0; element < space->GetNE(); ++element) {
        const mfem::FiniteElement* finite_element = space->GetFE(element);
        mfem::ElementTransformation* transformation =
            space->GetElementTransformation(element);
        const mfem::IntegrationRule& rule = mfem::IntRules.Get(
            finite_element->GetGeomType(), 2 * finite_element->GetOrder() + 4);
        for (int q = 0; q < rule.GetNPoints(); ++q) {
            const mfem::IntegrationPoint& point = rule.IntPoint(q);
            transformation->SetIntPoint(&point);
            mfem::Vector physical(2);
            transformation->Transform(point, physical);
            const std::complex<double> expected = exact(physical);
            const std::complex<double> actual(
                real_field.primary->GetValue(element, point),
                imag_field.primary->GetValue(element, point));
            const double weight = point.weight * transformation->Weight();
            error_squared += weight * std::norm(actual - expected);
            exact_squared += weight * std::norm(expected);
        }
    }
    return std::sqrt(error_squared / exact_squared);
}

// Observed order of accuracy between two errors on meshes differing by a
// uniform refinement factor. For e ~ C h^p and h halving, this returns p.
//
// This is strictly stronger than asserting the errors merely shrink: a solver
// that silently drops from O(h^2) to O(h) still reduces error under refinement
// and would pass a monotonicity check, but shows up immediately here.
double ObservedOrder(double coarse_error, double fine_error,
                     double refinement_ratio = 2.0) {
    REQUIRE(coarse_error > 0.0);
    REQUIRE(fine_error > 0.0);
    return std::log(coarse_error / fine_error) / std::log(refinement_ratio);
}

struct CouplingMatrix {
    std::vector<std::string> labels;
    std::vector<std::vector<double>> values;
};

CouplingMatrix ReadHdf5Matrix(const std::string& filename,
                             const std::string& quantity,
                             std::size_t frequency_index = 0) {
    HighFive::File file(filename, HighFive::File::ReadOnly);
    CouplingMatrix matrix;
    file.getDataSet("/coupling/terminal_names").read(matrix.labels);
    auto dataset = file.getDataSet("/coupling/" + quantity + "/values");
    const auto dimensions = dataset.getDimensions();
    if (dimensions.size() == 2) {
        dataset.read(matrix.values);
    } else {
        REQUIRE(dimensions.size() == 3);
        REQUIRE(frequency_index < dimensions[0]);
        std::vector<std::vector<std::vector<double>>> values;
        dataset.read(values);
        matrix.values = values[frequency_index];
    }
    for (const auto& row : matrix.values) {
        REQUIRE(row.size() == matrix.labels.size());
    }
    REQUIRE(matrix.values.size() == matrix.labels.size());
    return matrix;
}

void RequireMatricesEqual(const CouplingMatrix& actual, const CouplingMatrix& expected,
                          double tolerance = 1e-10) {
    REQUIRE(actual.labels == expected.labels);
    REQUIRE(actual.values.size() == expected.values.size());
    for (std::size_t row = 0; row < actual.values.size(); ++row) {
        REQUIRE(actual.values[row].size() == expected.values[row].size());
        for (std::size_t column = 0; column < actual.values[row].size(); ++column) {
            REQUIRE(actual.values[row][column] ==
                Catch::Approx(expected.values[row][column]).epsilon(tolerance));
        }
    }
}

// Build a 2D axisymmetric "coaxial capacitor" triangle mesh spanning
// r in [r_inner, r_outer], z in [0, height], and write it as an MFEM v1.0 mesh.
// Boundary attributes are assigned by geometry (independent of MFEM's internal
// edge numbering): 1 = inner conductor (r=r_inner), 2 = outer conductor
// (r=r_outer), 3 = top/bottom symmetry edges (z=0 or z=height). This exercises
// the axisymmetric ZienkiewiczZhu flux/energy path, and the analytic field
// E_r ~ 1/r concentrates error near the inner conductor so AMR has something to
// resolve.
void CreateCoaxMesh(const std::string& filename,
                    double r_inner, double r_outer, double height,
                    int nr, int nz, int interface_column = 0) {
    const int nvr = nr + 1;
    const int nvz = nz + 1;
    auto vid = [nvr](int i, int j) { return j * nvr + i; }; // i along r, j along z

    std::vector<std::array<double, 2>> verts;
    verts.reserve(static_cast<size_t>(nvr) * nvz);
    for (int j = 0; j < nvz; ++j) {
        const double z = height * static_cast<double>(j) / nz;
        for (int i = 0; i < nvr; ++i) {
            const double r = r_inner + (r_outer - r_inner) * static_cast<double>(i) / nr;
            verts.push_back({ r, z });
        }
    }

    std::ofstream m(filename);
    m << "MFEM mesh v1.0\n\n";
    m << "dimension\n2\n\n";

    // Two triangles per cell. With a positive interface column, radial cells to
    // its right use domain attribute 2; otherwise the whole domain is attribute 1.
    m << "elements\n" << (2 * nr * nz) << "\n";
    for (int j = 0; j < nz; ++j) {
        for (int i = 0; i < nr; ++i) {
            const int attribute = interface_column > 0 && i >= interface_column ? 2 : 1;
            const int v00 = vid(i, j);
            const int v10 = vid(i + 1, j);
            const int v11 = vid(i + 1, j + 1);
            const int v01 = vid(i, j + 1);
            m << attribute << " 2 " << v00 << " " << v10 << " " << v11 << "\n";
            m << attribute << " 2 " << v00 << " " << v11 << " " << v01 << "\n";
        }
    }
    m << "\n";

    // Boundary edges with geometry-based attributes.
    std::vector<std::array<int, 3>> bdr; // {attr, va, vb}
    for (int j = 0; j < nz; ++j) {       // vertical edges at r=r_inner / r=r_outer
        bdr.push_back({ 1, vid(0, j),  vid(0, j + 1) });
        bdr.push_back({ 2, vid(nr, j), vid(nr, j + 1) });
    }
    for (int i = 0; i < nr; ++i) {       // horizontal edges at z=0 / z=height
        bdr.push_back({ 3, vid(i, 0),  vid(i + 1, 0) });
        bdr.push_back({ 3, vid(i, nz), vid(i + 1, nz) });
    }

    m << "boundary\n" << bdr.size() << "\n";
    for (const auto& b : bdr) {
        m << b[0] << " 1 " << b[1] << " " << b[2] << "\n";
    }
    m << "\n";

    m << "vertices\n" << verts.size() << "\n2\n";
    for (const auto& v : verts) {
        m << v[0] << " " << v[1] << "\n";
    }
    m.close();
}

// Build a 2D axisymmetric shield-and-turns mesh: an aluminum shield band
// nearest the axis, then two rectangular conductor turns further out in r,
// each separated by air. All conductors are held off the symmetry axis so the
// massive-port conductance integral sigma/(2*pi*r) stays finite.
//
// Radial layout (r increasing), with the domain attribute in parentheses:
//   air (1) | shield (2) | air (1) | turn A (3) | air (1) | turn B (4) | air (1)
//
// Boundary attributes: 1 = inner (r=r_min), 2 = outer (r=r_max),
// 3 = top/bottom (z=0 or z=height).
void CreateShieldedTurnsMesh(const std::string& filename,
                             double r_min, double r_max, double height,
                             int nz = 4, int cells_per_band = 2) {
    // Radial band edges as fractions of the span, one entry per band, each
    // paired with the domain attribute carried by that band.
    struct Band { int cells; int attribute; };
    const std::vector<Band> bands{
        { cells_per_band, 1 },  // air gap next to the axis side
        { cells_per_band, 2 },  // aluminum shield
        { cells_per_band, 1 },  // air between shield and turns
        { cells_per_band, 3 },  // turn A
        { cells_per_band, 1 },  // air between turns
        { cells_per_band, 4 },  // turn B
        { cells_per_band, 1 }   // air out to the far boundary
    };

    int nr = 0;
    for (const Band& band : bands) nr += band.cells;
    const int nvr = nr + 1;
    const int nvz = nz + 1;
    auto vid = [nvr](int i, int j) { return j * nvr + i; };

    // Per-radial-cell attribute, expanded from the band table.
    std::vector<int> cell_attribute;
    cell_attribute.reserve(nr);
    for (const Band& band : bands) {
        for (int c = 0; c < band.cells; ++c) cell_attribute.push_back(band.attribute);
    }

    std::ofstream m(filename);
    m << "MFEM mesh v1.0\n\n";
    m << "dimension\n2\n\n";

    m << "elements\n" << (2 * nr * nz) << "\n";
    for (int j = 0; j < nz; ++j) {
        for (int i = 0; i < nr; ++i) {
            const int attribute = cell_attribute[i];
            const int v00 = vid(i, j);
            const int v10 = vid(i + 1, j);
            const int v11 = vid(i + 1, j + 1);
            const int v01 = vid(i, j + 1);
            m << attribute << " 2 " << v00 << " " << v10 << " " << v11 << "\n";
            m << attribute << " 2 " << v00 << " " << v11 << " " << v01 << "\n";
        }
    }
    m << "\n";

    std::vector<std::array<int, 3>> bdr; // {attr, va, vb}
    for (int j = 0; j < nz; ++j) {
        bdr.push_back({ 1, vid(0, j),  vid(0, j + 1) });
        bdr.push_back({ 2, vid(nr, j), vid(nr, j + 1) });
    }
    for (int i = 0; i < nr; ++i) {
        bdr.push_back({ 3, vid(i, 0),  vid(i + 1, 0) });
        bdr.push_back({ 3, vid(i, nz), vid(i + 1, nz) });
    }

    m << "boundary\n" << bdr.size() << "\n";
    for (const auto& b : bdr) {
        m << b[0] << " 1 " << b[1] << " " << b[2] << "\n";
    }
    m << "\n";

    m << "vertices\n" << (nvr * nvz) << "\n2\n";
    for (int j = 0; j < nvz; ++j) {
        const double z = height * static_cast<double>(j) / nz;
        for (int i = 0; i < nvr; ++i) {
            const double r = r_min + (r_max - r_min) * static_cast<double>(i) / nr;
            m << r << " " << z << "\n";
        }
    }
    m.close();
}

// Geometry of the axisymmetric current loop used by the inductance checks:
// a square cross-section of side kLoopSide centred at r = kLoopRadius, z = 0,
// surrounded by air out to a far boundary at kLoopDomain.
constexpr double kLoopRadius = 0.1;    // loop radius a [m]
constexpr double kLoopSide = 0.002;    // conductor cross-section side [m]

// Far-field boundary distance. A_phi = 0 is imposed here, which is only exact
// at infinity, so the truncation shows up as a -C/D bias in the inductance.
// D/a = 40 puts that bias near -0.06%, cheaply enough that the tests below can
// assert sub-percent accuracy. See docs/open_boundary.md for the measured
// convergence study behind this choice.
constexpr double kLoopDomain = 4.0;    // far-field boundary distance [m]

// Self-inductance of a circular loop of square cross-section, uniform current.
// The square section enters through its geometric mean distance,
// r_eq = 0.2235 * (w + h), which already carries the internal-inductance term,
// so the classic thin-ring formula is used in its ln(8a/r) - 2 form (Grover).
double AnalyticLoopInductance() {
    const double r_eq = 0.2235 * (kLoopSide + kLoopSide);
    return Constants::MU_0 * kLoopRadius *
        (std::log(8.0 * kLoopRadius / r_eq) - 2.0);
}

// cells+1 points spanning [from, to] whose increments grow by `growth`. Used to
// pack elements against the conductor while still reaching the far boundary
// without an unaffordable number of uniform cells.
std::vector<double> GeometricPoints(double from, double to, int cells, double growth) {
    double sum = 0.0;
    double weight = 1.0;
    for (int i = 0; i < cells; ++i) {
        sum += weight;
        weight *= growth;
    }
    const double base = (to - from) / sum;

    std::vector<double> points;
    points.reserve(static_cast<size_t>(cells) + 1);
    points.push_back(from);
    double x = from;
    double step = base;
    for (int i = 0; i < cells; ++i) {
        x += step;
        step *= growth;
        points.push_back(x);
    }
    points.back() = to;
    return points;
}

// SCOPE: this is a test fixture, not a mesh generator. It exists only to give
// the analytic current-loop tests a mesh with known attributes and no external
// tool dependency. It is a structured tensor grid over a rectangle -- it cannot
// represent curved boundaries, non-rectangular regions, or anything with more
// than two materials, and it is not a step toward in-tree meshing. Real
// geometry is meshed externally (see examples/generate_meshes.sh); see
// docs/mesh_formats.md for the generator evaluation. Do not extend this to
// cover new geometry: add a mesh file instead.
//
// The (r, z) tensor grid underlying the current-loop mesh, plus the
// element/boundary topology derived from it. Shared by the MFEM-format and
// Netgen-format writers below so that both emit the SAME geometry, elements
// and attributes; only the serialization differs. That is what makes the
// cross-format inductance comparison meaningful.
struct CurrentLoopGrid {
    std::vector<double> rs, zs;
    double r_lo, r_hi, z_lo, z_hi;

    int nr() const { return static_cast<int>(rs.size()) - 1; }
    int nz() const { return static_cast<int>(zs.size()) - 1; }
    int nvr() const { return nr() + 1; }
    int vid(int i, int j) const { return j * nvr() + i; }

    // Domain attribute 2 is the conductor, 1 the surrounding air.
    int ElementAttribute(int i, int j) const {
        const double rc = 0.5 * (rs[i] + rs[i + 1]);
        const double zc = 0.5 * (zs[j] + zs[j + 1]);
        const bool in_loop = rc > r_lo && rc < r_hi && zc > z_lo && zc < z_hi;
        return in_loop ? 2 : 1;
    }

    // Boundary attributes: 1 = axis (r=0), 2 = outer (r=kLoopDomain),
    // 3 = top/bottom. Returned as {attr, va, vb} with 0-based vertex ids.
    std::vector<std::array<int, 3>> BoundaryElements() const {
        std::vector<std::array<int, 3>> bdr;
        for (int j = 0; j < nz(); ++j) {
            bdr.push_back({ 1, vid(0, j),      vid(0, j + 1) });
            bdr.push_back({ 2, vid(nr(), j),   vid(nr(), j + 1) });
        }
        for (int i = 0; i < nr(); ++i) {
            bdr.push_back({ 3, vid(i, 0),      vid(i + 1, 0) });
            bdr.push_back({ 3, vid(i, nz()),   vid(i + 1, nz()) });
        }
        return bdr;
    }
};

CurrentLoopGrid BuildCurrentLoopGrid(int loop_cells = 4, int air_cells = 23,
                                     double growth = 1.35) {
    CurrentLoopGrid g;
    g.r_lo = kLoopRadius - 0.5 * kLoopSide;
    g.r_hi = kLoopRadius + 0.5 * kLoopSide;
    g.z_lo = -0.5 * kLoopSide;
    g.z_hi = 0.5 * kLoopSide;

    auto build_axis = [&](double lo, double hi, double far_lo, double far_hi) {
        std::vector<double> inner = GeometricPoints(lo, far_lo, air_cells, growth);
        std::reverse(inner.begin(), inner.end());   // now runs far_lo -> lo
        const std::vector<double> loop = GeometricPoints(lo, hi, loop_cells, 1.0);
        const std::vector<double> outer = GeometricPoints(hi, far_hi, air_cells, growth);

        std::vector<double> coords = inner;
        coords.insert(coords.end(), loop.begin() + 1, loop.end());
        coords.insert(coords.end(), outer.begin() + 1, outer.end());
        return coords;
    };

    g.rs = build_axis(g.r_lo, g.r_hi, 0.0, kLoopDomain);
    g.zs = build_axis(g.z_lo, g.z_hi, -kLoopDomain, kLoopDomain);
    return g;
}

// Build a 2D axisymmetric (r, z) mesh for a single current loop. Domain
// attribute 2 is the conductor, 1 is the surrounding air. Boundary attributes:
// 1 = axis (r=0), 2 = outer (r=kLoopDomain), 3 = top/bottom.
void CreateCurrentLoopMesh(const std::string& filename,
                           int loop_cells = 4, int air_cells = 23,
                           double growth = 1.35) {
    const CurrentLoopGrid g =
        BuildCurrentLoopGrid(loop_cells, air_cells, growth);
    const int nr = g.nr();
    const int nz = g.nz();

    std::ofstream m(filename);
    m << "MFEM mesh v1.0\n\n";
    m << "dimension\n2\n\n";

    m << "elements\n" << (2 * nr * nz) << "\n";
    for (int j = 0; j < nz; ++j) {
        for (int i = 0; i < nr; ++i) {
            const int attribute = g.ElementAttribute(i, j);
            const int v00 = g.vid(i, j);
            const int v10 = g.vid(i + 1, j);
            const int v11 = g.vid(i + 1, j + 1);
            const int v01 = g.vid(i, j + 1);
            m << attribute << " 2 " << v00 << " " << v10 << " " << v11 << "\n";
            m << attribute << " 2 " << v00 << " " << v11 << " " << v01 << "\n";
        }
    }
    m << "\n";

    const std::vector<std::array<int, 3>> bdr = g.BoundaryElements();
    m << "boundary\n" << bdr.size() << "\n";
    for (const auto& b : bdr) {
        m << b[0] << " 1 " << b[1] << " " << b[2] << "\n";
    }
    m << "\n";

    m << "vertices\n" << (g.nvr() * (nz + 1)) << "\n2\n";
    m << std::setprecision(17);
    for (int j = 0; j <= nz; ++j) {
        for (int i = 0; i < g.nvr(); ++i) {
            m << g.rs[i] << " " << g.zs[j] << "\n";
        }
    }
    m.close();
}

// The same current-loop geometry written in Netgen's 2D "areamesh2" format,
// which MFEM dispatches to ReadNetgen2DMesh().
//
// This exists to exercise a mesh format that carries NO named groups at all.
// MFEM only populates attribute_sets for the MFEM v1.3+, Gmsh and Cubit
// readers; the Netgen readers produce bare integer attributes and nothing else.
// A configuration that resolved group names through mesh-provided names would
// therefore silently fail here, so pairing this writer with the analytic
// inductance assertion is a direct test that name -> attribute binding comes
// from the config alone.
//
// Format (all indices 1-based, no header fields beyond the type line):
//   areamesh2
//   <n_bdr>            then n_bdr lines of: attr v1 v2
//   <n_elem>           then n_elem lines of: attr nverts v1 ... vn
//   <n_vert>           then n_vert lines of: r z
void CreateCurrentLoopNetgenMesh(const std::string& filename,
                                 int loop_cells = 4, int air_cells = 23,
                                 double growth = 1.35) {
    const CurrentLoopGrid g =
        BuildCurrentLoopGrid(loop_cells, air_cells, growth);
    const int nr = g.nr();
    const int nz = g.nz();

    std::ofstream m(filename);
    m << "areamesh2\n\n";

    // Boundary elements come FIRST in this format, unlike the MFEM format.
    const std::vector<std::array<int, 3>> bdr = g.BoundaryElements();
    m << bdr.size() << "\n";
    for (const auto& b : bdr) {
        m << b[0] << " " << (b[1] + 1) << " " << (b[2] + 1) << "\n";
    }
    m << "\n";

    // Triangles, split exactly as the MFEM writer does.
    m << (2 * nr * nz) << "\n";
    for (int j = 0; j < nz; ++j) {
        for (int i = 0; i < nr; ++i) {
            const int attribute = g.ElementAttribute(i, j);
            const int v00 = g.vid(i, j) + 1;
            const int v10 = g.vid(i + 1, j) + 1;
            const int v11 = g.vid(i + 1, j + 1) + 1;
            const int v01 = g.vid(i, j + 1) + 1;
            m << attribute << " 3 " << v00 << " " << v10 << " " << v11 << "\n";
            m << attribute << " 3 " << v00 << " " << v11 << " " << v01 << "\n";
        }
    }
    m << "\n";

    m << (g.nvr() * (nz + 1)) << "\n";
    m << std::setprecision(17);
    for (int j = 0; j <= nz; ++j) {
        for (int i = 0; i < g.nvr(); ++i) {
            m << g.rs[i] << " " << g.zs[j] << "\n";
        }
    }
    m.close();
}

// Coupling-matrix config for the current-loop mesh. The far boundary and the
// top/bottom planes hold A_phi = 0; the axis condition is applied by the
// axisymmetric solvers themselves.
json MakeCurrentLoopConfig(const std::string& physics,
                           const std::string& mesh_file,
                           double sigma) {
    return json{
        {"simulation", {
            {"physics_type", physics},
            {"mesh", mesh_file},
            {"order", 2},
            {"geometry_type", "axisymmetric"},
            {"analysis_type", "coupling_matrix"},
            {"solver_tolerance", 1e-14},
            {"solver_max_iter", 8000},
            {"solver_print_level", 0}
        }},
        {"entity_groups", json::array({
            {{"name", "AirDomain"},  {"dim", 2}, {"attribute_ids", {1}}},
            {{"name", "LoopDomain"}, {"dim", 2}, {"attribute_ids", {2}}},
            {{"name", "Outer"},      {"dim", 1}, {"attribute_ids", {2, 3}}}
        })},
        {"regions", json::array({
            {{"name", "Air"},  {"entity_group", "AirDomain"},  {"material", "Air"}},
            {{"name", "Loop"}, {"entity_group", "LoopDomain"}, {"material", "Conductor"}}
        })},
        {"materials", json::array({
            {{"name", "Air"},       {"properties", {{"mu_r", 1.0}, {"sigma", 0.0}}}},
            {{"name", "Conductor"}, {"properties", {{"mu_r", 1.0}, {"sigma", sigma}}}}
        })},
        {"boundary_conditions", json::array({
            {{"name", "Outer"}, {"type", "dirichlet"},
             {"entity_group", "Outer"}, {"value", 0.0}}
        })},
        {"scenarios", json::array({
            {{"name", "loop"}, {"excitations", json::array()}}
        })}
    };
}

// Coupling-matrix config for the shield-and-turns mesh at a single frequency
// point. Region index 1 is the shield, so callers switch its material or add a
// current constraint to select between the shielding cases under test.
json MakeShieldedTurnsConfig(const std::string& mesh_file, double frequency) {
    return json{
        {"simulation", {
            {"physics_type", "magnetoquasistatics"},
            {"mesh", mesh_file},
            {"order", 1},
            {"geometry_type", "axisymmetric"},
            {"analysis_type", "coupling_matrix"},
            {"solver_tolerance", 1e-12},
            {"solver_max_iter", 4000},
            {"solver_print_level", 0}
        }},
        {"entity_groups", json::array({
            {{"name", "AirDomain"},    {"dim", 2}, {"attribute_ids", {1}}},
            {{"name", "ShieldDomain"}, {"dim", 2}, {"attribute_ids", {2}}},
            {{"name", "TurnADomain"},  {"dim", 2}, {"attribute_ids", {3}}},
            {{"name", "TurnBDomain"},  {"dim", 2}, {"attribute_ids", {4}}},
            {{"name", "Outer"},        {"dim", 1}, {"attribute_ids", {2}}}
        })},
        {"regions", json::array({
            {{"name", "Air"},    {"entity_group", "AirDomain"},    {"material", "Air"}},
            {{"name", "Shield"}, {"entity_group", "ShieldDomain"}, {"material", "Aluminum"}},
            {{"name", "TurnA"},  {"entity_group", "TurnADomain"},  {"material", "Copper"}},
            {{"name", "TurnB"},  {"entity_group", "TurnBDomain"},  {"material", "Copper"}}
        })},
        {"materials", json::array({
            {{"name", "Air"},      {"properties", {{"mu_r", 1.0}, {"sigma", 0.0}}}},
            {{"name", "Aluminum"}, {"properties", {{"mu_r", 1.0}, {"sigma", 3.5e7}}}},
            {{"name", "Copper"},   {"properties", {{"mu_r", 1.0}, {"sigma", 5.8e7}}}}
        })},
        {"boundary_conditions", json::array({
            {{"name", "Outer"}, {"type", "dirichlet"}, {"entity_group", "Outer"}, {"value", 0.0}}
        })},
        {"terminals", json::array({
            {{"name", "TurnA"}, {"quantity", "current"},
             {"conductor_type", "massive"}, {"entity_group", "TurnADomain"}},
            {{"name", "TurnB"}, {"quantity", "current"},
             {"conductor_type", "massive"}, {"entity_group", "TurnBDomain"}}
        })},
        {"scenarios", json::array({
            {{"name", "point"},
             {"frequency", {{"scale", "linear"}, {"start", frequency},
                            {"stop", frequency}, {"points", 1}}},
             {"excitations", json::array()}}
        })}
    };
}

// Common AMR-enabled coaxial-capacitor config (Field analysis, one energized
// scenario). Outputs are off by default; callers enable Gmsh output as needed.
json MakeCoaxAmrConfig(const std::string& mesh_file, int max_iterations) {
    return json{
        {"simulation", {
            {"physics_type", "electrostatics"},
            {"mesh", mesh_file},
            {"order", 1},
            {"geometry_type", "axisymmetric"},
            {"analysis_type", "field"},
            {"amr", {
                {"enabled", true},
                {"max_iterations", max_iterations},
                {"error_fraction", 0.7},
                {"max_dofs", 0},          // disabled; iteration cap governs
                {"error_tolerance", 0.0}  // disabled; iteration cap governs
            }}
        }},
        {"entity_groups", json::array({
            {{"name", "Dielectric"}, {"dim", 2}, {"attribute_ids", {1}}},
            {{"name", "Inner"},      {"dim", 1}, {"attribute_ids", {1}}},
            {{"name", "Outer"},      {"dim", 1}, {"attribute_ids", {2}}}
        })},
        {"regions", json::array({
            {{"name", "Dielectric"}, {"entity_group", "Dielectric"}, {"material", "Vacuum"}}
        })},
        {"materials", json::array({
            {{"name", "Vacuum"}, {"properties", {{"epsilon_r", 1.0}}}}
        })},
        {"terminals", json::array({
            {{"name", "Inner"}, {"quantity", "voltage"}, {"entity_group", "Inner"}},
            {{"name", "Outer"}, {"quantity", "voltage"}, {"entity_group", "Outer"}}
        })},
        {"scenarios", json::array({
            {{"name", "energized"}, {"excitations", json::array({
                {{"terminal", "Inner"}, {"value", 1.0}},
                {{"terminal", "Outer"}, {"value", 0.0}}
            })}}
        })}
    };
}

mfem::Vector AxisymmetricDiffusionErrors(
    mfem::Mesh& mesh, mfem::GridFunction& solution,
    mfem::Coefficient& coefficient, bool with_coefficient) {
    mfem::H1_FECollection flux_fec(1, mesh.Dimension());
    mfem::FiniteElementSpace flux_fes(
        &mesh, &flux_fec, mesh.SpaceDimension());
    AxisymmetricDiffusionIntegrator integrator(coefficient);
    mfem::ZienkiewiczZhuEstimator estimator(
        integrator, solution, flux_fes);
    estimator.SetWithCoeff(with_coefficient);
    estimator.SetFluxAveraging(1);
    return estimator.GetLocalErrors();
}

mfem::Vector AxisymmetricCurlCurlErrors(
    mfem::Mesh& mesh, mfem::GridFunction& solution,
    mfem::Coefficient& coefficient, bool with_coefficient) {
    mfem::H1_FECollection flux_fec(1, mesh.Dimension());
    mfem::FiniteElementSpace flux_fes(
        &mesh, &flux_fec, mesh.SpaceDimension());
    AxisymmetricCurlCurlIntegrator integrator(
        coefficient, axisym::InspectAxisGeometry(mesh).tolerance);
    mfem::ZienkiewiczZhuEstimator estimator(
        integrator, solution, flux_fes);
    estimator.SetWithCoeff(with_coefficient);
    estimator.SetFluxAveraging(1);
    return estimator.GetLocalErrors();
}

// Normalize a copy of @p v by its l2 norm, so two indicator vectors can be
// compared for SHAPE (the relative distribution of error across elements)
// independently of any uniform scale factor.
mfem::Vector NormalizedErrors(const mfem::Vector& v) {
    mfem::Vector out(v);
    const double norm = out.Norml2();
    if (norm > 0.0) { out /= norm; }
    return out;
}

// Compare error indicators up to a uniform positive scale.
//
// The solvers divide their raw ZZ indicator by sqrt(field energy) to produce a
// dimensionless relative error (see EstimateCurrentSolutionError), so solver
// output is a scalar multiple of the raw estimator output these tests build by
// hand. That scale factor is deliberate and not what these cases are checking:
// they verify WHICH elements carry the error and in what proportion, i.e. the
// property that actually drives Dorfler marking. Comparing normalized vectors
// keeps the assertions on that invariant.
void RequireErrorsEqual(const mfem::Vector& actual,
                        const mfem::Vector& expected) {
    REQUIRE(actual.Size() == expected.Size());
    const mfem::Vector a = NormalizedErrors(actual);
    const mfem::Vector e = NormalizedErrors(expected);
    for (int i = 0; i < a.Size(); ++i) {
        REQUIRE(a(i) == Catch::Approx(e(i)).epsilon(1e-12));
    }
}

bool ErrorsDiffer(const mfem::Vector& first, const mfem::Vector& second) {
    if (first.Size() != second.Size()) return true;
    // Shape comparison, to match RequireErrorsEqual: a pure rescaling must not
    // register as a difference, otherwise the negative assertions in these tests
    // would pass for the trivial reason that the solver normalizes.
    const mfem::Vector a = NormalizedErrors(first);
    const mfem::Vector b = NormalizedErrors(second);
    for (int i = 0; i < a.Size(); ++i) {
        const double scale = std::max(std::abs(a(i)), std::abs(b(i)));
        if (std::abs(a(i) - b(i)) > 1e-8 * scale) return true;
    }
    return false;
}

// Extract the raw text of a named MSH section, e.g. "$Nodes" ... "$EndNodes".
// Returns the lines strictly between the markers (exclusive), joined by '\n'.
std::string ExtractMshSection(const std::string& path, const std::string& tag) {
    std::ifstream f(path);
    REQUIRE(f.is_open());
    const std::string begin = "$" + tag;
    const std::string end = "$End" + tag;
    std::string line;
    bool in = false;
    std::ostringstream out;
    while (std::getline(f, line)) {
        if (!in) {
            if (line.rfind(begin, 0) == 0) { in = true; }
        } else {
            if (line.rfind(end, 0) == 0) { break; }
            out << line << '\n';
        }
    }
    return out.str();
}

} // namespace

TEST_CASE("Electrostatic solver reproduces a uniform field between plates",
          "[solvers][analytic][electrostatic]") {
    const std::string mesh_file = "test_analytic_electrostatic.mesh";
    constexpr double length = 0.2;
    constexpr double height = 0.05;
    constexpr double voltage = 100.0;
    constexpr int nx = 8;
    constexpr int ny = 2;
    CreatePlanarStripMesh(mesh_file, length, height, nx, ny);

    json config = MakePlanarStripConfig(
        "electrostatics", mesh_file, 1, {{"epsilon_r", 2.5}}, voltage, 0.0);
    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    ElectrostaticSolver solver(mesh, DecodeConfig(config));
    solver.Setup();
    solver.Run();

    FieldExportSet fields = solver.CollectExportFields();
    const mfem::IntegrationPoint point = TriangleCenter();
    const int element = 2 * (nx / 2);
    const FieldExport& potential_field = FindField(fields, "V");
    const mfem::Vector physical = PhysicalPoint(*potential_field.primary, element, point);
    const double expected_potential = voltage * (1.0 - physical(0) / length);
    const double expected_field = voltage / length;

    const double potential = SamplePrimaryScalar(fields, "V", element, point);
    const mfem::Vector electric_field = SampleDerivedVector(fields, "E", element, point);

    REQUIRE(potential == Catch::Approx(expected_potential).epsilon(1e-7));
    REQUIRE(electric_field(0) == Catch::Approx(expected_field).epsilon(1e-6));
    REQUIRE(electric_field(1) == Catch::Approx(0.0).margin(1e-4));

    fs::remove(mesh_file);
}

TEST_CASE("Capacitor example drives the dielectric field", "[solvers][electrostatic][examples]") {
    const fs::path path = fs::path(__FILE__).parent_path().parent_path()
        / "examples/simple_capacitor/config.json";
    ProblemConfig config = InputParser(path.string()).GetProblemConfig();
    config.Output = {};
    config.Scenarios.resize(1);
    auto mesh = mesh_io::LoadMesh(config.MeshPath);
    ElectrostaticSolver solver(*mesh, config);
    solver.Setup();
    solver.Run();
    const auto fields = solver.CollectExportFields();
    int sampled = 0;
    for (int element = 0; element < mesh->GetNE(); ++element) {
        if (mesh->GetAttribute(element) != 2) continue;
        mfem::Vector center;
        mesh->GetElementCenter(element, center);
        if (center[0] < 0.03 || center[0] > 0.07 || center[1] < 0.004 || center[1] > 0.008) continue;
        const auto& point = mfem::Geometries.GetCenter(mesh->GetElementBaseGeometry(element));
        const auto electric = SampleDerivedVector(fields, "E", element, point);
        REQUIRE(electric[1] == Catch::Approx(-100000.0).epsilon(0.02));
        REQUIRE(std::abs(electric[0]) < 2000.0);
        ++sampled;
    }
    REQUIRE(sampled > 0);
}

TEST_CASE("Electrostatic field energy gives analytic capacitance and conserves flux",
          "[solvers][analytic][electrostatic][energy][conservation]") {
    const std::string mesh_file = "test_electrostatic_energy.mesh";
    constexpr double length = 0.2;
    constexpr double height = 0.05;
    constexpr double voltage = 100.0;
    constexpr double relative_permittivity = 2.5;
    constexpr int nx = 8;
    constexpr int ny = 3;
    CreatePlanarStripMesh(mesh_file, length, height, nx, ny);

    json config = MakePlanarStripConfig(
        "electrostatics", mesh_file, 1,
        {{"epsilon_r", relative_permittivity}}, voltage, 0.0);
    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    ElectrostaticSolver solver(mesh, DecodeConfig(config));
    solver.Setup();
    solver.Run();

    FieldExportSet fields = solver.CollectExportFields();
    const double permittivity = Constants::EPSILON_0 * relative_permittivity;
    const double field_integral = IntegrateVectorMagnitudeSquared(fields, "E");
    const double electric_energy = 0.5 * permittivity * field_integral;
    const double computed_capacitance = 2.0 * electric_energy / (voltage * voltage);
    const double analytic_capacitance = permittivity * height / length;

    REQUIRE(electric_energy == Catch::Approx(
        0.5 * analytic_capacitance * voltage * voltage).epsilon(1e-8));
    REQUIRE(computed_capacitance ==
        Catch::Approx(analytic_capacitance).epsilon(1e-8));

    const mfem::IntegrationPoint center = TriangleCenter();
    const double boundary_segment_length = height / ny;
    double outward_flux = 0.0;
    double absolute_flux = 0.0;
    for (int j = 0; j < ny; ++j) {
        const int left_element = 2 * (j * nx) + 1;
        const int right_element = 2 * (j * nx + nx - 1);
        const mfem::Vector left_field =
            SampleDerivedVector(fields, "E", left_element, center);
        const mfem::Vector right_field =
            SampleDerivedVector(fields, "E", right_element, center);
        const double left_flux = -permittivity * left_field(0) * boundary_segment_length;
        const double right_flux = permittivity * right_field(0) * boundary_segment_length;
        outward_flux += left_flux + right_flux;
        absolute_flux += std::abs(left_flux) + std::abs(right_flux);
    }
    REQUIRE(std::abs(outward_flux) < 1e-7 * absolute_flux);

    fs::remove(mesh_file);
}

TEST_CASE("Electrostatic solver satisfies dielectric interface conditions",
          "[solvers][analytic][electrostatic][materials][interface]") {
    const std::string mesh_file = "test_dielectric_interface.mesh";
    constexpr double length = 0.2;
    constexpr double height = 0.05;
    constexpr double voltage = 10.0;
    constexpr double epsilon_r_left = 2.0;
    constexpr double epsilon_r_right = 5.0;
    constexpr int nx = 8;
    constexpr int ny = 2;
    constexpr int interface_column = nx / 2;
    CreateLayeredStripMesh(
        mesh_file, length, height, nx, ny, interface_column);

    json config = {
        {"simulation", {
            {"physics_type", "electrostatics"},
            {"mesh", mesh_file},
            {"order", 1},
            {"geometry_type", "planar"},
            {"analysis_type", "field"},
            {"solver_tolerance", 1e-12},
            {"solver_max_iter", 4000},
            {"solver_print_level", 0}
        }},
        {"entity_groups", json::array({
            {{"name", "LeftDielectric"},  {"dim", 2}, {"attribute_ids", {1}}},
            {{"name", "RightDielectric"}, {"dim", 2}, {"attribute_ids", {2}}},
            {{"name", "Left"},  {"dim", 1}, {"attribute_ids", {1}}},
            {{"name", "Right"}, {"dim", 1}, {"attribute_ids", {2}}}
        })},
        {"regions", json::array({
            {{"name", "LeftDielectric"}, {"entity_group", "LeftDielectric"}, {"material", "LowPermittivity"}},
            {{"name", "RightDielectric"}, {"entity_group", "RightDielectric"}, {"material", "HighPermittivity"}}
        })},
        {"materials", json::array({
            {{"name", "LowPermittivity"}, {"properties", {{"epsilon_r", epsilon_r_left}}}},
            {{"name", "HighPermittivity"}, {"properties", {{"epsilon_r", epsilon_r_right}}}}
        })},
        {"boundary_conditions", json::array({
            {{"name", "Left"},  {"type", "dirichlet"}, {"entity_group", "Left"},  {"value", voltage}},
            {{"name", "Right"}, {"type", "dirichlet"}, {"entity_group", "Right"}, {"value", 0.0}}
        })},
        {"scenarios", json::array({
            {{"name", "interface"}, {"excitations", json::array()}}
        })}
    };

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    ElectrostaticSolver solver(mesh, DecodeConfig(config));
    solver.Setup();
    solver.Run();

    FieldExportSet fields = solver.CollectExportFields();
    const mfem::IntegrationPoint center = TriangleCenter();
    const int left_element = 2 * (interface_column - 1);
    const int right_element = 2 * interface_column;
    const mfem::Vector left_field =
        SampleDerivedVector(fields, "E", left_element, center);
    const mfem::Vector right_field =
        SampleDerivedVector(fields, "E", right_element, center);
    const double epsilon_left = Constants::EPSILON_0 * epsilon_r_left;
    const double epsilon_right = Constants::EPSILON_0 * epsilon_r_right;
    const double left_width = length * interface_column / nx;
    const double right_width = length - left_width;
    const double analytic_displacement =
        voltage / (left_width / epsilon_left + right_width / epsilon_right);

    REQUIRE(epsilon_left * left_field(0) ==
        Catch::Approx(analytic_displacement).epsilon(1e-6));
    REQUIRE(epsilon_right * right_field(0) ==
        Catch::Approx(analytic_displacement).epsilon(1e-6));
    REQUIRE(std::abs(left_field(1)) < 1e-6 * std::abs(left_field(0)));
    REQUIRE(std::abs(right_field(1)) < 1e-6 * std::abs(right_field(0)));

    fs::remove(mesh_file);
}

TEST_CASE("Electrostatic capacitance matrix is analytic and reciprocal",
          "[solvers][analytic][electrostatic][coupling][reciprocity]") {
    const std::string mesh_file = "test_capacitance_reciprocity.mesh";
    const std::string results_directory = "test_capacitance_hdf5_results";
    const std::string matrix_file = results_directory + "/nested/coupling_electrostatics.h5";
    constexpr double length = 0.2;
    constexpr double height = 0.05;
    constexpr double relative_permittivity = 2.5;
    CreatePlanarStripMesh(mesh_file, length, height, 8, 2);

    json config = MakePlanarStripConfig(
        "electrostatics", mesh_file, 1,
        {{"epsilon_r", relative_permittivity}}, 0.0, 0.0);
    config["simulation"]["analysis_type"] = "coupling_matrix";
    config["output"] = {{"directory", results_directory + "/nested"},
        {"hdf5", {{"file", "coupling_electrostatics.h5"}}}};
    config["boundary_conditions"] = json::array();
    config["terminals"] = json::array({
        {{"name", "Left"}, {"quantity", "voltage"}, {"entity_group", "Left"}},
        {{"name", "Right"}, {"quantity", "voltage"}, {"entity_group", "Right"}}
    });

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    ElectrostaticSolver solver(mesh, DecodeConfig(config, matrix_file));
    solver.Setup();
    solver.Run();
    solver.SaveAnalysis();

    const auto matrix = ReadHdf5Matrix(matrix_file, "Capacitance");
    REQUIRE(matrix.labels == std::vector<std::string>{"Left", "Right"});
    const double analytic_capacitance =
        Constants::EPSILON_0 * relative_permittivity * height / length;
    REQUIRE(matrix.values[0][1] == Catch::Approx(matrix.values[1][0]).epsilon(1e-7));
    REQUIRE(matrix.values[0][0] == Catch::Approx(analytic_capacitance).epsilon(1e-8));
    REQUIRE(matrix.values[1][1] == Catch::Approx(analytic_capacitance).epsilon(1e-8));
    REQUIRE(matrix.values[0][1] == Catch::Approx(-analytic_capacitance).epsilon(1e-8));
    REQUIRE(matrix.values[1][0] == Catch::Approx(-analytic_capacitance).epsilon(1e-8));
    REQUIRE(matrix.values[0][0] + matrix.values[0][1] ==
        Catch::Approx(0.0).margin(1e-7 * analytic_capacitance));
    REQUIRE(matrix.values[1][0] + matrix.values[1][1] ==
        Catch::Approx(0.0).margin(1e-7 * analytic_capacitance));

    fs::remove(matrix_file);
    fs::remove_all(results_directory);
    fs::remove(mesh_file);
}

// A planar model is translationally invariant out of plane and assembles over
// the cross-section alone, i.e. a unit depth, so every extracted coupling
// quantity is per unit length. Axisymmetric assembly carries the full 2*pi*r
// measure and is absolute. The written labels must say which convention
// produced the numbers, otherwise an F/m result reads as farads.
TEST_CASE("Coupling matrix units distinguish planar from axisymmetric",
          "[solvers][units][m6]") {
    struct LabelProbe : PhysicsSolver {
        using PhysicsSolver::PhysicsSolver;
        using PhysicsSolver::CouplingUnitLabel;
        using PhysicsSolver::geometry;
        void Setup() override {}
        void SaveAnalysisResults() override {}
        FieldExportSet CollectExportFields() const override { return {}; }
        void BuildOperators() override {}
        void RunOnCurrentMesh() override {}
        void EstimateCurrentSolutionError(mfem::Vector&) override {}
        double ComputePeakFieldMagnitude() const override { return 0.0; }
    };

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
        1, 1, mfem::Element::QUADRILATERAL);
    ProblemConfig config;
    LabelProbe probe(mesh, config);

    SECTION("planar results are reported per unit length") {
        probe.geometry = GeometryType::Planar;
        REQUIRE(probe.CouplingUnitLabel("F") == "[F/m]");
        REQUIRE(probe.CouplingUnitLabel("H") == "[H/m]");
        REQUIRE(probe.CouplingUnitLabel("Ohm") == "[Ohm/m]");
    }

    SECTION("axisymmetric results are absolute") {
        probe.geometry = GeometryType::Axisymmetric;
        REQUIRE(probe.CouplingUnitLabel("F") == "[F]");
        REQUIRE(probe.CouplingUnitLabel("H") == "[H]");
        REQUIRE(probe.CouplingUnitLabel("Ohm") == "[Ohm]");
    }

    SECTION("3D results are absolute") {
        probe.geometry = GeometryType::Cartesian3D;
        REQUIRE(probe.CouplingUnitLabel("F") == "[F]");
        REQUIRE(probe.CouplingUnitLabel("H") == "[H]");
        REQUIRE(probe.CouplingUnitLabel("Ohm") == "[Ohm]");
    }
}

namespace {
// A 3D tetrahedral box [0,lx] x [0,ly] x [0,d] saved as an MFEM mesh. MFEM's
// Cartesian generator labels the boundary faces 1 = bottom (z = 0) and
// 6 = top (z = d); the four sides (2-5) are left natural.
void CreateBoxTetMesh(const std::string& filename,
                      double lx, double ly, double d, int nx, int ny, int nz) {
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
        nx, ny, nz, mfem::Element::TETRAHEDRON, lx, ly, d);
    std::ofstream out(filename);
    out.precision(17);
    mesh.Print(out);
}

json MakeBoxCapacitorConfig(const std::string& mesh_file, double epsilon_r) {
    return json{
        {"simulation", {
            {"physics_type", "electrostatics"},
            {"mesh", mesh_file},
            {"order", 1},
            {"geometry_type", "3d"},
            {"analysis_type", "coupling_matrix"},
            {"solver_tolerance", 1e-12},
            {"solver_max_iter", 4000},
            {"solver_print_level", 0}
        }},
        {"entity_groups", json::array({
            {{"name", "Domain"}, {"dim", 3}, {"attribute_ids", {1}}},
            {{"name", "Bottom"}, {"dim", 2}, {"attribute_ids", {1}}},
            {{"name", "Top"},    {"dim", 2}, {"attribute_ids", {6}}}
        })},
        {"regions", json::array({
            {{"name", "Domain"}, {"entity_group", "Domain"}, {"material", "Dielectric"}}
        })},
        {"materials", json::array({
            {{"name", "Dielectric"}, {"properties", {{"epsilon_r", epsilon_r}}}}
        })},
        {"terminals", json::array({
            {{"name", "Bottom"}, {"quantity", "voltage"}, {"entity_group", "Bottom"}},
            {{"name", "Top"},    {"quantity", "voltage"}, {"entity_group", "Top"}}
        })},
        {"boundary_conditions", json::array()},
        {"scenarios", json::array()}
    };
}
} // namespace

// First 3D end-to-end check. Between two parallel plates with natural (zero
// normal flux) side walls the exact potential is linear in z, which P1
// tetrahedra represent exactly, so the extracted capacitance must equal
// eps*A/d to solver precision -- and be labeled in farads, not F/m.
TEST_CASE("3D parallel-plate capacitance is exact and absolute",
          "[solvers][analytic][electrostatic][coupling][3d]") {
    const std::string mesh_file = "test_3d_box_capacitor.mesh";
    const std::string matrix_file = "coupling_electrostatics_3d.h5";
    constexpr double lx = 0.2, ly = 0.1, d = 0.05, eps_r = 3.0;
    CreateBoxTetMesh(mesh_file, lx, ly, d, 2, 2, 2);

    const auto mesh = mesh_io::LoadMesh(mesh_file);
    REQUIRE(mesh->Dimension() == 3);
    ElectrostaticSolver solver(*mesh, DecodeConfig(
        MakeBoxCapacitorConfig(mesh_file, eps_r), matrix_file));
    solver.Setup();
    solver.Run();
    solver.SaveAnalysis();

    const auto matrix = ReadHdf5Matrix(matrix_file, "Capacitance");
    REQUIRE(matrix.labels == std::vector<std::string>{"Bottom", "Top"});
    const double analytic = Constants::EPSILON_0 * eps_r * lx * ly / d;
    REQUIRE(matrix.values[0][0] == Catch::Approx(analytic).epsilon(1e-8));
    REQUIRE(matrix.values[1][1] == Catch::Approx(analytic).epsilon(1e-8));
    REQUIRE(matrix.values[0][1] == Catch::Approx(-analytic).epsilon(1e-8));
    REQUIRE(matrix.values[1][0] == Catch::Approx(-analytic).epsilon(1e-8));

    HighFive::File file(matrix_file, HighFive::File::ReadOnly);
    std::string geometry, units;
    file.getGroup("/coupling").getAttribute("geometry_type").read(geometry);
    file.getDataSet("/coupling/Capacitance/values").getAttribute("units").read(units);
    REQUIRE(geometry == "3d");
    REQUIRE(units == "F");

    fs::remove(matrix_file);
    fs::remove(mesh_file);
}

namespace {
// Spherical shell a <= |x| <= b as a curved (order-2 geometry) hexahedral
// "cubed sphere": six patches of n x n equiangular cells, nr radial layers.
// Boundary attribute 1 is the inner sphere, 2 the outer one.
//
// The mesh is first built with straight edges, then given quadratic geometry
// whose nodes are pushed radially onto their intended sphere. A node's
// intended radius is recovered by snapping |x| to the nearest radial lattice
// value (layer radii and layer midpoints): the chord sag is far smaller than
// half a layer, so the snap is unambiguous.
void CreateSphericalShellMesh(const std::string& filename, double a, double b,
                              int n, int nr) {
    std::map<std::array<long long, 3>, int> index;
    std::vector<std::array<double, 3>> verts;
    auto vertex = [&](const std::array<double, 3>& x) {
        std::array<long long, 3> key;
        for (int c = 0; c < 3; ++c) { key[c] = std::llround(x[c] * 1e9); }
        const auto it = index.find(key);
        if (it != index.end()) { return it->second; }
        verts.push_back(x);
        return index[key] = static_cast<int>(verts.size()) - 1;
    };
    auto point = [&](int axis, int sign, int i, int j, int k) {
        const double pi4 = std::atan(1.0);
        const double u = std::tan(pi4 * (-1.0 + 2.0 * i / n));
        const double v = std::tan(pi4 * (-1.0 + 2.0 * j / n));
        std::array<double, 3> d{};
        d[axis] = sign;
        d[(axis + 1) % 3] = u;
        d[(axis + 2) % 3] = v;
        const double len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        const double r = a + (b - a) * k / nr;
        for (auto& c : d) { c *= r / len; }
        return vertex(d);
    };

    struct Hex { int v[8]; };
    struct Quad { int v[4]; int attr; };
    std::vector<Hex> hexes;
    std::vector<Quad> quads;
    for (int axis = 0; axis < 3; ++axis) {
        for (int sign : { -1, 1 }) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    for (int k = 0; k < nr; ++k) {
                        hexes.push_back({ {
                            point(axis, sign, i, j, k), point(axis, sign, i + 1, j, k),
                            point(axis, sign, i + 1, j + 1, k), point(axis, sign, i, j + 1, k),
                            point(axis, sign, i, j, k + 1), point(axis, sign, i + 1, j, k + 1),
                            point(axis, sign, i + 1, j + 1, k + 1), point(axis, sign, i, j + 1, k + 1) } });
                    }
                    quads.push_back({ { point(axis, sign, i, j, 0), point(axis, sign, i + 1, j, 0),
                                        point(axis, sign, i + 1, j + 1, 0), point(axis, sign, i, j + 1, 0) }, 1 });
                    quads.push_back({ { point(axis, sign, i, j, nr), point(axis, sign, i + 1, j, nr),
                                        point(axis, sign, i + 1, j + 1, nr), point(axis, sign, i, j + 1, nr) }, 2 });
                }
            }
        }
    }

    // Half the patches come out left-handed; MFEM cannot reorient hexahedra,
    // so mirror those here (swap the 1<->3 and 5<->7 corners).
    for (auto& h : hexes) {
        const auto& o = verts[h.v[0]];
        std::array<double, 3> e[3];
        const int corner[3] = { h.v[1], h.v[3], h.v[4] };
        for (int d = 0; d < 3; ++d) {
            for (int c = 0; c < 3; ++c) { e[d][c] = verts[corner[d]][c] - o[c]; }
        }
        const double det =
            e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
            e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
            e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
        if (det < 0.0) {
            std::swap(h.v[1], h.v[3]);
            std::swap(h.v[5], h.v[7]);
        }
    }

    mfem::Mesh mesh(3, static_cast<int>(verts.size()), static_cast<int>(hexes.size()),
                    static_cast<int>(quads.size()), 3);
    for (const auto& x : verts) { mesh.AddVertex(x[0], x[1], x[2]); }
    for (const auto& h : hexes) { mesh.AddHex(h.v, 1); }
    for (const auto& q : quads) { mesh.AddBdrQuad(q.v, q.attr); }
    mesh.FinalizeHexMesh(/*generate_edges=*/1, /*refine=*/0, /*fix_orientation=*/true);

    mesh.SetCurvature(2);
    const double half_layer = 0.5 * (b - a) / nr;
    mesh.Transform([&](const mfem::Vector& x, mfem::Vector& y) {
        const double r = x.Norml2();
        const double snapped = a + half_layer * std::round((r - a) / half_layer);
        y = x;
        y *= snapped / r;
    });

    std::ofstream out(filename);
    out.precision(17);
    mesh.Print(out);
}

json MakeSphereConfig(const std::string& mesh_file, const json& outer_bc,
                      bool outer_terminal) {
    json terminals = json::array({
        {{"name", "Inner"}, {"quantity", "voltage"}, {"entity_group", "Inner"}}});
    if (outer_terminal) {
        terminals.push_back(
            {{"name", "Outer"}, {"quantity", "voltage"}, {"entity_group", "Outer"}});
    }
    return json{
        {"simulation", {
            {"physics_type", "electrostatics"},
            {"mesh", mesh_file},
            {"order", 2},
            {"geometry_type", "3d"},
            {"analysis_type", "coupling_matrix"},
            {"solver_tolerance", 1e-12},
            {"solver_max_iter", 4000},
            {"solver_print_level", 0}
        }},
        {"entity_groups", json::array({
            {{"name", "Domain"}, {"dim", 3}, {"attribute_ids", {1}}},
            {{"name", "Inner"},  {"dim", 2}, {"attribute_ids", {1}}},
            {{"name", "Outer"},  {"dim", 2}, {"attribute_ids", {2}}}
        })},
        {"regions", json::array({
            {{"name", "Domain"}, {"entity_group", "Domain"}, {"material", "Vacuum"}}
        })},
        {"materials", json::array({
            {{"name", "Vacuum"}, {"properties", {{"epsilon_r", 1.0}}}}
        })},
        {"terminals", terminals},
        {"boundary_conditions", outer_bc.is_null() ? json::array() : json::array({outer_bc})},
        {"scenarios", json::array()}
    };
}

double SolveSphereCapacitance(const std::string& mesh_file, const json& config,
                              const std::string& matrix_file) {
    const auto mesh = mesh_io::LoadMesh(mesh_file);
    ElectrostaticSolver solver(*mesh, DecodeConfig(config, matrix_file));
    solver.Setup();
    solver.Run();
    solver.SaveAnalysis();
    const double c = ReadHdf5Matrix(matrix_file, "Capacitance").values[0][0];
    fs::remove(matrix_file);
    return c;
}
} // namespace

// Concentric spheres held at 1 V and 0 V: C = 4*pi*eps0 * a*b / (b - a). Exercises
// curved 3D geometry, the 3D measure and absolute units on a non-trivial field.
TEST_CASE("3D concentric-sphere capacitance matches the analytic value",
          "[solvers][analytic][electrostatic][coupling][3d]") {
    const std::string mesh_file = "test_3d_sphere_shell.mesh";
    const std::string matrix_file = "coupling_sphere_shell.h5";
    constexpr double a = 0.1, b = 0.3;
    CreateSphericalShellMesh(mesh_file, a, b, 6, 6);

    json config = MakeSphereConfig(mesh_file, nullptr, /*outer_terminal=*/true);
    config["simulation"]["linear_solver"] = "direct";
    const double analytic = 2.0 * Constants::TWO_PI * Constants::EPSILON_0 * a * b / (b - a);
    REQUIRE(SolveSphereCapacitance(mesh_file, config, matrix_file) ==
            Catch::Approx(analytic).epsilon(2e-3));

    SECTION("the default ('3d' -> multigrid PCG) agrees with the direct solve") {
        json iterative = config;
        iterative["simulation"].erase("linear_solver");
        REQUIRE(SolveSphereCapacitance(mesh_file, iterative, matrix_file) ==
                Catch::Approx(SolveSphereCapacitance(mesh_file, config, matrix_file))
                    .epsilon(1e-8));
    }
    fs::remove(mesh_file);
}

// An isolated sphere of radius a in free space has C = 4*pi*eps0*a. Its exact
// potential V = a/r satisfies eps dV/dn + (eps/R) V = 0 on any sphere r = R,
// so a Robin far-field boundary with coefficient eps0/R truncates the domain
// with no modeling error, while grounding the same boundary (Dirichlet) gives
// the concentric-sphere value 4*pi*eps0 * a*R/(R - a) -- 50% high for R = 3a.
TEST_CASE("3D Robin far-field boundary recovers the isolated-sphere capacitance",
          "[solvers][analytic][electrostatic][robin][3d]") {
    const std::string mesh_file = "test_3d_sphere_farfield.mesh";
    const std::string matrix_file = "coupling_sphere_farfield.h5";
    constexpr double a = 0.1, R = 0.3;
    CreateSphericalShellMesh(mesh_file, a, R, 6, 6);
    const double isolated = 2.0 * Constants::TWO_PI * Constants::EPSILON_0 * a;

    const json robin = {{"name", "FarField"}, {"type", "robin"},
                        {"entity_group", "Outer"}, {"value", 0.0},
                        {"robin_coefficient", Constants::EPSILON_0 / R}};
    const double c_robin = SolveSphereCapacitance(
        mesh_file, MakeSphereConfig(mesh_file, robin, false), matrix_file);
    REQUIRE(c_robin == Catch::Approx(isolated).epsilon(2e-3));

    const json grounded = {{"name", "Ground"}, {"type", "dirichlet"},
                           {"entity_group", "Outer"}, {"value", 0.0}};
    const double c_dirichlet = SolveSphereCapacitance(
        mesh_file, MakeSphereConfig(mesh_file, grounded, false), matrix_file);
    REQUIRE(c_dirichlet == Catch::Approx(isolated * R / (R - a)).epsilon(2e-3));

    fs::remove(mesh_file);
}

// Every output format end to end on a 3D run: the ParaView collection, the
// Gmsh results file (second-order tetrahedra, read back by MFEM's own Gmsh
// reader), and the HDF5 mesh description.
TEST_CASE("3D field run writes ParaView, Gmsh and HDF5 output",
          "[solvers][output][3d]") {
    const fs::path root = fs::temp_directory_path() / "mfem_output_3d";
    fs::remove_all(root);
    fs::create_directories(root);
    const std::string mesh_file = (root / "box.mesh").string();
    CreateBoxTetMesh(mesh_file, 0.2, 0.1, 0.05, 2, 2, 2);

    json config = MakeBoxCapacitorConfig(mesh_file, 2.0);
    config["simulation"]["analysis_type"] = "field";
    config["simulation"]["order"] = 2;
    config["scenarios"] = json::array({
        {{"name", "Drive"}, {"excitations", json::array({{{"terminal", "Top"}, {"value", 1.0}}})}}});
    config["output"] = {{"directory", (root / "out").string()},
        {"paraview", {{"directory", "vtk"}}}, {"gmsh", {{"directory", "msh"}}},
        {"hdf5", {{"file", "run.h5"}}}};

    int elements = 0;
    {
        const auto mesh = mesh_io::LoadMesh(mesh_file);
        elements = mesh->GetNE();
        auto solver = SolverFactory::Instance().Create(*mesh, DecodeConfig(config));
        solver->Setup();
        solver->Run();
        solver->SaveAnalysis();
    }

    const std::string artifact = "scenario_000000_Drive";
    REQUIRE(fs::exists(root / "out/vtk" / artifact / "Cycle000000/proc000000.vtu"));

    const fs::path msh = root / "out/msh" / (artifact + ".msh");
    REQUIRE(fs::exists(msh));
    std::ifstream in(msh);
    mfem::Mesh reloaded(in, 1, 0);
    REQUIRE(reloaded.Dimension() == 3);
    REQUIRE(reloaded.GetNE() == elements);

    HighFive::File archive((root / "out/run.h5").string(), HighFive::File::ReadOnly);
    int dimension = 0;
    std::string geometry;
    archive.getGroup("mesh").getAttribute("dimension").read(dimension);
    archive.getAttribute("geometry_type").read(geometry);
    REQUIRE(dimension == 3);
    REQUIRE(geometry == "3d");

    fs::remove_all(root);
}

TEST_CASE("Solvers reject a mesh whose dimension contradicts geometry_type",
          "[solvers][geometry][3d]") {
    const std::string box_file = "test_3d_dimension_box.mesh";
    CreateBoxTetMesh(box_file, 1.0, 1.0, 1.0, 1, 1, 1);
    const auto box = mesh_io::LoadMesh(box_file);

    SECTION("a planar run on a 3D mesh") {
        json config = MakeBoxCapacitorConfig(box_file, 1.0);
        config["simulation"]["geometry_type"] = "planar";
        ElectrostaticSolver solver(*box, DecodeConfig(config));
        REQUIRE_THROWS_WITH(solver.Setup(),
            Catch::Matchers::ContainsSubstring("requires a 2D mesh"));
    }

    SECTION("a 3D run on a 2D mesh") {
        const std::string strip_file = "test_3d_dimension_strip.mesh";
        CreatePlanarStripMesh(strip_file, 0.1, 0.02, 2, 1);
        json config = MakePlanarStripConfig(
            "electrostatics", strip_file, 1, {{"epsilon_r", 1.0}}, 0.0, 1.0);
        config["simulation"]["geometry_type"] = "3d";
        mfem::Mesh strip(strip_file.c_str(), 1, 1);
        ElectrostaticSolver solver(strip, DecodeConfig(config));
        REQUIRE_THROWS_WITH(solver.Setup(),
            Catch::Matchers::ContainsSubstring("requires a 3D mesh"));
        fs::remove(strip_file);
    }

    SECTION("3D magnetics is not yet implemented") {
        json config = MakeBoxCapacitorConfig(box_file, 1.0);
        config["simulation"]["physics_type"] = "magnetostatics";
        config["terminals"] = json::array();
        MagnetostaticSolver solver(*box, DecodeConfig(config));
        REQUIRE_THROWS_WITH(solver.Setup(),
            Catch::Matchers::ContainsSubstring("H(curl)"));
    }

    fs::remove(box_file);
}

TEST_CASE("Mesh loader accepts a tetrahedral mesh and marks it for refinement",
          "[mesh_loader][3d]") {
    const std::string mesh_file = "test_3d_loader_box.mesh";
    CreateBoxTetMesh(mesh_file, 1.0, 1.0, 1.0, 2, 2, 2);
    const auto mesh = mesh_io::LoadMesh(mesh_file);

    REQUIRE(mesh->Dimension() == 3);
    REQUIRE(mesh->CheckElementOrientation(false) == 0);
    REQUIRE(mesh->CheckBdrElementOrientation(false) == 0);

    // Conforming bisection of tets needs the refinement marking applied by
    // Finalize(refine=true); refining must leave the mesh conforming.
    mfem::Array<int> marked;
    marked.Append(0);
    const int before = mesh->GetNE();
    REQUIRE_NOTHROW(amr::RefineConforming(*mesh, marked));
    REQUIRE(mesh->GetNE() > before);
    REQUIRE_FALSE(mesh->Nonconforming());
    REQUIRE(mesh->CheckElementOrientation(false) == 0);

    fs::remove(mesh_file);
}

TEST_CASE("Electrostatic coupling ignores fixed Neumann background",
          "[solvers][electrostatic][coupling][m2]") {
    const std::string mesh_file = "test_es_coupling_background.mesh";
    const std::string matrix_file = "coupling_electrostatics.h5";
    CreatePlanarStripMesh(mesh_file, 0.2, 0.05, 4, 2);

    json config = MakePlanarStripConfig(
        "electrostatics", mesh_file, 1, {{"epsilon_r", 2.5}}, 0.0, 0.0);
    config["simulation"]["analysis_type"] = "coupling_matrix";
    config["boundary_conditions"] = json::array();
    config["terminals"] = json::array({
        {{"name", "Left"}, {"quantity", "voltage"}, {"entity_group", "Left"}},
        {{"name", "Right"}, {"quantity", "voltage"}, {"entity_group", "Right"}}
    });

    auto solve = [&]() {
        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        ElectrostaticSolver solver(mesh, DecodeConfig(config, matrix_file));
        solver.Setup();
        solver.Run();
        solver.SaveAnalysis();
        return ReadHdf5Matrix(matrix_file, "Capacitance");
    };

    const auto baseline = solve();
    config["entity_groups"].push_back(
        {{"name", "Horizontal"}, {"dim", 1}, {"attribute_ids", {3}}});
    config["boundary_conditions"].push_back(
        {{"name", "BackgroundFlux"}, {"type", "neumann"},
         {"entity_group", "Horizontal"}, {"value", 3.0}});
    const auto with_background = solve();

    RequireMatricesEqual(with_background, baseline);
    fs::remove(matrix_file);
    fs::remove(mesh_file);
}

// Guards the axisymmetric 2*pi normalization end to end: the stiffness
// integrator omits the global 2*pi and GatherChargeColumn restores it exactly
// once. Applying it twice (or zero times) shifts every entry by a factor of
// 2*pi, which this analytic comparison catches.
TEST_CASE("Axisymmetric capacitance matches the analytic coaxial value",
          "[solvers][analytic][electrostatic][coupling][axisymmetric]") {
    const std::string mesh_file = "test_coax_capacitance.mesh";
    const std::string matrix_file = "coupling_electrostatics.h5";
    constexpr double r_inner = 0.01;
    constexpr double r_outer = 0.03;
    constexpr double height = 0.05;
    CreateCoaxMesh(mesh_file, r_inner, r_outer, height, 64, 1);

    json config = MakeCoaxAmrConfig(mesh_file, 1);
    config["simulation"]["amr"]["enabled"] = false;
    config["simulation"]["analysis_type"] = "coupling_matrix";

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    ElectrostaticSolver solver(mesh, DecodeConfig(config, matrix_file));
    solver.Setup();
    solver.Run();
    solver.SaveAnalysis();

    const auto matrix = ReadHdf5Matrix(matrix_file, "Capacitance");
    REQUIRE(matrix.labels == std::vector<std::string>{"Inner", "Outer"});

    // Coaxial annulus of axial extent `height` closed by symmetry planes:
    //   C = 2*pi*eps*height / ln(r_outer / r_inner)
    const double analytic_capacitance = Constants::TWO_PI * Constants::EPSILON_0 *
        height / std::log(r_outer / r_inner);

    REQUIRE(matrix.values[0][0] == Catch::Approx(analytic_capacitance).epsilon(0.01));
    REQUIRE(matrix.values[1][1] == Catch::Approx(analytic_capacitance).epsilon(0.01));
    REQUIRE(matrix.values[0][1] == Catch::Approx(-analytic_capacitance).epsilon(0.01));
    REQUIRE(matrix.values[0][1] == Catch::Approx(matrix.values[1][0]).epsilon(1e-5));

    fs::remove(matrix_file);
    fs::remove(mesh_file);
}

// The single-mesh check above pins the coaxial capacitance to 1% on one
// particular discretization. That cannot distinguish "correct and converging"
// from "wrong but inside the tolerance at nr = 64". This sweep refines the
// radial direction and measures the order of accuracy instead.
// Axisymmetric Robin: a coaxial annulus r_i <= r <= r_o, inner conductor at
// 1 V, Robin condition eps dV/dr + alpha V = 0 at r_o. With V = 1 + A ln(r/r_i),
// A = -alpha / (eps/r_o + alpha ln(r_o/r_i)), and the inner charge per the
// full revolved measure is Q = -2*pi*h*eps*A. The alpha -> infinity limit is
// the grounded coax. Exercises the 2*pi*r boundary mass term.
TEST_CASE("Axisymmetric Robin boundary matches the analytic coaxial charge",
          "[solvers][analytic][electrostatic][robin][axisymmetric]") {
    const std::string mesh_file = "test_coax_robin.mesh";
    const std::string matrix_file = "coupling_coax_robin.h5";
    constexpr double r_inner = 0.01, r_outer = 0.03, height = 0.05;
    CreateCoaxMesh(mesh_file, r_inner, r_outer, height, 64, 1);

    const double eps = Constants::EPSILON_0;
    const double alpha = 2.0 * eps / r_outer;
    json config = MakeCoaxAmrConfig(mesh_file, 1);
    config["simulation"]["amr"]["enabled"] = false;
    config["simulation"]["order"] = 2;
    config["simulation"]["analysis_type"] = "coupling_matrix";
    config["terminals"] = json::array({
        {{"name", "Inner"}, {"quantity", "voltage"}, {"entity_group", "Inner"}}});
    config["boundary_conditions"] = json::array({
        {{"name", "FarField"}, {"type", "robin"}, {"entity_group", "Outer"},
         {"value", 0.0}, {"robin_coefficient", alpha}}});

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    ElectrostaticSolver solver(mesh, DecodeConfig(config, matrix_file));
    solver.Setup();
    solver.Run();
    solver.SaveAnalysis();

    const double A = -alpha / (eps / r_outer + alpha * std::log(r_outer / r_inner));
    const double analytic = -Constants::TWO_PI * height * eps * A;
    const auto matrix = ReadHdf5Matrix(matrix_file, "Capacitance");
    REQUIRE(matrix.values[0][0] == Catch::Approx(analytic).epsilon(1e-4));

    fs::remove(matrix_file);
    fs::remove(mesh_file);
}

TEST_CASE("Axisymmetric coaxial capacitance converges at the expected order",
          "[solvers][analytic][electrostatic][coupling][axisymmetric][convergence]") {
    constexpr double r_inner = 0.01;
    constexpr double r_outer = 0.03;
    constexpr double height = 0.05;
    const double analytic_capacitance = Constants::TWO_PI * Constants::EPSILON_0 *
        height / std::log(r_outer / r_inner);

    std::vector<double> errors;
    for (const int nr : {8, 16, 32}) {
        const std::string mesh_file =
            "test_coax_convergence_" + std::to_string(nr) + ".mesh";
        const std::string matrix_file = "coupling_electrostatics.h5";
        CreateCoaxMesh(mesh_file, r_inner, r_outer, height, nr, 1);

        json config = MakeCoaxAmrConfig(mesh_file, 1);
        config["simulation"]["amr"]["enabled"] = false;
        config["simulation"]["analysis_type"] = "coupling_matrix";

        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        ElectrostaticSolver solver(mesh, DecodeConfig(config, matrix_file));
        solver.Setup();
        solver.Run();
        solver.SaveAnalysis();

        const auto matrix = ReadHdf5Matrix(matrix_file, "Capacitance");
        errors.push_back(std::abs(matrix.values[0][0] - analytic_capacitance) /
                         analytic_capacitance);

        fs::remove(matrix_file);
        fs::remove(mesh_file);
    }

    REQUIRE(errors[1] < errors[0]);
    REQUIRE(errors[2] < errors[1]);

    const double first_order = ObservedOrder(errors[0], errors[1]);
    const double second_order = ObservedOrder(errors[1], errors[2]);

    // Capacitance is an energy-derived quantity, so for order-1 elements its
    // error converges at O(h^2) even though the field gradient itself is only
    // O(h). Measured: 1.991 and 1.998, from relative errors 2.03e-3, 5.11e-4,
    // 1.28e-4 -- close enough to the asymptotic rate to bound tightly.
    //
    // The lower bound is the real assertion: it fails if the axisymmetric
    // 2*pi*r measure or the energy extraction degrades to first order. The
    // upper bound catches a pre-asymptotic coarse mesh flattering the study.
    REQUIRE(first_order > 1.8);
    REQUIRE(first_order < 2.2);
    REQUIRE(second_order > 1.8);
    REQUIRE(second_order < 2.2);

}

// The inductance matrix must be built from each MEASURED terminal's winding
// functional, not from the driving scenario's source. Reusing the drive makes
// every row of a column identical, which the asymmetry checks below reject.
TEST_CASE("Magnetostatic inductance matrix is reciprocal and distinguishes rows",
          "[solvers][magnetostatic][coupling][reciprocity]") {
    const std::string mesh_file = "test_magnetostatic_two_coil.mesh";
    const std::string matrix_file = "coupling_magnetostatics.h5";
    CreateLayeredStripMesh(mesh_file, 0.2, 0.05, 4, 2, 2);

    json config = MakePlanarStripConfig(
        "magnetostatics", mesh_file, 1, {{"mu_r", 1.0}}, 0.0, 0.0);
    config["simulation"]["analysis_type"] = "coupling_matrix";
    config["entity_groups"].push_back(
        {{"name", "CoilA"}, {"dim", 2}, {"attribute_ids", {1}}});
    config["entity_groups"].push_back(
        {{"name", "CoilB"}, {"dim", 2}, {"attribute_ids", {2}}});
    config["regions"] = json::array({
        {{"name", "CoilA"}, {"entity_group", "CoilA"}, {"material", "Material"}},
        {{"name", "CoilB"}, {"entity_group", "CoilB"}, {"material", "Material"}}
    });
    config["terminals"] = json::array({
        {{"name", "CoilA"}, {"quantity", "current"}, {"entity_group", "CoilA"}},
        {{"name", "CoilB"}, {"quantity", "current"}, {"entity_group", "CoilB"}}
    });

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    MagnetostaticSolver solver(mesh, DecodeConfig(config, matrix_file));
    solver.Setup();
    solver.Run();
    solver.SaveAnalysis();

    const auto matrix = ReadHdf5Matrix(matrix_file, "Inductance");
    REQUIRE(matrix.labels == std::vector<std::string>{"CoilA", "CoilB"});
    REQUIRE(matrix.values[0][0] > 0.0);
    REQUIRE(matrix.values[1][1] > 0.0);
    REQUIRE(matrix.values[0][1] == Catch::Approx(matrix.values[1][0]).epsilon(1e-7));
    fs::remove(matrix_file);
    fs::remove(mesh_file);
}

// Analytic check of the current-loop example against the hand calculation in
// examples/current_loop/hand_calc.ipynb. This pins the absolute scale of the
// magnetostatic inductance -- the reciprocity tests above only constrain the
// matrix's symmetry and sign, so a uniform factor error (a missing 2*pi from
// the axisymmetric measure, say) would pass them and fail here.
TEST_CASE("Magnetostatic loop inductance matches the analytic ring value",
          "[solvers][analytic][magnetostatic][coupling][axisymmetric]") {
    const std::string mesh_file = "test_current_loop_ms.mesh";
    const std::string matrix_file = "coupling_magnetostatics.h5";
    CreateCurrentLoopMesh(mesh_file);

    json config = MakeCurrentLoopConfig("magnetostatics", mesh_file, 0.0);
    config["terminals"] = json::array({
        {{"name", "LoopCurrent"}, {"quantity", "current"},
         {"entity_group", "LoopDomain"}}
    });

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    MagnetostaticSolver solver(mesh, DecodeConfig(config, matrix_file));
    solver.Setup();
    solver.Run();
    solver.SaveAnalysis();

    const auto matrix = ReadHdf5Matrix(matrix_file, "Inductance");
    REQUIRE(matrix.labels == std::vector<std::string>{"LoopCurrent"});

    // Truncating the domain at D/a = 40 biases the result low by roughly
    // -0.06%; the GMD form of the ring formula is good to ~0.01% here, so the
    // boundary dominates. 0.5% leaves headroom for that bias and for mesh
    // effects while still catching any error in the absolute scale.
    REQUIRE(matrix.values[0][0] ==
        Catch::Approx(AnalyticLoopInductance()).epsilon(0.005));

    fs::remove(matrix_file);
    fs::remove(mesh_file);
}

// The multigrid-preconditioned CG path (linear_solver "iterative") must
// reproduce the direct factorization on both scalar static operators: the
// axisymmetric r-weighted diffusion operator (coaxial capacitance) and the
// axisymmetric curl-curl operator with its 1/r term (loop inductance). With
// solver_tolerance a true relative residual of 1e-12, agreement well below
// the discretization error is expected.
TEST_CASE("Multigrid PCG and the direct solver agree on the 2D static operators",
          "[solvers][linear_solver][amg]") {
    auto solve = [](auto make_solver, json config, const std::string& matrix_file,
                    const std::string& quantity, const std::string& solver_type) {
        config["simulation"]["linear_solver"] = solver_type;
        config["simulation"]["solver_tolerance"] = 1e-12;
        auto solver = make_solver(config, matrix_file);
        solver->Setup();
        solver->Run();
        solver->SaveAnalysis();
        const auto matrix = ReadHdf5Matrix(matrix_file, quantity);
        fs::remove(matrix_file);
        return matrix;
    };

    SECTION("axisymmetric electrostatics") {
        const std::string mesh_file = "test_amg_coax.mesh";
        CreateCoaxMesh(mesh_file, 0.01, 0.03, 0.05, 64, 4);
        json config = MakeCoaxAmrConfig(mesh_file, 1);
        config["simulation"]["amr"]["enabled"] = false;
        config["simulation"]["order"] = 2;
        config["simulation"]["analysis_type"] = "coupling_matrix";
        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        auto make = [&mesh](const json& c, const std::string& archive) {
            return std::make_unique<ElectrostaticSolver>(mesh, DecodeConfig(c, archive));
        };
        const auto direct = solve(make, config, "amg_coax.h5", "Capacitance", "direct");
        const auto iterative = solve(make, config, "amg_coax.h5", "Capacitance", "iterative");
        RequireMatricesEqual(iterative, direct, 1e-8);
        fs::remove(mesh_file);
    }

    SECTION("axisymmetric magnetostatics") {
        const std::string mesh_file = "test_amg_loop.mesh";
        CreateCurrentLoopMesh(mesh_file);
        json config = MakeCurrentLoopConfig("magnetostatics", mesh_file, 0.0);
        config["terminals"] = json::array({
            {{"name", "LoopCurrent"}, {"quantity", "current"},
             {"entity_group", "LoopDomain"}}});
        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        auto make = [&mesh](const json& c, const std::string& archive) {
            return std::make_unique<MagnetostaticSolver>(mesh, DecodeConfig(c, archive));
        };
        const auto direct = solve(make, config, "amg_loop.h5", "Inductance", "direct");
        const auto iterative = solve(make, config, "amg_loop.h5", "Inductance", "iterative");
        RequireMatricesEqual(iterative, direct, 1e-8);
        REQUIRE(iterative.values[0][0] ==
            Catch::Approx(AnalyticLoopInductance()).epsilon(0.005));
        fs::remove(mesh_file);
    }
}

// Same analytic reference, but through the MQS assembly, which is a different
// code path: a complex block system with a massive port constraint rather than
// a real stiffness solve with a prescribed current density. At a low enough
// frequency the skin depth dwarfs the conductor, so the MQS inductance must
// collapse onto the magnetostatic DC value.
TEST_CASE("Magnetoquasistatic loop inductance matches the analytic ring value at low frequency",
          "[solvers][analytic][mqs][coupling][axisymmetric]") {
    const std::string mesh_file = "test_current_loop_mqs.mesh";
    const std::string matrix_file = "coupling_magnetoquasistatics.h5";
    CreateCurrentLoopMesh(mesh_file);

    json config = MakeCurrentLoopConfig("magnetoquasistatics", mesh_file, 5.8e7);
    config["terminals"] = json::array({
        {{"name", "LoopCurrent"}, {"quantity", "current"},
         {"conductor_type", "massive"}, {"entity_group", "LoopDomain"}}
    });
    config["scenarios"] = json::array({
        {{"name", "loop"}, {"frequency", 0.1}, {"excitations", json::array()}}
    });

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    MagnetoquasistaticSolver solver(mesh, DecodeConfig(config, matrix_file));
    solver.Setup();
    solver.Run();
    solver.SaveAnalysis();

    REQUIRE(fs::exists(matrix_file));
    const auto matrix = ReadHdf5Matrix(matrix_file, "Inductance");
    REQUIRE(matrix.labels == std::vector<std::string>{"LoopCurrent"});
    REQUIRE(matrix.values[0][0] ==
        Catch::Approx(AnalyticLoopInductance()).epsilon(0.005));

    fs::remove(matrix_file);
    fs::remove(mesh_file);
}


// Feasibility probe for swapping mesh generators (see docs/mesh_formats.md).
//
// Identical geometry to the magnetostatic test above, but read from a Netgen
// areamesh2 file instead of an MFEM one. Netgen carries no named groups
// whatsoever, so this passes only if group name -> attribute binding is
// resolved entirely from the configuration. The config below is byte-identical
// to the MFEM-mesh case apart from the mesh path: notably, attribute_ids did
// NOT have to change, because both writers assign the same attributes.
//
// Guarding this with the analytic value rather than a golden number means a
// format misread (swapped element/boundary blocks, off-by-one vertex indexing)
// shows up as a wrong inductance rather than as a plausible-looking solve.
TEST_CASE("Magnetostatic loop inductance is mesh-format independent (Netgen)",
          "[solvers][analytic][magnetostatic][coupling][axisymmetric][netgen]") {
    const std::string mesh_file = "test_current_loop_netgen.mesh";
    const std::string matrix_file = "coupling_magnetostatics.h5";
    CreateCurrentLoopNetgenMesh(mesh_file);

    json config = MakeCurrentLoopConfig("magnetostatics", mesh_file, 0.0);
    config["terminals"] = json::array({
        {{"name", "LoopCurrent"}, {"quantity", "current"},
         {"entity_group", "LoopDomain"}}
    });

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);

    // The Netgen reader must have produced the attributes the config names,
    // and must NOT have produced any named sets. If MFEM ever starts naming
    // Netgen groups, this is the line that will say so.
    REQUIRE(mesh.Dimension() == 2);
    REQUIRE(mesh.attributes.Max() == 2);
    REQUIRE(mesh.bdr_attributes.Max() == 3);
    REQUIRE(mesh.attribute_sets.GetAttributeSetNames().empty());
    REQUIRE(mesh.bdr_attribute_sets.GetAttributeSetNames().empty());

    MagnetostaticSolver solver(mesh, DecodeConfig(config, matrix_file));
    solver.Setup();
    solver.Run();
    solver.SaveAnalysis();

    const auto matrix = ReadHdf5Matrix(matrix_file, "Inductance");
    REQUIRE(matrix.labels == std::vector<std::string>{"LoopCurrent"});
    REQUIRE(matrix.values[0][0] ==
        Catch::Approx(AnalyticLoopInductance()).epsilon(0.005));

    fs::remove(matrix_file);
    fs::remove(mesh_file);
}


TEST_CASE("Magnetostatic coupling ignores fixed Neumann background",
          "[solvers][magnetostatic][coupling][m2]") {
    const std::string mesh_file = "test_ms_coupling_background.mesh";
    const std::string matrix_file = "coupling_magnetostatics.h5";
    CreateLayeredStripMesh(mesh_file, 0.2, 0.05, 4, 2, 2);

    json config = MakePlanarStripConfig(
        "magnetostatics", mesh_file, 1, {{"mu_r", 1.0}}, 0.0, 0.0);
    config["simulation"]["analysis_type"] = "coupling_matrix";
    config["entity_groups"].push_back(
        {{"name", "CoilA"}, {"dim", 2}, {"attribute_ids", {1}}});
    config["entity_groups"].push_back(
        {{"name", "CoilB"}, {"dim", 2}, {"attribute_ids", {2}}});
    config["regions"] = json::array({
        {{"name", "CoilA"}, {"entity_group", "CoilA"}, {"material", "Material"}},
        {{"name", "CoilB"}, {"entity_group", "CoilB"}, {"material", "Material"}}
    });
    config["terminals"] = json::array({
        {{"name", "CoilA"}, {"quantity", "current"}, {"entity_group", "CoilA"}},
        {{"name", "CoilB"}, {"quantity", "current"}, {"entity_group", "CoilB"}}
    });

    auto solve = [&]() {
        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        MagnetostaticSolver solver(mesh, DecodeConfig(config, matrix_file));
        solver.Setup();
        solver.Run();
        solver.SaveAnalysis();
        return ReadHdf5Matrix(matrix_file, "Inductance");
    };

    const auto baseline = solve();
    config["entity_groups"].push_back(
        {{"name", "Horizontal"}, {"dim", 1}, {"attribute_ids", {3}}});
    config["boundary_conditions"].push_back(
        {{"name", "BackgroundFlux"}, {"type", "neumann"},
         {"entity_group", "Horizontal"}, {"value", 2.0}});
    const auto with_background = solve();

    RequireMatricesEqual(with_background, baseline);
    fs::remove(matrix_file);
    fs::remove(mesh_file);
}

TEST_CASE("Magnetostatic solver reproduces the field of a uniform current slab",
          "[solvers][analytic][magnetostatic]") {
    const std::string mesh_file = "test_analytic_magnetostatic.mesh";
    constexpr double length = 0.1;
    constexpr double height = 0.02;
    constexpr double current_density = 2.0e6;
    constexpr int nx = 8;
    constexpr int ny = 2;
    CreatePlanarStripMesh(mesh_file, length, height, nx, ny);

    json config = MakePlanarStripConfig(
        "magnetostatics", mesh_file, 2, {{"mu_r", 1.0}}, 0.0, 0.0);
    config["terminals"] = json::array({
        {{"name", "Current"}, {"quantity", "current"},
         {"conductor_type", "stranded"}, {"entity_group", "Domain"}}
    });
    config["scenarios"][0]["excitations"] = json::array({
        {{"terminal", "Current"}, {"value", current_density * length * height}}
    });

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    MagnetostaticSolver solver(mesh, DecodeConfig(config));
    solver.Setup();
    solver.Run();

    FieldExportSet fields = solver.CollectExportFields();
    const mfem::IntegrationPoint point = TriangleCenter();
    const int element = 2 * (nx / 4);
    const FieldExport& vector_potential_field = FindField(fields, "A");
    const mfem::Vector physical =
        PhysicalPoint(*vector_potential_field.primary, element, point);
    const double x = physical(0);
    const double expected_potential =
        0.5 * Constants::MU_0 * current_density * x * (length - x);
    const double expected_by =
        Constants::MU_0 * current_density * (x - 0.5 * length);

    const double vector_potential = SamplePrimaryScalar(fields, "A", element, point);
    const mfem::Vector magnetic_field = SampleDerivedVector(fields, "B", element, point);

    REQUIRE(vector_potential == Catch::Approx(expected_potential).epsilon(1e-8));
    REQUIRE(magnetic_field(0) == Catch::Approx(0.0).margin(1e-7));
    REQUIRE(magnetic_field(1) == Catch::Approx(expected_by).epsilon(1e-7));

    fs::remove(mesh_file);
}

// The magnetostatic energy had no independent benchmark: the existing analytic
// test checks A and B pointwise, which cannot catch an error in the assembled
// measure that cancels out of the field solution. This closes that gap by
// checking the global energy against a closed form, and by deriving inductance
// from it via a route that does not reuse the solver's own extraction.
//
// Same slab as the pointwise test: uniform J_z in a strip of width `length`
// with A = 0 on both vertical faces, giving
//     A(x)   = (mu0*J/2) x (L - x)
//     B_y(x) = mu0*J (x - L/2)
// Per unit depth and per unit height, the stored energy is
//     W = (1/(2 mu0)) integral_0^L B_y^2 dx = mu0 J^2 L^3 / 24.
TEST_CASE("Magnetostatic field energy matches the analytic slab value",
          "[solvers][analytic][magnetostatic][energy]") {
    const std::string mesh_file = "test_magnetostatic_energy.mesh";
    constexpr double length = 0.1;
    constexpr double height = 0.02;
    constexpr double current_density = 2.0e6;
    constexpr int nx = 24;
    constexpr int ny = 2;
    CreatePlanarStripMesh(mesh_file, length, height, nx, ny);

    json config = MakePlanarStripConfig(
        "magnetostatics", mesh_file, 2, {{"mu_r", 1.0}}, 0.0, 0.0);
    config["terminals"] = json::array({
        {{"name", "Current"}, {"quantity", "current"},
         {"conductor_type", "stranded"}, {"entity_group", "Domain"}}
    });
    config["scenarios"][0]["excitations"] = json::array({
        {{"terminal", "Current"}, {"value", current_density * length * height}}
    });

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    MagnetostaticSolver solver(mesh, DecodeConfig(config));
    solver.Setup();
    solver.Run();

    const FieldExportSet fields = solver.CollectExportFields();

    // W = (1/(2 mu)) integral |B|^2 dOmega over the meshed cross-section.
    const double field_integral = IntegrateVectorMagnitudeSquared(fields, "B");
    const double magnetic_energy = 0.5 * field_integral / Constants::MU_0;

    // Closed form scaled by the meshed height (the model is per unit depth).
    const double analytic_energy = Constants::MU_0 * current_density *
        current_density * length * length * length * height / 24.0;

    REQUIRE(magnetic_energy == Catch::Approx(analytic_energy).epsilon(1e-6));

    // Inductance per unit depth from W = (1/2) L I^2, with the total current
    // I = J * area. This is an independent route to L: it uses only the field
    // energy, not the solver's flux-linkage extraction.
    const double total_current = current_density * length * height;
    const double energy_inductance =
        2.0 * magnetic_energy / (total_current * total_current);
    const double analytic_inductance =
        2.0 * analytic_energy / (total_current * total_current);
    REQUIRE(energy_inductance ==
        Catch::Approx(analytic_inductance).epsilon(1e-6));
    REQUIRE(energy_inductance > 0.0);

    fs::remove(mesh_file);
}

// The magnetostatic Neumann sign had no analytic benchmark, so a sign flip in
// the natural boundary term would have gone unnoticed. The natural BC for the
// A_z formulation supplies nu * dA/dn on the boundary, and since
// B = (dA/dy, -dA/dx), prescribing a positive outward value on the right-hand
// face must produce a specific B sign, not merely a nonzero field.
//
// With nu*dA/dn = g on the right face, A = 0 on the left, and no source, the
// solution is the linear field A = mu0*g*x, giving B_y = -mu0*g exactly.
TEST_CASE("Magnetostatic solver applies natural-flux Neumann data with correct sign",
          "[solvers][analytic][magnetostatic][neumann]") {
    const std::string mesh_file = "test_magnetostatic_neumann.mesh";
    constexpr double length = 0.2;
    constexpr double height = 0.05;
    constexpr double gradient = 3.0;
    constexpr int nx = 8;
    constexpr int ny = 2;
    CreatePlanarStripMesh(mesh_file, length, height, nx, ny);

    const double reluctivity = 1.0 / Constants::MU_0;
    json config = MakePlanarStripConfig(
        "magnetostatics", mesh_file, 1, {{"mu_r", 1.0}}, 0.0, 0.0);
    config["boundary_conditions"][1]["type"] = "neumann";
    config["boundary_conditions"][1]["value"] = reluctivity * gradient;

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    MagnetostaticSolver solver(mesh, DecodeConfig(config));
    solver.Setup();
    solver.Run();

    const FieldExportSet fields = solver.CollectExportFields();
    const mfem::IntegrationPoint point = TriangleCenter();
    const int element = 2 * (nx / 2);
    const FieldExport& potential_field = FindField(fields, "A");
    const mfem::Vector physical =
        PhysicalPoint(*potential_field.primary, element, point);

    REQUIRE(SamplePrimaryScalar(fields, "A", element, point) ==
        Catch::Approx(gradient * physical(0)).epsilon(1.0e-7));

    // B = (dA/dy, -dA/dx) for A = A_z(x, y). The negative sign on B_y is the
    // property under test: a flipped natural term would leave the magnitude
    // right and the sign wrong.
    const mfem::Vector magnetic_field =
        SampleDerivedVector(fields, "B", element, point);
    REQUIRE(magnetic_field(0) == Catch::Approx(0.0).margin(1.0e-6));
    REQUIRE(magnetic_field(1) == Catch::Approx(-gradient).epsilon(1.0e-7));

    // A sign flip in the Neumann assembly reverses the field, so pin the
    // orientation explicitly rather than relying on the tolerance above.
    REQUIRE(magnetic_field(1) < 0.0);

    fs::remove(mesh_file);
}

TEST_CASE("Magnetoquasistatic frequency sweep updates conducting-slab skin effect",
          "[solvers][analytic][mqs]") {
    const std::string mesh_file = "test_analytic_mqs.mesh";
    constexpr double length = 0.04;
    constexpr double height = 0.005;
    constexpr double conductivity = 3.5e7;
    constexpr double low_frequency = 60.0;
    constexpr double frequency = 600.0;
    constexpr int nx = 48;
    constexpr int ny = 1;
    CreatePlanarStripMesh(mesh_file, length, height, nx, ny);

    json config = MakePlanarStripConfig(
        "magnetoquasistatics", mesh_file, 2,
        {{"mu_r", 1.0}, {"sigma", conductivity}}, 1.0, 0.0);
    config["scenarios"] = json::array({
        {{"name", "skin"},
         {"frequency", {{"scale", "linear"}, {"start", low_frequency},
                        {"stop", frequency}, {"points", 2}}},
         {"excitations", json::array()}}
    });
    config["simulation"]["amr"] = {{"enabled", true}, {"max_iterations", 1}};

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    MagnetoquasistaticSolver solver(mesh, DecodeConfig(config));
    solver.Setup();
    solver.Run();

    FieldExportSet fields = solver.CollectExportFields();
    const mfem::IntegrationPoint point = TriangleCenter();
    const int element = 2 * (nx / 4);
    const FieldExport& real_field = FindField(fields, "A_Real");
    const mfem::Vector physical = PhysicalPoint(*real_field.primary, element, point);

    const double omega = Constants::TWO_PI * frequency;
    const std::complex<double> wave_number =
        std::sqrt(std::complex<double>(0.0, omega * Constants::MU_0 * conductivity));
    const std::complex<double> expected =
        std::sinh(wave_number * (length - physical(0))) /
        std::sinh(wave_number * length);
    const std::complex<double> low_wave_number =
        std::sqrt(std::complex<double>(
            0.0, Constants::TWO_PI * low_frequency * Constants::MU_0 * conductivity));
    const std::complex<double> low_expected =
        std::sinh(low_wave_number * (length - physical(0))) /
        std::sinh(low_wave_number * length);

    const double actual_real = SamplePrimaryScalar(fields, "A_Real", element, point);
    const double actual_imag = SamplePrimaryScalar(fields, "A_Imag", element, point);

    REQUIRE(actual_real == Catch::Approx(expected.real()).epsilon(5e-3));
    REQUIRE(actual_imag == Catch::Approx(expected.imag()).epsilon(5e-3));
    REQUIRE(std::abs(std::complex<double>(actual_real, actual_imag) - expected) <
            std::abs(std::complex<double>(actual_real, actual_imag) - low_expected));
    REQUIRE(std::hypot(actual_real, actual_imag) < 1.0);
    REQUIRE(actual_imag < 0.0);

    fs::remove(mesh_file);
}

TEST_CASE("Magnetoquasistatic skin-effect solution converges under mesh refinement",
          "[solvers][analytic][mqs][convergence]") {
    constexpr double length = 0.04;
    constexpr double height = 0.005;
    constexpr double conductivity = 3.5e7;
    constexpr double frequency = 60.0;
    const double omega = Constants::TWO_PI * frequency;
    const std::complex<double> wave_number =
        std::sqrt(std::complex<double>(0.0, omega * Constants::MU_0 * conductivity));
    const auto exact = [=](const mfem::Vector& point) {
        return std::sinh(wave_number * (length - point(0))) /
               std::sinh(wave_number * length);
    };

    std::vector<double> errors;
    for (const int nx : {8, 16, 32}) {
        const std::string mesh_file =
            "test_mqs_convergence_" + std::to_string(nx) + ".mesh";
        CreatePlanarStripMesh(mesh_file, length, height, nx, 1);

        json config = MakePlanarStripConfig(
            "magnetoquasistatics", mesh_file, 1,
            {{"mu_r", 1.0}, {"sigma", conductivity}}, 1.0, 0.0);
        config["scenarios"][0]["frequency"] = frequency;

        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        MagnetoquasistaticSolver solver(mesh, DecodeConfig(config));
        solver.Setup();
        solver.Run();
        errors.push_back(RelativeComplexL2Error(solver.CollectExportFields(), exact));

        fs::remove(mesh_file);
    }

    REQUIRE(errors[1] < errors[0]);
    REQUIRE(errors[2] < errors[1]);

    // Order-1 elements in the L2 norm converge at O(h^2), so the observed
    // order should sit near 2. Measured: 1.95 and 1.86 (the second is slightly
    // low because the 60 Hz skin depth is still only marginally resolved on
    // the coarse meshes, not because the rate is degrading).
    //
    // Bounding the rate on both sides is the point. The previous form asserted
    // only errors[i]/errors[i+1] > 2.5, i.e. order > 1.32, which a solver that
    // silently fell to first order would still pass. The upper bound catches
    // the opposite failure: a rate well above 2 usually means the coarse mesh
    // is in a pre-asymptotic regime and the study is not measuring what it
    // claims to.
    const double first_order = ObservedOrder(errors[0], errors[1]);
    const double second_order = ObservedOrder(errors[1], errors[2]);
    REQUIRE(first_order > 1.7);
    REQUIRE(first_order < 2.4);
    REQUIRE(second_order > 1.7);
    REQUIRE(second_order < 2.4);
}

TEST_CASE("MQS coupling supports mixed massive and stranded conductors",
          "[solvers][mqs][coupling][reciprocity]") {
    const std::string mesh_file = "test_mqs_mixed_conductors.mesh";
    const std::string matrix_file = "coupling_magnetoquasistatics.h5";
    CreateLayeredStripMesh(mesh_file, 0.2, 0.05, 2, 1, 1);

    json config = MakePlanarStripConfig(
        "magnetoquasistatics", mesh_file, 1, {{"mu_r", 1.0}}, 0.0, 0.0);
    config["simulation"]["analysis_type"] = "coupling_matrix";
    config["scenarios"] = json::array({
        {{"name", "low"}, {"frequency", 100.0}, {"excitations", json::array()}},
        {{"name", "high"}, {"frequency", 1000.0}, {"excitations", json::array()}},
        {{"name", "duplicate"}, {"frequency", 100.0}, {"excitations", json::array()}}
    });
    config["entity_groups"].push_back(
        {{"name", "StrandedDomain"}, {"dim", 2}, {"attribute_ids", {1}}});
    config["entity_groups"].push_back(
        {{"name", "MassiveDomain"}, {"dim", 2}, {"attribute_ids", {2}}});
    config["materials"] = json::array({
        {{"name", "StrandedMaterial"},
         {"properties", {{"mu_r", 1.0}, {"sigma", 0.0}}}},
        {{"name", "MassiveMaterial"},
         {"properties", {{"mu_r", 1.0}, {"sigma", 1.0e6}}}}
    });
    config["regions"] = json::array({
        {{"name", "StrandedRegion"}, {"entity_group", "StrandedDomain"},
         {"material", "StrandedMaterial"}},
        {{"name", "MassiveRegion"}, {"entity_group", "MassiveDomain"},
         {"material", "MassiveMaterial"}}
    });
    config["terminals"] = json::array({
        {{"name", "Massive"}, {"quantity", "current"},
         {"conductor_type", "massive"}, {"entity_group", "MassiveDomain"}},
        {{"name", "Stranded"}, {"quantity", "current"},
         {"conductor_type", "stranded"}, {"entity_group", "StrandedDomain"}}
    });

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    MagnetoquasistaticSolver solver(mesh, DecodeConfig(config, matrix_file));
    solver.Setup();
    solver.Run();
    solver.SaveAnalysis();

    {
        HighFive::File file(matrix_file, HighFive::File::ReadOnly);
        std::vector<double> frequencies;
        file.getDataSet("/coupling/frequency_hz").read(frequencies);
        REQUIRE(frequencies == std::vector<double>{100.0, 1000.0});
        REQUIRE(file.getDataSet("/coupling/Inductance/values").getDimensions() ==
            std::vector<std::size_t>{2, 2, 2});
        REQUIRE(file.getDataSet("/coupling/Resistance/values").getDimensions() ==
            std::vector<std::size_t>{2, 2, 2});
    }
    const auto low_inductance = ReadHdf5Matrix(matrix_file, "Inductance", 0);
    const auto low_resistance = ReadHdf5Matrix(matrix_file, "Resistance", 0);
    const auto inductance = ReadHdf5Matrix(matrix_file, "Inductance", 1);
    const auto resistance = ReadHdf5Matrix(matrix_file, "Resistance", 1);
    const std::vector<std::string> expected_labels{"Massive", "Stranded"};
    REQUIRE(inductance.labels == expected_labels);
    REQUIRE(resistance.labels == expected_labels);
    REQUIRE(inductance.values[0][1] ==
        Catch::Approx(inductance.values[1][0]).epsilon(1e-7));
    REQUIRE(resistance.values[0][1] ==
        Catch::Approx(resistance.values[1][0]).epsilon(1e-7));
    REQUIRE(inductance.values[0][0] > 0.0);
    REQUIRE(inductance.values[1][1] > 0.0);
    REQUIRE(resistance.values[0][0] > 0.0);
    REQUIRE(std::abs(inductance.values[0][0] - low_inductance.values[0][0]) > 1e-12);
    REQUIRE(std::abs(resistance.values[0][0] - low_resistance.values[0][0]) > 1e-12);

    fs::remove(matrix_file);
    fs::remove(mesh_file);
}

TEST_CASE("MQS coupling ignores fixed Neumann background",
          "[solvers][mqs][coupling][m2]") {
    const std::string mesh_file = "test_mqs_coupling_background.mesh";
    const std::string matrix_file = "coupling_magnetoquasistatics.h5";
    CreateLayeredStripMesh(mesh_file, 0.2, 0.05, 2, 1, 1);

    json config = MakePlanarStripConfig(
        "magnetoquasistatics", mesh_file, 1,
        {{"mu_r", 1.0}, {"sigma", 1.0e6}}, 0.0, 0.0);
    config["simulation"]["analysis_type"] = "coupling_matrix";
    config["scenarios"] = json::array({
        {{"name", "point"}, {"frequency", 1000.0}, {"excitations", json::array()}}
    });
    config["entity_groups"].push_back(
        {{"name", "DriveDomain"}, {"dim", 2}, {"attribute_ids", {1}}});
    config["regions"] = json::array({
        {{"name", "Domain"}, {"entity_group", "Domain"}, {"material", "Material"}}
    });
    config["terminals"] = json::array({
        {{"name", "Drive"}, {"quantity", "current"},
         {"conductor_type", "massive"}, {"entity_group", "DriveDomain"}}
    });

    auto solve = [&]() {
        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        MagnetoquasistaticSolver solver(mesh, DecodeConfig(config, matrix_file));
        solver.Setup();
        solver.Run();
        solver.SaveAnalysis();
        return std::pair{
            ReadHdf5Matrix(matrix_file, "Inductance"),
            ReadHdf5Matrix(matrix_file, "Resistance")};
    };

    const auto baseline = solve();
    config["entity_groups"].push_back(
        {{"name", "Horizontal"}, {"dim", 1}, {"attribute_ids", {3}}});
    config["boundary_conditions"].push_back(
        {{"name", "BackgroundFlux"}, {"type", "neumann"},
         {"entity_group", "Horizontal"}, {"value", 2.0}});
    const auto with_background = solve();

    RequireMatricesEqual(with_background.first, baseline.first);
    RequireMatricesEqual(with_background.second, baseline.second);
    fs::remove(matrix_file);
    fs::remove(mesh_file);
}

// Axisymmetric massive-port DC resistance against a closed form. The planar
// heterogeneous test pins the analogous planar value, but the axisymmetric
// conductance path is different in kind: an azimuthal conductor carries
// E_phi = V/(2*pi*r), so
//     G_dc = integral sigma/(2*pi*r) dA
//          = (sigma/(2*pi)) integral_a^b integral_0^h (1/r) dz dr
//          = sigma*h*ln(b/a)/(2*pi),
// and R -> 1/G_dc as the frequency tends to zero. That 1/r factor is physical
// and deliberately NOT part of the 2*pi*r volume measure, so this test also
// guards against the two being conflated. A frequency low enough that the skin
// depth greatly exceeds the conductor thickness isolates the DC limit.
TEST_CASE("Axisymmetric massive-port resistance matches the analytic DC value",
          "[solvers][analytic][mqs][axisymmetric][conductance]") {
    const std::string mesh_file = "test_mqs_axisym_dc_port.mesh";
    const std::string matrix_file = "coupling_magnetoquasistatics.h5";
    constexpr double r_inner = 0.05;
    constexpr double r_outer = 0.08;
    constexpr double height = 0.02;
    constexpr double conductivity = 1.0e5;
    constexpr double frequency = 1.0e-3;
    CreateCoaxMesh(mesh_file, r_inner, r_outer, height, 24, 6);

    json config{
        {"simulation", {
            {"physics_type", "magnetoquasistatics"},
            {"mesh", mesh_file},
            {"order", 2},
            {"geometry_type", "axisymmetric"},
            {"analysis_type", "coupling_matrix"},
            {"solver_tolerance", 1e-14},
            {"solver_max_iter", 4000},
            {"solver_print_level", 0}
        }},
        {"entity_groups", json::array({
            {{"name", "PortDomain"}, {"dim", 2}, {"attribute_ids", {1}}},
            {{"name", "Outer"}, {"dim", 1}, {"attribute_ids", {2}}}
        })},
        {"regions", json::array({
            {{"name", "Conductor"}, {"entity_group", "PortDomain"},
             {"material", "Conductor"}}
        })},
        {"materials", json::array({
            {{"name", "Conductor"},
             {"properties", {{"mu_r", 1.0}, {"sigma", conductivity}}}}
        })},
        {"boundary_conditions", json::array({
            {{"name", "Outer"}, {"type", "dirichlet"},
             {"entity_group", "Outer"}, {"value", 0.0}}
        })},
        {"terminals", json::array({
            {{"name", "Port"}, {"quantity", "current"},
             {"conductor_type", "massive"}, {"entity_group", "PortDomain"}}
        })},
        {"scenarios", json::array({
            {{"name", "dc"}, {"frequency", frequency},
             {"excitations", json::array()}}
        })}
    };

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    MagnetoquasistaticSolver solver(mesh, DecodeConfig(config, matrix_file));
    solver.Setup();
    solver.Run();
    solver.SaveAnalysis();

    const auto resistance = ReadHdf5Matrix(matrix_file, "Resistance");
    REQUIRE(resistance.labels == std::vector<std::string>{"Port"});
    REQUIRE(resistance.values.size() == 1);

    const double conductance = conductivity * height *
        std::log(r_outer / r_inner) / Constants::TWO_PI;
    const double expected_resistance = 1.0 / conductance;

    REQUIRE(resistance.values[0][0] ==
        Catch::Approx(expected_resistance).epsilon(1e-4));

    // Guard against the 1/r factor being dropped: without it the conductance
    // would be sigma*area/(2*pi*r_mid)-like and the resistance would differ by
    // a factor well outside the tolerance above.
    const double no_radial_weight_conductance =
        conductivity * height * (r_outer - r_inner);
    REQUIRE(std::abs(expected_resistance - 1.0 / no_radial_weight_conductance) >
            1e-2 * expected_resistance);

    fs::remove(matrix_file);
    fs::remove(mesh_file);
}

// Joule loss in the DC limit. Driving a massive port with current I and letting
// the frequency go to zero makes the conductor a pure resistance, so the
// time-averaged dissipation must approach the peak-phasor value
//
//     P = 0.5 * I^2 * R = I^2 / (2 * G_dc),   G_dc = sigma*h*ln(b/a)/(2*pi).
//
// This is the sharpest available check on the loss integral: it pins the factor
// of 1/2 (the peak-phasor convention), the axisymmetric 2*pi*r volume measure,
// and the physical 1/(2*pi*r) drive field at once, against a closed form that
// involves none of the solver's internals. It would also catch the
// tempting-but-wrong simplification P = 0.5*sigma*omega^2*|A|^2, which drops the
// drive term and tends to zero with frequency instead of to I^2/(2*G_dc).
TEST_CASE("MQS Joule loss matches the analytic DC value",
          "[solvers][analytic][mqs][axisymmetric][loss]") {
    const std::string mesh_file = "test_mqs_loss_dc.mesh";
    constexpr double r_inner = 0.05;
    constexpr double r_outer = 0.08;
    constexpr double height = 0.02;
    constexpr double conductivity = 1.0e5;
    constexpr double frequency = 1.0e-3;
    constexpr double current = 1.0;
    CreateCoaxMesh(mesh_file, r_inner, r_outer, height, 24, 6);

    json config{
        {"simulation", {
            {"physics_type", "magnetoquasistatics"},
            {"mesh", mesh_file},
            {"order", 2},
            {"geometry_type", "axisymmetric"},
            {"analysis_type", "field"},
            {"solver_tolerance", 1e-14},
            {"solver_max_iter", 4000},
            {"solver_print_level", 0}
        }},
        {"entity_groups", json::array({
            {{"name", "PortDomain"}, {"dim", 2}, {"attribute_ids", {1}}},
            {{"name", "Outer"}, {"dim", 1}, {"attribute_ids", {2}}}
        })},
        {"regions", json::array({
            {{"name", "Conductor"}, {"entity_group", "PortDomain"},
             {"material", "Conductor"}}
        })},
        {"materials", json::array({
            {{"name", "Conductor"},
             {"properties", {{"mu_r", 1.0}, {"sigma", conductivity}}}}
        })},
        {"boundary_conditions", json::array({
            {{"name", "Outer"}, {"type", "dirichlet"},
             {"entity_group", "Outer"}, {"value", 0.0}}
        })},
        {"terminals", json::array({
            {{"name", "Port"}, {"quantity", "current"},
             {"conductor_type", "massive"}, {"entity_group", "PortDomain"}}
        })},
        {"scenarios", json::array({
            {{"name", "dc"}, {"frequency", frequency},
             {"excitations", json::array({
                 {{"terminal", "Port"}, {"value", current}}
             })}}
        })}
    };

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    MagnetoquasistaticSolver solver(mesh, DecodeConfig(config));
    solver.Setup();
    solver.Run();

    const auto losses = solver.ComputeRegionLosses();
    // Exactly one entry: the terminal and the region name the same metal, and
    // an attribute must have a single reporting owner or the total double-counts.
    REQUIRE(losses.size() == 1);
    REQUIRE(losses[0].Name == "Port");

    const double conductance = conductivity * height *
        std::log(r_outer / r_inner) / Constants::TWO_PI;
    const double expected_loss = current * current / (2.0 * conductance);

    REQUIRE(losses[0].Power == Catch::Approx(expected_loss).epsilon(1e-4));

    // An RMS-convention implementation would be exactly twice this, far outside
    // the tolerance above; pin the convention explicitly.
    REQUIRE(std::abs(losses[0].Power - 2.0 * expected_loss) >
            0.1 * expected_loss);

    fs::remove(mesh_file);
}

TEST_CASE("MQS heterogeneous massive-port conductance is piecewise and order independent",
          "[solvers][mqs][coupling][conductance]") {
    const std::string mesh_file = "test_mqs_heterogeneous_port.mesh";
    const std::string matrix_file = "coupling_magnetoquasistatics.h5";
    constexpr double length = 0.2;
    constexpr double height = 0.05;
    constexpr double sigma_left = 2.0;
    constexpr double sigma_right = 5.0;
    CreateLayeredStripMesh(mesh_file, length, height, 2, 1, 1);

    auto resistance_for = [&](const json& port_attributes) {
        json config = MakePlanarStripConfig(
            "magnetoquasistatics", mesh_file, 1,
            {{"mu_r", 1.0}, {"sigma", sigma_left}}, 0.0, 0.0);
        config["simulation"]["analysis_type"] = "coupling_matrix";
        config["scenarios"] = json::array({
            {{"name", "heterogeneous"}, {"frequency", 60.0},
             {"excitations", json::array()}}
        });
        config["entity_groups"].push_back(
            {{"name", "RightDomain"}, {"dim", 2}, {"attribute_ids", {2}}});
        config["entity_groups"].push_back(
            {{"name", "PortDomain"}, {"dim", 2},
             {"attribute_ids", port_attributes}});
        config["entity_groups"].push_back(
            {{"name", "Horizontal"}, {"dim", 1}, {"attribute_ids", {3}}});
        config["materials"].push_back(
            {{"name", "RightMaterial"},
             {"properties", {{"mu_r", 1.0}, {"sigma", sigma_right}}}});
        config["regions"].push_back(
            {{"name", "RightRegion"}, {"entity_group", "RightDomain"},
             {"material", "RightMaterial"}});
        config["boundary_conditions"].push_back(
            {{"name", "Horizontal"}, {"type", "dirichlet"},
             {"entity_group", "Horizontal"}, {"value", 0.0}});
        config["terminals"] = json::array({
            {{"name", "Port"}, {"quantity", "current"},
             {"conductor_type", "massive"}, {"entity_group", "PortDomain"}}
        });

        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        MagnetoquasistaticSolver solver(mesh, DecodeConfig(config, matrix_file));
        solver.Setup();
        solver.Run();
        solver.SaveAnalysis();
        const auto resistance = ReadHdf5Matrix(matrix_file, "Resistance");
        REQUIRE(resistance.labels == std::vector<std::string>{"Port"});
        REQUIRE(resistance.values.size() == 1);
        REQUIRE(resistance.values[0].size() == 1);
        return resistance.values[0][0];
    };

    const double left_area = 0.5 * length * height;
    const double right_area = 0.5 * length * height;
    const double expected_resistance =
        1.0 / (sigma_left * left_area + sigma_right * right_area);
    const double forward = resistance_for(json::array({1, 2}));
    const double reversed = resistance_for(json::array({2, 1}));

    REQUIRE(forward == Catch::Approx(expected_resistance).epsilon(1e-6));
    REQUIRE(reversed == Catch::Approx(expected_resistance).epsilon(1e-6));
    REQUIRE(reversed == Catch::Approx(forward).epsilon(1e-12));

    fs::remove(matrix_file);
    fs::remove(mesh_file);
}

TEST_CASE("MQS coupling keeps open-current regions passive and off-matrix",
          "[solvers][mqs][coupling][regions]") {
    const std::string mesh_file = "test_mqs_open_current_region.mesh";
    const std::string matrix_file = "coupling_magnetoquasistatics.h5";
    const std::string legacy_inductance_file =
        "inductance_matrix_point_f1_1000Hz.csv";
    const std::string legacy_resistance_file =
        "resistance_matrix_point_f1_1000Hz.csv";
    fs::remove(legacy_inductance_file);
    fs::remove(legacy_resistance_file);
    CreateLayeredStripMesh(mesh_file, 0.2, 0.05, 4, 1, 2);

    json config = MakePlanarStripConfig(
        "magnetoquasistatics", mesh_file, 1,
        {{"mu_r", 1.0}, {"sigma", 1.0e6}}, 0.0, 0.0);
    config["simulation"]["analysis_type"] = "coupling_matrix";
    config["scenarios"] = json::array({
        {{"name", "point"},
         {"frequency", {{"scale", "linear"}, {"start", 1000.0},
                          {"stop", 1000.0}, {"points", 1}}},
         {"excitations", json::array()}}
    });
    config["entity_groups"].push_back(
        {{"name", "DriveDomain"}, {"dim", 2}, {"attribute_ids", {1}}});
    config["entity_groups"].push_back(
        {{"name", "PassiveDomain"}, {"dim", 2}, {"attribute_ids", {2}}});
    config["regions"] = json::array({
        {{"name", "DriveRegion"}, {"entity_group", "DriveDomain"}, {"material", "Material"}},
        {{"name", "PassiveRegion"}, {"entity_group", "PassiveDomain"}, {"material", "Material"}}
    });
    config["terminals"] = json::array({
        {{"name", "Drive"}, {"quantity", "current"},
         {"conductor_type", "massive"}, {"entity_group", "DriveDomain"}}
    });

    mfem::Mesh baseline_mesh(mesh_file.c_str(), 1, 1);
    MagnetoquasistaticSolver baseline_solver(baseline_mesh, DecodeConfig(config, matrix_file));
    baseline_solver.Setup();
    baseline_solver.Run();
    baseline_solver.SaveAnalysis();
    const auto baseline_inductance = ReadHdf5Matrix(matrix_file, "Inductance");
    const auto baseline_resistance = ReadHdf5Matrix(matrix_file, "Resistance");

    config["regions"][1]["current_constraint"] = "open";
    mfem::Mesh constrained_mesh(mesh_file.c_str(), 1, 1);
    MagnetoquasistaticSolver constrained_solver(constrained_mesh, DecodeConfig(config, matrix_file));
    constrained_solver.Setup();
    constrained_solver.Run();
    constrained_solver.SaveAnalysis();
    const auto constrained_inductance = ReadHdf5Matrix(matrix_file, "Inductance");
    const auto constrained_resistance = ReadHdf5Matrix(matrix_file, "Resistance");

    const std::vector<std::string> expected_labels{"Drive"};
    REQUIRE(constrained_inductance.labels == expected_labels);
    REQUIRE(constrained_resistance.labels == expected_labels);
    REQUIRE(constrained_inductance.values.size() == 1);
    REQUIRE(constrained_resistance.values.size() == 1);
    REQUIRE_FALSE(fs::exists(legacy_inductance_file));
    REQUIRE_FALSE(fs::exists(legacy_resistance_file));
    const double inductance_change = std::abs(
        constrained_inductance.values[0][0] - baseline_inductance.values[0][0]);
    const double resistance_change = std::abs(
        constrained_resistance.values[0][0] - baseline_resistance.values[0][0]);
    REQUIRE(inductance_change + resistance_change > 1e-12);

    fs::remove(matrix_file);
    fs::remove(mesh_file);
}

// Axisymmetric companion to the planar open-current test: an aluminum shield
// modeled as a passive open-current region, with two driven massive turns
// outside it. Covers the axisymmetric massive-port path (G_dc = integral
// sigma/(2*pi*r)) for a region that is not a terminal, which the planar test
// cannot exercise.
//
// Mesh resolution note: the shield must be resolved against the skin depth
// delta = sqrt(2/(omega*mu*sigma)). At 1 kHz in aluminum delta is about 2.7 mm,
// and the radial cell size here is about 2.7 mm, so delta/cell is roughly 1.
// Coarser meshes drive the extracted port resistance negative (unphysical
// negative loss), so do not reduce cells_per_band without also lowering the
// frequency.
TEST_CASE("MQS axisymmetric shield stays passive and loads the turns",
          "[solvers][mqs][coupling][regions][axisymmetric]") {
    const std::string mesh_file = "test_mqs_axisym_shield.mesh";
    const std::string matrix_file = "coupling_magnetoquasistatics.h5";
    CreateShieldedTurnsMesh(mesh_file, /*r_min=*/0.05, /*r_max=*/0.20,
                            /*height=*/0.04, /*nz=*/8, /*cells_per_band=*/8);

    json config = MakeShieldedTurnsConfig(mesh_file, 1000.0);

    // Reference 1: no shield at all. The shield band is air, so nothing is
    // induced there and the turns see their unshielded coupling.
    json air_config = config;
    air_config["regions"][1]["material"] = "Air";
    mfem::Mesh air_mesh(mesh_file.c_str(), 1, 1);
    MagnetoquasistaticSolver air_solver(air_mesh, DecodeConfig(air_config, matrix_file));
    air_solver.Setup();
    air_solver.Run();
    air_solver.SaveAnalysis();
    const auto air_inductance = ReadHdf5Matrix(matrix_file, "Inductance");

    // Reference 2: a conducting shield with no current constraint. Without a
    // port unknown there is no net-current equation, so the region behaves as a
    // short-circuited closed loop and shields most strongly.
    mfem::Mesh shorted_mesh(mesh_file.c_str(), 1, 1);
    MagnetoquasistaticSolver shorted_solver(shorted_mesh, DecodeConfig(config, matrix_file));
    shorted_solver.Setup();
    shorted_solver.Run();
    shorted_solver.SaveAnalysis();
    const auto shorted_inductance = ReadHdf5Matrix(matrix_file, "Inductance");

    // Case under test: the shield becomes an open-current region, gaining a
    // port unknown whose net current is pinned to zero. This is the physical
    // model for a shield that is not a closed turn.
    config["regions"][1]["current_constraint"] = "open";
    mfem::Mesh shielded_mesh(mesh_file.c_str(), 1, 1);
    MagnetoquasistaticSolver shielded_solver(shielded_mesh, DecodeConfig(config, matrix_file));
    shielded_solver.Setup();
    shielded_solver.Run();
    shielded_solver.SaveAnalysis();
    const auto shielded_inductance = ReadHdf5Matrix(matrix_file, "Inductance");
    const auto shielded_resistance = ReadHdf5Matrix(matrix_file, "Resistance");

    // The shield is passive: it never becomes a matrix row/column, so the
    // matrices stay 2x2 over the two driven turns.
    const std::vector<std::string> expected_labels{"TurnA", "TurnB"};
    REQUIRE(shielded_inductance.labels == expected_labels);
    REQUIRE(shielded_resistance.labels == expected_labels);
    REQUIRE(shielded_inductance.values.size() == 2);
    REQUIRE(shielded_resistance.values.size() == 2);

    // Reciprocity: mutual terms must match in both matrices.
    REQUIRE(shielded_inductance.values[0][1] ==
        Catch::Approx(shielded_inductance.values[1][0]).epsilon(1e-8));
    REQUIRE(shielded_resistance.values[0][1] ==
        Catch::Approx(shielded_resistance.values[1][0]).epsilon(1e-8));

    // Self terms stay physical: positive inductance and positive loss.
    for (int turn = 0; turn < 2; ++turn) {
        REQUIRE(shielded_inductance.values[turn][turn] > 0.0);
        REQUIRE(shielded_resistance.values[turn][turn] > 0.0);
    }

    // The shield is not inert: constraining net current to zero still permits
    // eddy currents to circulate within the cross-section, so the turns must
    // see different coupling than with no shield present.
    double change_vs_air = 0.0;
    for (int row = 0; row < 2; ++row) {
        for (int column = 0; column < 2; ++column) {
            change_vs_air += std::abs(shielded_inductance.values[row][column] -
                                      air_inductance.values[row][column]);
        }
    }
    REQUIRE(change_vs_air > 1e-14);

    // Shielding strength is ordered by how much net current the shield may
    // carry. A short-circuited closed loop develops a large opposing net
    // current and shields strongly. Pinning the net current to zero removes
    // that bulk opposition entirely, leaving only redistribution within the
    // cross-section, so the turns behave nearly as if the shield were absent.
    for (int turn = 0; turn < 2; ++turn) {
        const double shorted = shorted_inductance.values[turn][turn];
        const double open = shielded_inductance.values[turn][turn];
        const double air = air_inductance.values[turn][turn];

        // The closed loop shields substantially, by a margin well outside
        // discretization noise. The exact factor depends on how far the turn
        // sits from the shield, so only the direction is asserted.
        REQUIRE(shorted < 0.8 * air);
        // The zero-net-current shield does not: it stays within a few percent
        // of the unshielded value, and clearly apart from the shorted case.
        REQUIRE(std::abs(open - air) < 0.05 * air);
        REQUIRE(shorted < 0.9 * open);
    }

    fs::remove(matrix_file);
    fs::remove(mesh_file);
}

// Loss in a conductor that owns no port unknown.
//
// A flux shield or a steel brace is neither a terminal nor an open-current
// region, yet the sigma mass term induces eddy currents in it all the same, so
// it dissipates real power while appearing in no coupling matrix. This is the
// case that motivated reporting every region with sigma > 0 rather than only the
// ported ones, and it is the easiest to get silently wrong: with no port there
// is no drive amplitude to look up, so a missing-entry bug would surface as
// garbage rather than as the physically correct zero.
//
// With E_drive = 0 the general expression collapses to the closed form
// 0.5*sigma*omega^2*|A|^2, which is recomputed here directly from the exported
// potential as an independent check on the integration path.
TEST_CASE("MQS unported conductive region dissipates and is reported",
          "[solvers][mqs][loss][regions][axisymmetric]") {
    const std::string mesh_file = "test_mqs_unported_loss.mesh";
    constexpr double frequency = 1000.0;
    CreateShieldedTurnsMesh(mesh_file, /*r_min=*/0.05, /*r_max=*/0.20,
                            /*height=*/0.04, /*nz=*/8, /*cells_per_band=*/8);

    // Field analysis with one turn driven. The shield keeps the default
    // current_constraint of "none", so it owns no voltage unknown at all.
    json config = MakeShieldedTurnsConfig(mesh_file, frequency);
    config["simulation"]["analysis_type"] = "field";
    config["scenarios"] = json::array({
        {{"name", "drive"}, {"frequency", frequency},
         {"excitations", json::array({
             {{"terminal", "TurnA"}, {"value", 1.0}}
         })}}
    });

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    MagnetoquasistaticSolver solver(mesh, DecodeConfig(config));
    solver.Setup();
    solver.Run();

    const auto losses = solver.ComputeRegionLosses();

    auto find = [&losses](const std::string& name) {
        return std::find_if(losses.begin(), losses.end(),
            [&name](const auto& l) { return l.Name == name; });
    };

    // The shield is reported even though it is not a terminal and not an
    // open-current region. Regions carry no name of their own in the config, so
    // they report under their entity group, matching how CollectMassivePorts
    // names open-current region ports.
    const auto shield = find("ShieldDomain");
    REQUIRE(shield != losses.end());
    REQUIRE(shield->Power > 0.0);

    // Air has sigma = 0 and must not appear at all: no conductivity, no loss.
    REQUIRE(find("AirDomain") == losses.end());

    // Both driven turns are conductive and must be reported.
    REQUIRE(find("TurnA") != losses.end());
    REQUIRE(find("TurnB") != losses.end());

    // With no port unknown the drive field is zero, so the shield's loss must
    // equal the closed form 0.5*sigma*omega^2*|A|^2. Recomputing that directly
    // from the solved potential exercises a different code path than
    // IntegrateLossDensity and so is an independent check on the measure, the
    // quadrature, and the zero-drive default all at once.
    constexpr double aluminum_sigma = 3.5e7;
    const double omega = Constants::TWO_PI * frequency;
    const mfem::GridFunction& a_re = solver.GetSolutionReal();
    const mfem::GridFunction& a_im = solver.GetSolutionImag();
    mfem::Mesh& solved_mesh = *a_re.FESpace()->GetMesh();

    double expected = 0.0;
    for (int e = 0; e < solved_mesh.GetNE(); ++e) {
        if (solved_mesh.GetAttribute(e) != 2) { continue; }  // ShieldDomain
        mfem::ElementTransformation& T = *solved_mesh.GetElementTransformation(e);
        const mfem::FiniteElement& fe = *a_re.FESpace()->GetFE(e);
        const mfem::IntegrationRule& ir =
            mfem::IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder() + T.OrderW() + 2);
        for (int q = 0; q < ir.GetNPoints(); ++q) {
            const mfem::IntegrationPoint& ip = ir.IntPoint(q);
            T.SetIntPoint(&ip);
            mfem::Vector pos;
            T.Transform(ip, pos);
            const double a2 = a_re.GetValue(T, ip) * a_re.GetValue(T, ip) +
                              a_im.GetValue(T, ip) * a_im.GetValue(T, ip);
            expected += 0.5 * aluminum_sigma * omega * omega * a2 *
                        ip.weight * T.Weight() * Constants::TWO_PI * pos(0);
        }
    }

    REQUIRE(shield->Power == Catch::Approx(expected).epsilon(1e-10));

    fs::remove(mesh_file);
}

// Regions that must not report eddy loss, for two different reasons.
//
// A non-conducting region has sigma = 0 and physically cannot dissipate. A
// stranded terminal is a modelling choice: it represents a bundle of fine
// insulated strands carrying an imposed current, with eddy effects deliberately
// not represented, so the field-based expression 0.5*sigma*|E|^2 does not
// describe it even when the bulk material property is conductive. Applying the
// formula there anyway would yield a plausible-looking number that is simply
// wrong, which is the worst kind of error to ship.
//
// The stranded region here is given a genuinely nonzero conductivity on purpose.
// Existing mixed-conductor fixtures set sigma = 0 for stranded material, which
// would make this test pass for the wrong reason - the exclusion would be
// indistinguishable from the trivial sigma = 0 case.
TEST_CASE("MQS loss excludes non-conducting and stranded regions",
          "[solvers][mqs][loss][exclusion]") {
    const std::string mesh_file = "test_mqs_loss_exclusion.mesh";
    CreateLayeredStripMesh(mesh_file, 0.2, 0.05, 2, 1, 1);

    json config = MakePlanarStripConfig(
        "magnetoquasistatics", mesh_file, 1, {{"mu_r", 1.0}}, 0.0, 0.0);
    config["simulation"]["analysis_type"] = "field";
    config["entity_groups"].push_back(
        {{"name", "StrandedDomain"}, {"dim", 2}, {"attribute_ids", {1}}});
    config["entity_groups"].push_back(
        {{"name", "MassiveDomain"}, {"dim", 2}, {"attribute_ids", {2}}});
    config["materials"] = json::array({
        // Conductive on purpose: only the stranded modelling choice, not a zero
        // material property, may keep this region out of the loss report.
        {{"name", "StrandedMaterial"},
         {"properties", {{"mu_r", 1.0}, {"sigma", 1.0e6}}}},
        {{"name", "MassiveMaterial"},
         {"properties", {{"mu_r", 1.0}, {"sigma", 1.0e6}}}}
    });
    config["regions"] = json::array({
        {{"name", "StrandedRegion"}, {"entity_group", "StrandedDomain"},
         {"material", "StrandedMaterial"}},
        {{"name", "MassiveRegion"}, {"entity_group", "MassiveDomain"},
         {"material", "MassiveMaterial"}}
    });
    config["terminals"] = json::array({
        {{"name", "Massive"}, {"quantity", "current"},
         {"conductor_type", "massive"}, {"entity_group", "MassiveDomain"}},
        {{"name", "Stranded"}, {"quantity", "current"},
         {"conductor_type", "stranded"}, {"entity_group", "StrandedDomain"}}
    });
    config["scenarios"] = json::array({
        {{"name", "drive"}, {"frequency", 1000.0},
         {"excitations", json::array({
             {{"terminal", "Massive"}, {"value", 1.0}}
         })}}
    });

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    MagnetoquasistaticSolver solver(mesh, DecodeConfig(config));
    solver.Setup();
    solver.Run();

    const auto losses = solver.ComputeRegionLosses();
    auto find = [&losses](const std::string& name) {
        return std::find_if(losses.begin(), losses.end(),
            [&name](const auto& l) { return l.Name == name; });
    };

    // The massive conductor dissipates and is reported.
    const auto massive = find("Massive");
    REQUIRE(massive != losses.end());
    REQUIRE(massive->Power > 0.0);

    // The stranded conductor is excluded despite sigma = 1e6, under either the
    // terminal name or its entity group.
    REQUIRE(find("Stranded") == losses.end());
    REQUIRE(find("StrandedDomain") == losses.end());

    fs::remove(mesh_file);
}

// Global power balance: the total dissipation integrated over the fields must
// equal the real power delivered at the driven port,
//
//     sum_regions P = 0.5 * Re(sum_p V_p * conj(I_p)).
//
// This is the one check that ties the loss integral to the port equations, and
// it is deliberately a whole-model assertion. The shield and the undriven turn
// both dissipate while carrying zero net port current, so they contribute to the
// left side and nothing to the right; a balance test that summed only the driven
// region would fail for a reason that has nothing to do with correctness.
//
// The two sides discretize differently - one integrates a quadratic field
// quantity, the other reads a solved port unknown - so they agree only to
// discretization error rather than to round-off. The tolerance below was
// measured on this mesh, not assumed.
TEST_CASE("MQS total Joule loss balances the delivered port power",
          "[solvers][mqs][loss][balance][axisymmetric]") {
    const std::string mesh_file = "test_mqs_power_balance.mesh";
    constexpr double frequency = 1000.0;
    constexpr double current = 1.0;
    CreateShieldedTurnsMesh(mesh_file, /*r_min=*/0.05, /*r_max=*/0.20,
                            /*height=*/0.04, /*nz=*/8, /*cells_per_band=*/32);

    json config = MakeShieldedTurnsConfig(mesh_file, frequency);
    config["simulation"]["analysis_type"] = "field";
    config["simulation"]["order"] = 2;
    // The shield is left unconstrained, so it dissipates without owning a port.
    config["scenarios"] = json::array({
        {{"name", "drive"}, {"frequency", frequency},
         {"excitations", json::array({
             {{"terminal", "TurnA"}, {"value", current}}
         })}}
    });

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    MagnetoquasistaticSolver solver(mesh, DecodeConfig(config));
    solver.Setup();
    solver.Run();

    double total_loss = 0.0;
    for (const auto& loss : solver.ComputeRegionLosses()) {
        REQUIRE(loss.Power >= 0.0);  // dissipation is never negative
        total_loss += loss.Power;
    }
    REQUIRE(total_loss > 0.0);

    // Only TurnA carries net current, so it is the sole source of real power.
    // TurnB and the shield appear in total_loss but not here.
    const auto [v_re, v_im] = solver.GetPortVoltage("TurnA");
    const double delivered = 0.5 * v_re * current;

    // Measured: the relative gap is 3.0e-3 on this mesh, falling to 7.4e-4 when
    // the radial resolution is doubled - a factor of 4.0 for a 2x refinement,
    // i.e. clean second-order convergence to exact balance. That convergence is
    // what establishes the identity actually holds; the coarser mesh is kept
    // here because the finer one costs about two minutes to solve. The bound
    // sits just above the measured value so a real regression cannot hide in it.
    REQUIRE(delivered == Catch::Approx(total_loss).epsilon(4.0e-3));

    // The scope decision is load-bearing, not cosmetic. The shield and the
    // undriven turn own no net port current, so restricting the total to the
    // driven region alone must break the balance by far more than the bound
    // above. Without this, the test would still pass if unported conductors
    // were silently dropped from the total.
    double driven_only = 0.0;
    for (const auto& loss : solver.ComputeRegionLosses()) {
        if (loss.Name == "TurnA") { driven_only = loss.Power; }
    }
    REQUIRE(driven_only > 0.0);
    REQUIRE(std::abs(delivered - driven_only) > 0.01 * delivered);

    fs::remove(mesh_file);
}

// Cross-check on the shielding mechanism: a short-circuited closed shield
// excludes flux more strongly as frequency rises, because the induced opposing
// net current grows with the rate of change of flux. A zero-net-current (open)
// shield has no such bulk mechanism available, so it stays near the unshielded
// value at every frequency. Asserting the trend, rather than a single operating
// point, is what separates the two constraints physically.
TEST_CASE("MQS shielding strengthens with frequency only when current may close",
          "[solvers][mqs][coupling][regions][axisymmetric]") {
    const std::string mesh_file = "test_mqs_axisym_shield_sweep.mesh";
    // The radial cell must stay below the aluminum skin depth at the top of the
    // sweep (delta is about 0.85 mm at 10 kHz). At 32 cells per band the
    // extracted port resistance is still negative; 64 resolves it.
    CreateShieldedTurnsMesh(mesh_file, /*r_min=*/0.05, /*r_max=*/0.20,
                            /*height=*/0.04, /*nz=*/8, /*cells_per_band=*/64);

    // Self inductance of turn A with the shield present, divided by the same
    // quantity with the shield replaced by air. Lower means more flux excluded.
    auto shielding_ratio = [&](double frequency, bool closed) {
        const std::string matrix_file = "coupling_magnetoquasistatics.h5";

        json config = MakeShieldedTurnsConfig(mesh_file, frequency);
        if (!closed) config["regions"][1]["current_constraint"] = "open";

        json air_config = MakeShieldedTurnsConfig(mesh_file, frequency);
        air_config["regions"][1]["material"] = "Air";

        mfem::Mesh air_mesh(mesh_file.c_str(), 1, 1);
        MagnetoquasistaticSolver air_solver(air_mesh, DecodeConfig(air_config, matrix_file));
        air_solver.Setup();
        air_solver.Run();
        air_solver.SaveAnalysis();
        const auto air = ReadHdf5Matrix(matrix_file, "Inductance");

        mfem::Mesh shield_mesh(mesh_file.c_str(), 1, 1);
        MagnetoquasistaticSolver shield_solver(shield_mesh, DecodeConfig(config, matrix_file));
        shield_solver.Setup();
        shield_solver.Run();
        shield_solver.SaveAnalysis();
        const auto shielded = ReadHdf5Matrix(matrix_file, "Inductance");
        const auto resistance = ReadHdf5Matrix(matrix_file, "Resistance");

        // Guard the discretization: negative extracted loss means the shield is
        // under-resolved against the skin depth and the ratio is meaningless.
        for (int turn = 0; turn < 2; ++turn) {
            REQUIRE(resistance.values[turn][turn] > 0.0);
        }

        fs::remove(matrix_file);
        return shielded.values[0][0] / air.values[0][0];
    };

    // A closed (short-circuited) shield excludes more flux as frequency rises.
    const double closed_low = shielding_ratio(100.0, /*closed=*/true);
    const double closed_high = shielding_ratio(10000.0, /*closed=*/true);
    REQUIRE(closed_low < 1.0);
    REQUIRE(closed_high < closed_low);

    // An open shield cannot carry net current, so no such trend develops: it
    // stays near the unshielded value at both ends of the sweep.
    const double open_low = shielding_ratio(100.0, /*closed=*/false);
    const double open_high = shielding_ratio(10000.0, /*closed=*/false);
    REQUIRE(std::abs(open_low - 1.0) < 0.05);
    REQUIRE(std::abs(open_high - 1.0) < 0.05);
    REQUIRE(closed_high < 0.5 * open_high);

    fs::remove(mesh_file);
}


TEST_CASE("AMR refines an axisymmetric coax and stays conforming", "[solvers][amr]") {
    const std::string mesh_file = "test_amr_coax.mesh";
    CreateCoaxMesh(mesh_file, /*r_inner=*/1.0, /*r_outer=*/4.0,
                   /*height=*/1.0, /*nr=*/6, /*nz=*/4);

    json config = MakeCoaxAmrConfig(mesh_file, /*max_iterations=*/4);

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);

    ElectrostaticAmrProbe solver(mesh, DecodeConfig(config));
    solver.Setup();
    REQUIRE_NOTHROW(solver.Run());

    const auto& history = solver.AmrHistory();
    REQUIRE(history.size() >= 2);

    // (1) Conforming throughout: a simplex mesh refined conformingly never
    //     allocates an ncmesh; RefineConforming() also throws otherwise.
    REQUIRE(mesh.ncmesh == nullptr);
    REQUIRE_FALSE(mesh.Nonconforming());

    // (2) Work grows monotonically: each refined iteration adds true DOFs.
    for (std::size_t k = 1; k < history.size(); ++k) {
        REQUIRE(history[k].true_dofs > history[k - 1].true_dofs);
    }

    // (3) The recovery-based global error estimate trends down as we refine.
    //     Compare first vs last to avoid over-constraining intermediate steps.
    REQUIRE(history.back().global_error < history.front().global_error);

    // (4) Peak |E| stays finite/positive and physically plausible. With an
    //     analytic field E_r = V0 / (r ln(r_o/r_i)) the maximum is at the inner
    //     surface: 1 / (1 * ln(4)). Linear FEM recovers the peak from below and
    //     approaches it under refinement, so require the final peak to sit in a
    //     band around analytic - tight enough to catch a wrong axisymmetric
    //     measure, NaN, or sign error, loose enough not to depend on the exact
    //     (Dorfler + bisection) refinement depth.
    const double analytic_peak = 1.0 / (1.0 * std::log(4.0));
    const double last_peak = history.back().peak_field_magnitude;
    REQUIRE(std::isfinite(last_peak));
    REQUIRE(last_peak > 0.5 * analytic_peak);
    REQUIRE(last_peak < 1.2 * analytic_peak);

    fs::remove(mesh_file);
}

TEST_CASE("Axisymmetric electrostatic AMR applies permittivity once",
          "[solvers][amr][materials][m1]") {
    const std::string mesh_file = "test_amr_axisymmetric_es_materials.mesh";
    CreateCoaxMesh(mesh_file, 1.0, 4.0, 1.0, 6, 2, 3);

    json config = MakeCoaxAmrConfig(mesh_file, 1);
    config["simulation"]["amr"]["enabled"] = false;
    config["entity_groups"].push_back(
        {{"name", "OuterDielectric"}, {"dim", 2}, {"attribute_ids", {2}}});
    config["materials"].push_back(
        {{"name", "HighPermittivity"}, {"properties", {{"epsilon_r", 100.0}}}});
    config["regions"].push_back(
        {{"name", "OuterDielectric"}, {"entity_group", "OuterDielectric"},
         {"material", "HighPermittivity"}});

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    ElectrostaticAmrProbe solver(mesh, DecodeConfig(config));
    solver.Setup();
    solver.Run();

    const mfem::Vector actual = solver.LocalErrors();
    FieldExportSet fields = solver.CollectExportFields();
    mfem::GridFunction& potential = *FindField(fields, "V").primary;
    mfem::Vector epsilon_values(2);
    epsilon_values(0) = Constants::EPSILON_0;
    epsilon_values(1) = 100.0 * Constants::EPSILON_0;
    mfem::PWConstCoefficient epsilon(epsilon_values);
    const mfem::Vector intended =
        AxisymmetricDiffusionErrors(mesh, potential, epsilon, false);
    const mfem::Vector coefficient_cubed =
        AxisymmetricDiffusionErrors(mesh, potential, epsilon, true);

    RequireErrorsEqual(actual, intended);
    REQUIRE(ErrorsDiffer(actual, coefficient_cubed));

    mfem::Array<int> actual_marked;
    mfem::Array<int> intended_marked;
    amr::MarkElementsDorfler(actual, 0.7, actual_marked);
    amr::MarkElementsDorfler(intended, 0.7, intended_marked);
    REQUIRE(actual_marked == intended_marked);

    fs::remove(mesh_file);
}

TEST_CASE("Axisymmetric magnetostatic AMR applies reluctivity once",
          "[solvers][amr][materials][m1]") {
    const std::string mesh_file = "test_amr_axisymmetric_ms_materials.mesh";
    CreateCoaxMesh(mesh_file, 1.0, 4.0, 1.0, 6, 2, 3);

    json config = MakeCoaxAmrConfig(mesh_file, 1);
    config["simulation"]["physics_type"] = "magnetostatics";
    config["simulation"]["amr"]["enabled"] = false;
    config["entity_groups"].push_back(
        {{"name", "OuterMagnetic"}, {"dim", 2}, {"attribute_ids", {2}}});
    config["materials"] = json::array({
        {{"name", "LowPermeability"}, {"properties", {{"mu_r", 1.0}}}},
        {{"name", "HighPermeability"}, {"properties", {{"mu_r", 100.0}}}}
    });
    config["regions"] = json::array({
        {{"name", "InnerMagnetic"}, {"entity_group", "Dielectric"},
         {"material", "LowPermeability"}},
        {{"name", "OuterMagnetic"}, {"entity_group", "OuterMagnetic"},
         {"material", "HighPermeability"}}
    });
    config.erase("terminals");
    config["boundary_conditions"] = json::array({
        {{"name", "Inner"}, {"type", "dirichlet"},
         {"entity_group", "Inner"}, {"value", 1.0}},
        {{"name", "Outer"}, {"type", "dirichlet"},
         {"entity_group", "Outer"}, {"value", 0.0}}
    });
    config["scenarios"] = json::array({
        {{"name", "energized"}, {"excitations", json::array()}}
    });

    mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
    MagnetostaticAmrProbe solver(mesh, DecodeConfig(config));
    solver.Setup();
    solver.Run();

    const mfem::Vector actual = solver.LocalErrors();
    FieldExportSet fields = solver.CollectExportFields();
    mfem::GridFunction& potential = *FindField(fields, "A").primary;
    mfem::Vector reluctivity_values(2);
    reluctivity_values(0) = 1.0 / Constants::MU_0;
    reluctivity_values(1) = 1.0 / (100.0 * Constants::MU_0);
    mfem::PWConstCoefficient reluctivity(reluctivity_values);
    const mfem::Vector intended =
        AxisymmetricCurlCurlErrors(mesh, potential, reluctivity, false);
    const mfem::Vector coefficient_cubed =
        AxisymmetricCurlCurlErrors(mesh, potential, reluctivity, true);

    RequireErrorsEqual(actual, intended);
    REQUIRE(ErrorsDiffer(actual, coefficient_cubed));

    mfem::Array<int> actual_marked;
    mfem::Array<int> intended_marked;
    amr::MarkElementsDorfler(actual, 0.7, actual_marked);
    amr::MarkElementsDorfler(intended, 0.7, intended_marked);
    REQUIRE(actual_marked == intended_marked);

    fs::remove(mesh_file);
}

TEST_CASE("AMR with multiple scenarios writes a shared conforming mesh", "[solvers][amr]") {
    const std::string mesh_file = "test_amr_shared.mesh";
    CreateCoaxMesh(mesh_file, /*r_inner=*/1.0, /*r_outer=*/4.0,
                   /*height=*/1.0, /*nr=*/6, /*nz=*/4);

    // Unique temp output dir so the two results files land next to the mesh.
    const fs::path tmp_dir = fs::temp_directory_path() / "mfem_amr_shared_test";
    fs::create_directories(tmp_dir);
    const fs::path mesh_in_tmp = tmp_dir / mesh_file;
    fs::copy_file(mesh_file, mesh_in_tmp, fs::copy_options::overwrite_existing);

    json config = MakeCoaxAmrConfig(mesh_in_tmp.string(), /*max_iterations=*/3);
    config["output"] = {{"directory", tmp_dir.string()},
        {"gmsh", {{"directory", "."}}}, {"hdf5", json::object()}};
    // Two scenarios; AMR builds ONE shared mesh and writes both results on it.
    config["scenarios"] = json::array({
        json{{"name", "driveA"}, {"excitations", json::array({
            json{{"terminal", "Inner"}, {"value", 1.0}},
            json{{"terminal", "Outer"}, {"value", 0.0}}
        })}},
        json{{"name", "driveB"}, {"excitations", json::array({
            json{{"terminal", "Inner"}, {"value", 0.0}},
            json{{"terminal", "Outer"}, {"value", 1.0}}
        })}}
    });

    mfem::Mesh mesh(mesh_in_tmp.string().c_str(), 1, 1);

    ElectrostaticSolver solver(mesh, DecodeConfig(config));
    solver.Setup();
    REQUIRE_NOTHROW(solver.Run());

    REQUIRE(mesh.ncmesh == nullptr);

    solver.SaveAnalysis();
    const fs::path resA = tmp_dir / "scenario_000000_driveA.msh";
    const fs::path resB = tmp_dir / "scenario_000001_driveB.msh";
    REQUIRE(fs::exists(resA));
    REQUIRE(fs::exists(resB));

    // Spec: every <scenario>.results.msh shares an identical mesh tiling. The
    // $Nodes and $Elements sections must be byte-identical across scenarios.
    REQUIRE(ExtractMshSection(resA.string(), "Nodes") ==
            ExtractMshSection(resB.string(), "Nodes"));
    REQUIRE(ExtractMshSection(resA.string(), "Elements") ==
            ExtractMshSection(resB.string(), "Elements"));

    {
        HighFive::File archive((tmp_dir / "results.h5").string(), HighFive::File::ReadOnly);
        REQUIRE(archive.getGroup("scenarios").getNumberObjects() == 2);
        std::string mesh_text;
        archive.getDataSet("mesh/mfem").read(mesh_text);
        std::istringstream stream(mesh_text);
        mfem::Mesh saved_mesh(stream, 1, 0, false);
        REQUIRE(saved_mesh.GetNE() == mesh.GetNE());
        REQUIRE(saved_mesh.GetNV() == mesh.GetNV());
        mfem::H1_FECollection collection(1, mesh.Dimension());
        mfem::FiniteElementSpace space(&saved_mesh, &collection);
        REQUIRE(archive.getDataSet("scenarios/scenario_000000/fields/V/values").getElementCount()
            == space.GetVSize());
    }

    fs::remove(mesh_file);
    std::error_code ec;
    fs::remove_all(tmp_dir, ec);
}

TEST_CASE("Solvers share output destinations and coupling field policy", "[solvers][output][hdf5]") {
    for (const std::string physics : {"electrostatics", "magnetostatics", "magnetoquasistatics"}) {
        DYNAMIC_SECTION(physics) {
            const fs::path root = fs::temp_directory_path() / ("mfem_output_policy_" + physics);
            fs::remove_all(root);
            fs::create_directories(root);
            const std::string mesh_file = (root / "input.mesh").string();
            CreatePlanarStripMesh(mesh_file, 0.2, 0.1, 3, 2);
            json source = MakePlanarStripConfig(physics, mesh_file, 1,
                {{"epsilon_r", 1.0}, {"mu_r", 1.0}, {"sigma", 0.0}}, 0.0, 0.0);
            source["simulation"]["analysis_type"] = "coupling_matrix";
            const bool electrostatic = physics == "electrostatics";
            const bool mqs = physics == "magnetoquasistatics";
            source["terminals"] = json::array({{
                {"name", "Drive/A"}, {"quantity", electrostatic ? "voltage" : "current"},
                {"entity_group", electrostatic ? "Left" : "Domain"}, {"conductor_type", "stranded"}}});
            if (electrostatic) source["boundary_conditions"].erase(source["boundary_conditions"].begin());
            source["scenarios"] = json::array({
                {{"name", "First/point"}, {"frequency", 50.0},
                 {"excitations", json::array({{{"terminal", "Drive/A"}, {"value", 1.0}}})}},
                {{"name", "Second/point"}, {"frequency", 100.0},
                 {"excitations", json::array({{{"terminal", "Drive/A"}, {"value", 2.0}}})}}
            });
            source["output"] = {{"directory", (root / "out").string()},
                {"paraview", {{"directory", "vtk"}}}, {"gmsh", {{"directory", "msh"}}},
                {"hdf5", {{"file", "archive/run.h5"}}}};
            bool fields_enabled = false;
            bool formats_enabled = true;
            bool field_analysis = false;
            SECTION("coupling fields default to disabled") {}
            SECTION("coupling fields can be enabled") {
                source["output"]["export_fields_for_coupling_matrix"] = true;
                fields_enabled = true;
            }
            SECTION("field analysis supports HDF5 alone") {
                source["simulation"]["analysis_type"] = "field";
                source["output"].erase("paraview");
                source["output"].erase("gmsh");
                fields_enabled = true;
                field_analysis = true;
            }
            SECTION("omitting formats produces no files") {
                source["output"] = {{"directory", (root / "out").string()}, {"export_fields_for_coupling_matrix", true}};
                formats_enabled = false;
            }
            {
                mfem::Mesh mesh(mesh_file.c_str(), 1, 0);
                auto solver = SolverFactory::Instance().Create(mesh, DecodeConfig(source));
                solver->Setup();
                solver->Run();
                solver->SaveAnalysis();
                solver->SaveAnalysis();
            }
            if (formats_enabled) {
                HighFive::File archive((root / "out/archive/run.h5").string(), HighFive::File::ReadOnly);
                REQUIRE(archive.exist("mesh/mfem"));
                REQUIRE(archive.exist("coupling") == !field_analysis);
                REQUIRE(archive.exist("scenarios") == fields_enabled);
                if (fields_enabled) {
                    const int count = field_analysis || mqs ? 2 : 1;
                    REQUIRE(archive.getGroup("scenarios").getNumberObjects() == count);
                    const std::string primary = electrostatic ? "V" : (mqs ? "A_Real" : "A");
                    REQUIRE(archive.exist("scenarios/scenario_000000/fields/" + primary + "/values"));
                    if (mqs) {
                        double frequency = 0.0;
                        archive.getGroup("scenarios/scenario_000001").getAttribute("frequency_hz").read(frequency);
                        REQUIRE(frequency == 100.0);
                    }
                    if (!field_analysis) {
                        std::string terminal;
                        archive.getGroup("scenarios/scenario_000000").getAttribute("driven_terminal").read(terminal);
                        REQUIRE(terminal == "Drive/A");
                    }
                }
            } else {
                REQUIRE_FALSE(fs::exists(root / "out"));
            }
            const bool visualization = formats_enabled && fields_enabled && !field_analysis;
            REQUIRE(fs::exists(root / "out/vtk") == visualization);
            REQUIRE(fs::exists(root / "out/msh") == visualization);
            if (visualization) {
                const std::string artifact = mqs ? "scenario_000000_First_point_Drive_A"
                    : "scenario_000000_CouplingMatrix_Drive_A_Drive_A";
                REQUIRE(fs::exists(root / "out/vtk" / artifact / (artifact + ".pvd")));
                REQUIRE(fs::exists(root / "out/msh" / (artifact + ".msh")));
            }
            fs::remove_all(root);
        }
    }
}

// Axisymmetric open-boundary mesh: a fixed rectangular coil section (attribute
// 2) embedded in air (attribute 1), surrounded by a pad of air out to the
// truncation surface. Boundary attributes are assigned by geometry:
// 1 = symmetry axis (r=0), 2 = outer radial surface (r=r_far),
// 3 = top/bottom surfaces (z=+/-z_far).
//
// The cell layout inside the coil is FIXED regardless of the pad, and the air
// cell SIZE is held constant by scaling the air cell count with the pad. Both
// matter: holding the air cell count fixed instead would stretch the air
// elements as the pad grows, so the discretization would coarsen in step with
// the truncation distance and the two effects could not be told apart. With
// cell size pinned, the only thing that changes between cases is how much air
// separates the coil from the truncation surface.
void CreateOpenCoilMesh(const std::string& filename,
                        double r_coil_inner, double r_coil_outer,
                        double z_coil_half, double pad) {
    const double r_far = r_coil_outer + pad;
    const double z_far = z_coil_half + pad;

    // Air cells per unit length, matched to the coil band's own resolution.
    constexpr int kAirCellsPerCoilRadius = 8;
    const int pad_cells = static_cast<int>(
        std::lround(kAirCellsPerCoilRadius * pad / r_coil_outer));

    auto fill = [](std::vector<double>& out, double a, double b, int cells) {
        for (int i = 1; i <= cells; ++i) {
            out.push_back(a + (b - a) * static_cast<double>(i) / cells);
        }
    };

    std::vector<double> r{0.0};
    fill(r, 0.0, r_coil_inner, 4);
    fill(r, r_coil_inner, r_coil_outer, 2);
    fill(r, r_coil_outer, r_far, pad_cells);

    std::vector<double> z{-z_far};
    fill(z, -z_far, -z_coil_half, pad_cells);
    fill(z, -z_coil_half, z_coil_half, 2);
    fill(z, z_coil_half, z_far, pad_cells);

    const int nr = static_cast<int>(r.size()) - 1;
    const int nz = static_cast<int>(z.size()) - 1;
    const int nvr = nr + 1;
    auto vid = [nvr](int i, int j) { return j * nvr + i; };

    std::ofstream m(filename);
    m << "MFEM mesh v1.0\n\n";
    m << "dimension\n2\n\n";

    m << "elements\n" << (2 * nr * nz) << "\n";
    for (int j = 0; j < nz; ++j) {
        for (int i = 0; i < nr; ++i) {
            const double rc = 0.5 * (r[i] + r[i + 1]);
            const double zc = 0.5 * (z[j] + z[j + 1]);
            const bool coil = rc > r_coil_inner && rc < r_coil_outer &&
                              zc > -z_coil_half && zc < z_coil_half;
            const int attribute = coil ? 2 : 1;
            const int v00 = vid(i, j);
            const int v10 = vid(i + 1, j);
            const int v11 = vid(i + 1, j + 1);
            const int v01 = vid(i, j + 1);
            m << attribute << " 2 " << v00 << " " << v10 << " " << v11 << "\n";
            m << attribute << " 2 " << v00 << " " << v11 << " " << v01 << "\n";
        }
    }
    m << "\n";

    std::vector<std::array<int, 3>> bdr; // {attr, va, vb}
    for (int j = 0; j < nz; ++j) {
        bdr.push_back({ 1, vid(0, j),  vid(0, j + 1) });
        bdr.push_back({ 2, vid(nr, j), vid(nr, j + 1) });
    }
    for (int i = 0; i < nr; ++i) {
        bdr.push_back({ 3, vid(i, 0),  vid(i + 1, 0) });
        bdr.push_back({ 3, vid(i, nz), vid(i + 1, nz) });
    }

    m << "boundary\n" << bdr.size() << "\n";
    for (const auto& b : bdr) {
        m << b[0] << " 1 " << b[1] << " " << b[2] << "\n";
    }
    m << "\n";

    m << "vertices\n" << (nvr * (nz + 1)) << "\n2\n";
    for (int j = 0; j <= nz; ++j) {
        for (int i = 0; i <= nr; ++i) {
            m << r[i] << " " << z[j] << "\n";
        }
    }
    m.close();
}

// The far-field truncation study that decides whether an absorbing (Robin)
// boundary condition is worth building at all.
//
// An open-boundary model is currently approximated by placing a Dirichlet
// A_phi = 0 surface at a finite distance. That surface forces the return flux
// to close early, so it UNDERESTIMATES the self inductance; the error is a
// property of the truncation distance, not of the mesh. Moving the surface
// outward while holding the coil and its discretization fixed isolates it.
//
// The assertion is convergence rather than a fixed value: no closed form gives
// the self inductance of a finite-section loop, and pinning a number here would
// make the test a change detector instead of a physics check. What must hold is
// that the sequence rises (flux is progressively less confined) and that the
// increments shrink (the truncation error is actually converging). A run that
// violated either would mean the far-field treatment is not converging at all,
// which is the only result that would make an ABC mandatory rather than
// optional.
TEST_CASE("Magnetostatic far-field truncation error converges as the boundary recedes",
          "[solvers][magnetostatic][axisymmetric][farfield][truncation]") {
    constexpr double r_coil_inner = 0.02;
    constexpr double r_coil_outer = 0.03;
    constexpr double z_coil_half  = 0.005;
    const std::string matrix_file = "coupling_magnetostatics.h5";

    auto self_inductance = [&](int index, double pad) {
        const std::string mesh_file =
            "test_ms_farfield_" + std::to_string(index) + ".mesh";
        CreateOpenCoilMesh(mesh_file, r_coil_inner, r_coil_outer, z_coil_half, pad);

        json config = json{
            {"simulation", {
                {"physics_type", "magnetostatics"},
                {"mesh", mesh_file},
                {"order", 1},
                {"geometry_type", "axisymmetric"},
                {"analysis_type", "coupling_matrix"},
                {"solver_tolerance", 1e-12},
                {"solver_max_iter", 4000},
                {"solver_print_level", 0}
            }},
            {"entity_groups", json::array({
                {{"name", "AirDomain"},  {"dim", 2}, {"attribute_ids", {1}}},
                {{"name", "CoilDomain"}, {"dim", 2}, {"attribute_ids", {2}}},
                {{"name", "FarField"},   {"dim", 1}, {"attribute_ids", {2, 3}}}
            })},
            {"regions", json::array({
                {{"name", "Air"},  {"entity_group", "AirDomain"},  {"material", "Air"}},
                {{"name", "Coil"}, {"entity_group", "CoilDomain"}, {"material", "Air"}}
            })},
            {"materials", json::array({
                {{"name", "Air"}, {"properties", {{"mu_r", 1.0}}}}
            })},
            // The axis (attribute 1) is deliberately unassigned: A_phi = 0 there
            // is imposed by the solver's axis regularity, not by a prescribed
            // condition. Only the truncation surface is Dirichlet.
            {"boundary_conditions", json::array({
                {{"name", "FarField"}, {"type", "dirichlet"},
                 {"entity_group", "FarField"}, {"value", 0.0}}
            })},
            {"terminals", json::array({
                {{"name", "Coil"}, {"quantity", "current"},
                 {"entity_group", "CoilDomain"}}
            })},
            {"scenarios", json::array({
                {{"name", "unit"}, {"excitations", json::array()}}
            })}
        };

        mfem::Mesh mesh(mesh_file.c_str(), 1, 1);
        MagnetostaticSolver solver(mesh, DecodeConfig(config, matrix_file));
        solver.Setup();
        solver.Run();
        solver.SaveAnalysis();

        const auto matrix = ReadHdf5Matrix(matrix_file, "Inductance");
        REQUIRE(matrix.labels == std::vector<std::string>{"Coil"});
        fs::remove(mesh_file);
        return matrix.values[0][0];
    };

    const double near = self_inductance(0, 1.0 * r_coil_outer);
    const double mid  = self_inductance(1, 2.0 * r_coil_outer);
    const double far  = self_inductance(2, 4.0 * r_coil_outer);

    // Reported so the test doubles as the measurement: this is the number that
    // says whether Dirichlet-at-distance is good enough for a given model.
    const double truncation_error = (far - near) / far;
    INFO("L(pad=1x) = " << near << ", L(pad=2x) = " << mid << ", L(pad=4x) = " << far
         << "; tightest-truncation relative error = " << truncation_error);

    REQUIRE(near > 0.0);
    REQUIRE(mid > near);
    REQUIRE(far > mid);
    REQUIRE((far - mid) < (mid - near));

    fs::remove(matrix_file);
}
