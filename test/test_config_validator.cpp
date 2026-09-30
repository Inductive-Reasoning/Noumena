// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include "config/config_validator.hpp"
#include "config/boundary_validation.hpp"
#include <functional>
#include <limits>

namespace {

bool HasError(const ConfigValidator& validator, const std::string& field);

json ValidConfig() {
	return json{
		{"simulation", {
			{"physics_type", "electrostatics"},
			{"geometry_type", "planar"},
			{"analysis_type", "field"},
			{"mesh", "test.msh"},
			{"order", 1}
		}},
		{"entity_groups", json::array({
			{{"name", "Domain"}, {"dim", 2}, {"attribute_ids", {1}}},
			{{"name", "Boundary"}, {"dim", 1}, {"attribute_ids", {1}}}
		})},
		{"regions", json::array({
			{{"entity_group", "Domain"}, {"material", "Dielectric"}}
		})},
		{"materials", json::array({
			{{"name", "Dielectric"}, {"properties", {{"epsilon_r", 1.0}}}}
		})},
		{"terminals", json::array({
			{{"name", "Drive"}, {"quantity", "voltage"}, {"entity_group", "Boundary"}}
		})},
		{"boundary_conditions", json::array({
			{{"type", "dirichlet"}, {"entity_group", "Boundary"}, {"value", 0.0}}
		})},
		{"scenarios", json::array({
			{{"name", "Driven"}, {"excitations", json::array({
				{{"terminal", "Drive"}, {"value", 1.0}}
			})}}
		})}
	};
}

TEST_CASE("BoundaryConditionValidator distinguishes essential and natural conditions",
		  "[boundary_validation]") {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
		1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
	for (int i = 0; i < mesh.GetNBE(); ++i) {
		mesh.GetBdrElement(i)->SetAttribute(i + 1);
	}

	mfem::H1_FECollection fec(1, mesh.Dimension());
	mfem::FiniteElementSpace fespace(&mesh, &fec);
	BoundaryConditionValidator validator(mesh, fespace);
	const std::unordered_map<std::string, mfem::Array<int>> no_terminals;

	auto marker = [&](int attribute) {
		mfem::Array<int> result(mesh.GetNBE());
		result = 0;
		result[attribute - 1] = 1;
		return result;
	};
	int adjacent_attribute = -1;
	mfem::Array<int> first_vertices;
	mesh.GetBdrElementVertices(0, first_vertices);
	for (int boundary = 1; boundary < mesh.GetNBE() && adjacent_attribute < 0;
		 ++boundary) {
		mfem::Array<int> candidate_vertices;
		mesh.GetBdrElementVertices(boundary, candidate_vertices);
		for (int i = 0; i < first_vertices.Size(); ++i) {
			for (int j = 0; j < candidate_vertices.Size(); ++j) {
				if (first_vertices[i] == candidate_vertices[j]) {
					adjacent_attribute = boundary + 1;
				}
			}
		}
	}
	REQUIRE(adjacent_attribute > 0);

	const BoundaryCondition dirichlet_zero(
		BoundaryConditionType::Dirichlet, "First", 0.0);
	const BoundaryCondition dirichlet_one(
		BoundaryConditionType::Dirichlet, "Second", 1.0);
	const BoundaryCondition neumann(
		BoundaryConditionType::Neumann, "Second", 2.0);

	SECTION("allows a Dirichlet and Neumann boundary junction") {
		const std::vector<MarkedBoundaryCondition> closures = {
			{marker(1), dirichlet_zero}, {marker(adjacent_attribute), neumann}
		};
		REQUIRE_NOTHROW(validator.ValidateBoundaryConditions(
			closures, no_terminals));
	}

	SECTION("rejects different Dirichlet values at a shared corner") {
		const std::vector<MarkedBoundaryCondition> closures = {
			{marker(1), dirichlet_zero},
			{marker(adjacent_attribute), dirichlet_one}
		};
		REQUIRE_THROWS(validator.ValidateBoundaryConditions(
			closures, no_terminals));
	}

	SECTION("rejects duplicate assignments on one boundary attribute") {
		const std::vector<MarkedBoundaryCondition> closures = {
			{marker(1), dirichlet_zero}, {marker(1), neumann}
		};
		REQUIRE_THROWS(validator.ValidateBoundaryConditions(
			closures, no_terminals));
	}
}

TEST_CASE("ConfigValidator validates boundary condition values and Robin metadata",
		  "[config_validator][boundary_conditions]") {
	SECTION("accepts Neumann without a Robin coefficient") {
		json config = ValidConfig();
		config["boundary_conditions"][0]["type"] = "neumann";
		config["boundary_conditions"][0]["value"] = -2.5;

		ConfigValidator validator;
		REQUIRE(validator.Validate(config));
	}

	SECTION("rejects non-finite boundary values") {
		json config = ValidConfig();
		config["boundary_conditions"][0]["value"] =
			std::numeric_limits<double>::infinity();

		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "boundary_conditions[0].value"));
	}

	SECTION("requires a finite coefficient for Robin") {
		json missing = ValidConfig();
		missing["boundary_conditions"][0]["type"] = "robin";

		ConfigValidator missing_validator;
		REQUIRE_FALSE(missing_validator.Validate(missing));
		REQUIRE(HasError(missing_validator,
						 "boundary_conditions[0].robin_coefficient"));

		json non_finite = missing;
		non_finite["boundary_conditions"][0]["robin_coefficient"] =
			std::numeric_limits<double>::infinity();

		ConfigValidator finite_validator;
		REQUIRE_FALSE(finite_validator.Validate(non_finite));
		REQUIRE(HasError(finite_validator,
						 "boundary_conditions[0].robin_coefficient"));
	}

	SECTION("rejects a Robin coefficient on non-Robin boundaries") {
		json config = ValidConfig();
		config["boundary_conditions"][0]["robin_coefficient"] = 1.0;

		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "boundary_conditions[0].robin_coefficient"));
	}
}

json ValidMqsConfig() {
	json config = ValidConfig();
	config["simulation"]["physics_type"] = "magnetoquasistatics";
	config["materials"][0]["properties"] = {{"mu_r", 1.0}, {"sigma", 0.0}};
	config["terminals"][0]["quantity"] = "current";
	config["terminals"][0]["entity_group"] = "Domain";
	config["scenarios"][0]["frequency"] = 60.0;
	return config;
}

bool HasError(const ConfigValidator& validator, const std::string& field) {
	for (const auto& error : validator.GetErrors()) {
		if (error.field == field) return true;
	}
	return false;
}

} // namespace

TEST_CASE("ConfigValidator validates open-current MQS regions",
		  "[config_validator][mqs][regions]") {
	SECTION("accepts a conductive open-current region") {
		json config = ValidMqsConfig();
		config["materials"][0]["properties"]["sigma"] = 1.0e6;
		config["regions"][0]["current_constraint"] = "open";
		config["terminals"] = json::array();
		config["scenarios"][0]["excitations"] = json::array();

		ConfigValidator validator;
		REQUIRE(validator.Validate(config));
	}

	SECTION("rejects unsupported values and physics") {
		json config = ValidConfig();
		config["regions"][0]["current_constraint"] = "fixed";

		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "regions[0].current_constraint"));
	}

	SECTION("rejects zero conductivity") {
		json config = ValidMqsConfig();
		config["regions"][0]["current_constraint"] = "open";
		config["terminals"] = json::array();
		config["scenarios"][0]["excitations"] = json::array();

		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "regions[0].current_constraint"));
	}

	SECTION("rejects overlap with a massive terminal") {
		json config = ValidMqsConfig();
		config["materials"][0]["properties"]["sigma"] = 1.0e6;
		config["regions"][0]["current_constraint"] = "open";
		config["terminals"] = json::array({
			{{"name", "Port"}, {"quantity", "current"},
			 {"conductor_type", "massive"}, {"entity_group", "Domain"}}
		});
		config["scenarios"][0]["excitations"] = json::array();

		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "regions[0].current_constraint"));
	}
}

TEST_CASE("ConfigValidator requires every meshed domain attribute to have a region",
		  "[config_validator][regions]") {
	// Two domain attributes in the mesh; the base config's single region covers
	// only attribute 1. An uncovered attribute gets no material, contributing zero
	// stiffness and making the assembled system singular, so it must be an error.
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
		2, 1, mfem::Element::QUADRILATERAL, true, 2.0, 1.0);
	mesh.SetAttribute(0, 1);
	mesh.SetAttribute(1, 2);
	mesh.SetAttributes();

	SECTION("rejects a domain attribute no region claims") {
		json config = ValidConfig();

		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config, &mesh));
		REQUIRE(HasError(validator, "regions"));
	}

	SECTION("accepts full coverage") {
		json config = ValidConfig();
		config["entity_groups"].push_back(
			{{"name", "Domain2"}, {"dim", 2}, {"attribute_ids", {2}}});
		config["regions"].push_back(
			{{"entity_group", "Domain2"}, {"material", "Dielectric"}});

		ConfigValidator validator;
		REQUIRE(validator.Validate(config, &mesh));
	}

	SECTION("is skipped without a mesh") {
		json config = ValidConfig();

		ConfigValidator validator;
		REQUIRE(validator.Validate(config));
	}
}

TEST_CASE("ConfigValidator rejects a domain attribute claimed by two regions",
		  "[config_validator][regions]") {
	// Attribute -> material resolution returns the first matching region, so two
	// regions claiming the same attribute would make the material silently depend
	// on declaration order. That must be an error, not a silent first-wins.
	SECTION("rejects a duplicate claim through a second entity group") {
		json config = ValidConfig();
		config["entity_groups"].push_back(
			{{"name", "DomainAgain"}, {"dim", 2}, {"attribute_ids", {1}}});
		config["materials"].push_back(
			{{"name", "Other"}, {"properties", {{"epsilon_r", 5.0}}}});
		config["regions"].push_back(
			{{"entity_group", "DomainAgain"}, {"material", "Other"}});

		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "regions[1].entity_group"));
	}

	SECTION("rejects two regions sharing one entity group") {
		json config = ValidConfig();
		config["regions"].push_back(config["regions"][0]);

		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "regions[1].entity_group"));
	}

	SECTION("accepts disjoint regions") {
		json config = ValidConfig();
		config["entity_groups"].push_back(
			{{"name", "Domain2"}, {"dim", 2}, {"attribute_ids", {2}}});
		config["regions"].push_back(
			{{"entity_group", "Domain2"}, {"material", "Dielectric"}});

		ConfigValidator validator;
		REQUIRE(validator.Validate(config));
	}
}

TEST_CASE("ConfigValidator validates material names",
		  "[config_validator][materials]") {
	SECTION("requires a name") {
		json config = ValidConfig();
		config["materials"][0].erase("name");

		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "materials[0].name"));
	}

	SECTION("rejects duplicate names") {
		json config = ValidConfig();
		config["materials"].push_back(
			{{"name", "Dielectric"}, {"properties", {{"epsilon_r", 3.0}}}});

		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "materials[1].name"));
	}

	SECTION("rejects a region referencing an undeclared material") {
		json config = ValidConfig();
		config["regions"][0]["material"] = "Missing";

		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "regions[0].material"));
	}

	SECTION("rejects a non-string material reference") {
		json config = ValidConfig();
		config["regions"][0]["material"] = 1;

		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "regions[0].material"));
	}
}

TEST_CASE("ConfigValidator accepts the canonical schema", "[config_validator]") {
	ConfigValidator validator;
	REQUIRE(validator.Validate(ValidConfig()));
	REQUIRE(validator.GetErrors().empty());
}

TEST_CASE("ConfigValidator enforces physics-specific terminal types",
		  "[config_validator][terminals]") {
	auto magnetic_config = [](const std::string& physics,
							 const std::string& excitation,
							 const std::string& conductor = "massive") {
		json config = ValidConfig();
		config["simulation"]["physics_type"] = physics;
		config["materials"][0]["properties"] = {
			{"mu_r", 1.0}, {"sigma", 5.8e7}
		};
		config["terminals"][0]["quantity"] = excitation;
		config["terminals"][0]["conductor_type"] = conductor;
		config["terminals"][0]["entity_group"] =
			excitation == "current" ? "Domain" : "Boundary";
		if (physics == "magnetoquasistatics") {
			config["scenarios"][0]["frequency"] = 1000.0;
		}
		return config;
	};

	SECTION("accepts electrostatic voltage terminals") {
		ConfigValidator validator;
		REQUIRE(validator.Validate(ValidConfig()));
	}

	SECTION("rejects electrostatic current terminals") {
		json config = ValidConfig();
		config["terminals"][0]["quantity"] = "current";
		config["terminals"][0]["entity_group"] = "Domain";

		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "terminals[0].quantity"));
	}

	SECTION("accepts magnetostatic current terminals") {
		ConfigValidator validator;
		REQUIRE(validator.Validate(
			magnetic_config("magnetostatics", "current")));
	}

	SECTION("rejects magnetostatic voltage terminals") {
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(
			magnetic_config("magnetostatics", "voltage")));
		REQUIRE(HasError(validator, "terminals[0].quantity"));
	}

	SECTION("accepts massive and stranded MQS current terminals") {
		for (const std::string conductor : {"massive", "stranded"}) {
			ConfigValidator validator;
			REQUIRE(validator.Validate(magnetic_config(
				"magnetoquasistatics", "current", conductor)));
		}
	}

	SECTION("rejects MQS voltage terminals") {
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(
			magnetic_config("magnetoquasistatics", "voltage")));
		REQUIRE(HasError(validator, "terminals[0].quantity"));
	}
}

TEST_CASE("ConfigValidator rejects wrong JSON types without throwing", "[config_validator]") {
	json config = ValidConfig();
	config["simulation"]["order"] = "first";
	config["entity_groups"][0]["attribute_ids"][0] = 1.5;
	config["scenarios"][0]["excitations"][0]["value"] = "ten";

	ConfigValidator validator;
	REQUIRE_NOTHROW(validator.Validate(config));
	REQUIRE_FALSE(validator.Validate(config));
	REQUIRE(HasError(validator, "simulation.order"));
	REQUIRE(HasError(validator, "entity_groups[0].attribute_ids[0]"));
	REQUIRE(HasError(validator, "scenarios[0].excitations[0].value"));
}

TEST_CASE("ConfigValidator enforces canonical simulation field names", "[config_validator]") {
	json config = ValidConfig();
	config["simulation"].erase("physics_type");
	config["simulation"]["physics"] = "electrostatics";
	config["simulation"]["geometry"] = "planar";
	config["simulation"]["results_file"] = "result.msh";

	ConfigValidator validator;
	REQUIRE_FALSE(validator.Validate(config));
	REQUIRE(HasError(validator, "simulation.physics"));
	REQUIRE(HasError(validator, "simulation.geometry"));
	REQUIRE(HasError(validator, "simulation.results_file"));
}

TEST_CASE("ConfigValidator validates output targets", "[config_validator][output]") {
	json config = ValidConfig();
	SECTION("accepts all formats and coupling fields") {
		config["output"] = {{"directory", "results"}, {"export_fields_for_coupling_matrix", true},
			{"paraview", json::object()}, {"gmsh", {{"version", "4.1"}}},
			{"hdf5", {{"file", "run.h5"}}}};
		ConfigValidator validator;
		REQUIRE(validator.Validate(config));
	}
	SECTION("rejects incorrect types without throwing") {
		config["output"] = {{"export_fields_for_coupling_matrix", "false"}, {"paraview", false},
			{"gmsh", {{"version", 4.1}}}, {"hdf5", {{"file", 1}}}};
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "output.export_fields_for_coupling_matrix"));
		REQUIRE(HasError(validator, "output.paraview"));
		REQUIRE(HasError(validator, "output.gmsh.version"));
		REQUIRE(HasError(validator, "output.hdf5.file"));
	}
	SECTION("rejects invalid paths versions and unknown options") {
		config["output"] = {{"directory", ""}, {"fields", true},
			{"gmsh", {{"version", "3.0"}}}, {"hdf5", {{"file", "folder/"}, {"filename", "x.h5"}}}};
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		for (const std::string field : {"output.directory", "output.fields", "output.gmsh.version",
			"output.hdf5.file", "output.hdf5.filename"}) REQUIRE(HasError(validator, field));
	}
	SECTION("rejects removed flat settings") {
		for (const std::string key : {"output_paraview", "output_gmsh", "output_hdf5",
			"gmsh_format", "results_path", "results_filename"}) {
			json legacy = config;
			legacy["simulation"][key] = true;
			ConfigValidator validator;
			REQUIRE_FALSE(validator.Validate(legacy));
			REQUIRE(HasError(validator, "simulation." + key));
		}
	}
}

TEST_CASE("ConfigValidator enforces canonical terminal field names", "[config_validator]") {
	for (const std::string legacy : {"excitation", "excitation_type"}) {
		json config = ValidConfig();
		config["terminals"][0].erase("quantity");
		config["terminals"][0][legacy] = "voltage";

		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "terminals[0]." + legacy));
		// The superseded spelling gets the actionable rename message on its own;
		// the generic "missing required field" error would only add noise here.
		REQUIRE_FALSE(HasError(validator, "terminals[0].quantity"));
	}
}

TEST_CASE("ConfigValidator enforces canonical schema section and field names", "[config_validator]") {
	SECTION("the superseded 'boundaries' section is rejected") {
		json config = ValidConfig();
		config["boundaries"] = config["boundary_conditions"];
		config.erase("boundary_conditions");

		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "boundaries"));
	}

	SECTION("the superseded entity group 'kind' field is rejected") {
		json config = ValidConfig();
		config["entity_groups"][0].erase("dim");
		config["entity_groups"][0]["kind"] = "domain";

		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "entity_groups[0].kind"));
	}

	SECTION("a dimension unusable in the mesh is rejected") {
		// The mesh is 2D, so only dim 2 (domain) and dim 1 (boundary) name
		// entities this solver can use. A volume group is neither.
		mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
			1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
		json config = ValidConfig();
		config["entity_groups"][0]["dim"] = 3;

		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config, &mesh));
		REQUIRE(HasError(validator, "entity_groups[0].dim"));
	}

	SECTION("capitalized boundary condition types name the lowercase form") {
		json config = ValidConfig();
		config["boundary_conditions"][0]["type"] = "Dirichlet";

		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "boundary_conditions[0].type"));
	}
}

TEST_CASE("ConfigValidator requires terminal quantity", "[config_validator]") {
	json config = ValidConfig();
	config["terminals"][0].erase("quantity");

	ConfigValidator validator;
	REQUIRE_FALSE(validator.Validate(config));
	REQUIRE(HasError(validator, "terminals[0].quantity"));
}

TEST_CASE("ConfigValidator rejects duplicate names", "[config_validator]") {
	SECTION("entity groups") {
		json config = ValidConfig();
		config["entity_groups"].push_back(
			{{"name", "Domain"}, {"dim", 2}, {"attribute_ids", {2}}});
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "entity_groups[2].name"));
	}

	SECTION("terminals") {
		json config = ValidConfig();
		config["terminals"].push_back(
			{{"name", "Drive"}, {"quantity", "voltage"}, {"entity_group", "Boundary"}});
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "terminals[1].name"));
	}

	SECTION("scenarios") {
		json config = ValidConfig();
		config["scenarios"].push_back(
			{{"name", "Driven"}, {"excitations", json::array()}});
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "scenarios[1].name"));
	}
}

TEST_CASE("ConfigValidator checks entity group references and kinds", "[config_validator]") {
	SECTION("unknown group") {
		json config = ValidConfig();
		config["regions"][0]["entity_group"] = "Missing";
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "regions[0].entity_group"));
	}

	SECTION("wrong group kind") {
		json config = ValidConfig();
		config["terminals"][0]["entity_group"] = "Domain";
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "terminals[0].entity_group"));
	}

	SECTION("unknown terminal") {
		json config = ValidConfig();
		config["scenarios"][0]["excitations"][0]["terminal"] = "Missing";
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "scenarios[0].excitations[0].terminal"));
	}
}

TEST_CASE("ConfigValidator rejects invalid ranges and enum values", "[config_validator]") {
	json config = ValidConfig();
	config["simulation"]["order"] = 0;
	config["simulation"]["amr"] = {
		{"max_iterations", -1},
		{"error_fraction", 1.5},
		{"error_tolerance", -1.0}
	};
	config["entity_groups"][0]["dim"] = 7;
	config["materials"][0]["properties"]["sigma"] = -1.0;
	config["terminals"][0]["conductor_type"] = "unknown";

	ConfigValidator validator;
	REQUIRE_FALSE(validator.Validate(config));
	REQUIRE(HasError(validator, "simulation.order"));
	REQUIRE(HasError(validator, "simulation.amr.max_iterations"));
	REQUIRE(HasError(validator, "simulation.amr.error_fraction"));
	REQUIRE(HasError(validator, "simulation.amr.error_tolerance"));
	REQUIRE(HasError(validator, "entity_groups[0].dim"));
	REQUIRE(HasError(validator, "materials[0].properties.sigma"));
	REQUIRE(HasError(validator, "terminals[0].conductor_type"));
}

TEST_CASE("ConfigValidator enforces MQS scenario frequencies", "[config_validator][mqs]") {
	SECTION("accepts scalar and sweep frequencies") {
		json config = ValidMqsConfig();
		config["scenarios"].push_back({
			{"name", "Sweep"},
			{"frequency", {{"scale", "log"}, {"start", 10.0},
						   {"stop", 1000.0}, {"points", 3}}},
			{"excitations", json::array()}
		});
		ConfigValidator validator;
		REQUIRE(validator.Validate(config));
	}

	SECTION("requires frequency on every scenario") {
		json config = ValidMqsConfig();
		config["scenarios"][0].erase("frequency");
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "scenarios[0].frequency"));
	}

	SECTION("rejects obsolete simulation frequency") {
		json config = ValidMqsConfig();
		config["simulation"]["frequency"] = 60.0;
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "simulation.frequency"));
	}

	SECTION("rejects invalid sweep values") {
		json config = ValidMqsConfig();
		config["scenarios"][0]["frequency"] = {
			{"scale", "octave"}, {"start", -1.0}, {"stop", -2.0}, {"points", 0}
		};
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "scenarios[0].frequency.scale"));
		REQUIRE(HasError(validator, "scenarios[0].frequency.start"));
		REQUIRE(HasError(validator, "scenarios[0].frequency.stop"));
		REQUIRE(HasError(validator, "scenarios[0].frequency.points"));
	}

	SECTION("rejects descending ranges") {
		json config = ValidMqsConfig();
		config["scenarios"][0]["frequency"] = {
			{"scale", "linear"}, {"start", 100.0}, {"stop", 10.0}, {"points", 2}
		};
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "scenarios[0].frequency.stop"));
	}

	SECTION("does not require frequency for other physics") {
		ConfigValidator validator;
		REQUIRE(validator.Validate(ValidConfig()));
	}
}

TEST_CASE("ConfigValidator checks geometry_type against the mesh and physics",
		  "[config_validator][geometry][3d]") {
	mfem::Mesh square = mfem::Mesh::MakeCartesian2D(
		1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
	mfem::Mesh cube = mfem::Mesh::MakeCartesian3D(
		1, 1, 1, mfem::Element::TETRAHEDRON, 1.0, 1.0, 1.0);

	// ValidConfig() is written for a 2D mesh; lift its groups one dimension.
	auto three_d_config = []() {
		json config = ValidConfig();
		config["simulation"]["geometry_type"] = "3d";
		config["entity_groups"][0]["dim"] = 3;
		config["entity_groups"][1]["dim"] = 2;
		return config;
	};

	SECTION("accepts '3d' electrostatics on a 3D mesh") {
		ConfigValidator validator;
		REQUIRE(validator.Validate(three_d_config(), &cube));
	}

	SECTION("rejects a 2D geometry_type on a 3D mesh") {
		json config = three_d_config();
		config["simulation"]["geometry_type"] = "planar";
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config, &cube));
		REQUIRE(HasError(validator, "simulation.geometry_type"));
	}

	SECTION("the planar default also rejects a 3D mesh") {
		json config = three_d_config();
		config["simulation"].erase("geometry_type");
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config, &cube));
		REQUIRE(HasError(validator, "simulation.geometry_type"));
	}

	SECTION("rejects '3d' on a 2D mesh") {
		json config = ValidConfig();
		config["simulation"]["geometry_type"] = "3d";
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config, &square));
		REQUIRE(HasError(validator, "simulation.geometry_type"));
	}

	SECTION("accepts '3d' magnetics") {
		for (const char* physics : { "magnetostatics", "magnetoquasistatics" }) {
			json config = three_d_config();
			config["simulation"]["physics_type"] = physics;
			ConfigValidator validator;
			validator.Validate(config);
			REQUIRE_FALSE(HasError(validator, "simulation.geometry_type"));
		}
	}

	SECTION("rejects region current constraints in '3d'") {
		json config = three_d_config();
		config["simulation"]["physics_type"] = "magnetoquasistatics";
		config["regions"][0]["current_constraint"] = "open";
		config["materials"][0]["properties"]["sigma"] = 1e6;
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		bool rejected_for_3d = false;
		for (const auto& error : validator.GetErrors()) {
			rejected_for_3d |= error.field == "regions[0].current_constraint" &&
				error.message.find("'3d'") != std::string::npos;
		}
		REQUIRE(rejected_for_3d);
	}

	SECTION("accepts Gmsh and ParaView output for '3d'") {
		json config = three_d_config();
		config["output"] = {{"gmsh", {{"directory", "msh"}}},
							{"paraview", {{"directory", "vtk"}}}};
		ConfigValidator validator;
		REQUIRE(validator.Validate(config, &cube));
	}

	SECTION("rejects an unknown geometry_type") {
		json config = ValidConfig();
		config["simulation"]["geometry_type"] = "spherical";
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "simulation.geometry_type"));
	}
}

TEST_CASE("ConfigValidator checks linear_solver", "[config_validator][linear_solver]") {
	for (const char* ok : {"direct", "iterative"}) {
		json config = ValidConfig();
		config["simulation"]["linear_solver"] = ok;
		ConfigValidator validator;
		REQUIRE(validator.Validate(config));
	}

	json misspelled = ValidConfig();
	misspelled["simulation"]["linear_solver"] = "amg";
	ConfigValidator validator;
	REQUIRE_FALSE(validator.Validate(misspelled));
	REQUIRE(HasError(validator, "simulation.linear_solver"));

	json wrong_type = ValidConfig();
	wrong_type["simulation"]["linear_solver"] = 1;
	ConfigValidator type_validator;
	REQUIRE_FALSE(type_validator.Validate(wrong_type));
	REQUIRE(HasError(type_validator, "simulation.linear_solver"));
}

TEST_CASE("ConfigValidator checks 3D coil directions", "[config_validator][3d][coil]") {
	mfem::Mesh cube = mfem::Mesh::MakeCartesian3D(
		1, 1, 1, mfem::Element::TETRAHEDRON, 1.0, 1.0, 1.0);
	auto coil_config = []() {
		json config = ValidConfig();
		config["simulation"]["physics_type"] = "magnetostatics";
		config["simulation"]["geometry_type"] = "3d";
		config["entity_groups"][0]["dim"] = 3;
		config["entity_groups"][1]["dim"] = 2;
		config["materials"][0]["properties"] = {{"mu_r", 1.0}};
		config["terminals"] = json::array({
			{{"name", "Coil"}, {"quantity", "current"}, {"entity_group", "Domain"},
			 {"direction", {{"type", "azimuthal"}, {"origin", {0.0, 0.0, 0.0}},
							{"axis", {0.0, 0.0, 1.0}}}}}});
		config["scenarios"][0]["excitations"][0]["terminal"] = "Coil";
		return config;
	};

	SECTION("accepts a well-formed azimuthal coil") {
		ConfigValidator validator;
		INFO(validator.GetErrorMessage());
		REQUIRE(validator.Validate(coil_config(), &cube));
	}

	SECTION("requires a direction on 3D magnetic current terminals") {
		json config = coil_config();
		config["terminals"][0].erase("direction");
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config, &cube));
		REQUIRE(HasError(validator, "terminals[0].direction"));
	}

	SECTION("rejects a direction in a 2D model") {
		json config = coil_config();
		config["simulation"]["geometry_type"] = "planar";
		config["entity_groups"][0]["dim"] = 2;
		config["entity_groups"][1]["dim"] = 1;
		ConfigValidator validator;
		REQUIRE_FALSE(validator.Validate(config));
		REQUIRE(HasError(validator, "terminals[0].direction"));
	}

	SECTION("rejects malformed directions") {
		auto error_for = [&](const std::function<void(json&)>& mutate, const std::string& field) {
			json config = coil_config();
			mutate(config["terminals"][0]["direction"]);
			ConfigValidator validator;
			REQUIRE_FALSE(validator.Validate(config, &cube));
			REQUIRE(HasError(validator, field));
		};
		error_for([](json& d) { d["type"] = "toroidal"; }, "terminals[0].direction.type");
		error_for([](json& d) { d.erase("axis"); }, "terminals[0].direction.axis");
		error_for([](json& d) { d["axis"] = {0.0, 0.0, 0.0}; }, "terminals[0].direction.axis");
		error_for([](json& d) { d["origin"] = {1.0, 2.0}; }, "terminals[0].direction.origin");
		error_for([](json& d) { d["center"] = {0.0, 0.0, 0.0}; }, "terminals[0].direction.center");
	}
}
