// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <highfive/H5File.hpp>
#include "io/coupling_matrix_writer.hpp"
#include "io/solver_field_writer.hpp"
#include "io/hdf5_results_writer.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

namespace {

struct TemporaryHdf5File {
	std::filesystem::path path = std::filesystem::temp_directory_path()
		/ ("mfem_em_hdf5_" + std::to_string(
			std::chrono::steady_clock::now().time_since_epoch().count()) + ".h5");

	~TemporaryHdf5File() {
		std::error_code error;
		std::filesystem::remove(path, error);
	}
};

} // namespace

TEST_CASE("HighFive round-trips field and matrix datasets", "[hdf5][io]") {
	const TemporaryHdf5File temporary;
	const std::vector<double> field{0.0, 1.25, -2.5, 4.0};
	const std::vector<std::vector<double>> matrix{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0}
	};

	{
		HighFive::File file(temporary.path.string(), HighFive::File::Truncate);
		file.createGroup("fields").createDataSet("potential", field);
		file.createGroup("matrices").createDataSet("coupling", matrix);
	}

	{
		HighFive::File file(temporary.path.string(), HighFive::File::ReadOnly);
		const auto field_dataset = file.getDataSet("fields/potential");
		const auto matrix_dataset = file.getDataSet("matrices/coupling");
		REQUIRE(field_dataset.getDimensions() == std::vector<std::size_t>{field.size()});
		REQUIRE(matrix_dataset.getDimensions() == (std::vector<std::size_t>{2, 3}));

		std::vector<double> actual_field;
		std::vector<std::vector<double>> actual_matrix;
		field_dataset.read(actual_field);
		matrix_dataset.read(actual_matrix);
		REQUIRE(actual_field == field);
		REQUIRE(actual_matrix == matrix);
	}
}

TEST_CASE("Coupling HDF5 stores labeled row-major doubles and metadata", "[hdf5][io][coupling]") {
	const TemporaryHdf5File temporary;
	const std::vector<std::string> terminals{"Coil/A", "Coil,B"};
	mfem::DenseMatrix matrix(2);
	matrix(0, 0) = 1.2345678901234567;
	matrix(0, 1) = -2.5;
	matrix(1, 0) = 3.75;
	matrix(1, 1) = 4.0;
	{
		matrix_io::CouplingMatrixWriter writer(
			temporary.path, terminals, "electrostatics", "planar");
		writer.WriteMatrix("Capacitance", matrix, "F/m");
	}
	{
		HighFive::File file(temporary.path.string(), HighFive::File::ReadOnly);
		int version = 0;
		file.getAttribute("schema_version").read(version);
		REQUIRE(version == 2);
		auto group = file.getGroup("/coupling");
		std::string physics, geometry, units;
		group.getAttribute("physics_type").read(physics);
		group.getAttribute("geometry_type").read(geometry);
		REQUIRE(physics == "electrostatics");
		REQUIRE(geometry == "planar");
		std::vector<std::string> labels;
		group.getDataSet("terminal_names").read(labels);
		REQUIRE(labels == terminals);
		auto dataset = group.getDataSet("Capacitance/values");
		REQUIRE(dataset.getDimensions() == std::vector<std::size_t>{2, 2});
		REQUIRE(dataset.getDataType().getSize() == sizeof(double));
		dataset.getAttribute("units").read(units);
		REQUIRE(units == "F/m");
		std::vector<std::vector<double>> values;
		dataset.read(values);
		REQUIRE(values == std::vector<std::vector<double>>{
			{matrix(0, 0), -2.5}, {3.75, 4.0}});
		REQUIRE_FALSE(group.exist("frequency_hz"));
	}
}

TEST_CASE("Coupling HDF5 sweep aligns numeric frequencies with both matrices", "[hdf5][io][coupling]") {
	const TemporaryHdf5File temporary;
	mfem::DenseMatrix low_r(2), low_l(2), high_r(2), high_l(2);
	for (int row = 0; row < 2; ++row) {
		for (int column = 0; column < 2; ++column) {
			low_r(row, column) = 10.0 * row + column;
			low_l(row, column) = 100.0 + 10.0 * row + column;
			high_r(row, column) = 200.0 + 10.0 * row + column;
			high_l(row, column) = 300.0 + 10.0 * row + column;
		}
	}
	const double low = 100.0;
	const double high = std::nextafter(low, std::numeric_limits<double>::infinity());
	{
		matrix_io::CouplingMatrixWriter writer(
			temporary.path, {"A", "B"}, "magnetoquasistatics", "axisymmetric");
		writer.WriteFrequencies({low, high});
		writer.WriteMatrixSeries("Inductance", {&low_l, &high_l}, "H");
		writer.WriteMatrixSeries("Resistance", {&low_r, &high_r}, "Ohm");
	}
	{
		HighFive::File file(temporary.path.string(), HighFive::File::ReadOnly);
		auto group = file.getGroup("/coupling");
		std::vector<double> frequencies;
		auto frequency_dataset = group.getDataSet("frequency_hz");
		frequency_dataset.read(frequencies);
		REQUIRE(frequencies == std::vector<double>{low, high});
		std::string units;
		frequency_dataset.getAttribute("units").read(units);
		REQUIRE(units == "Hz");
		for (const std::string quantity : {"Resistance", "Inductance"}) {
			auto dataset = group.getDataSet(quantity + "/values");
			REQUIRE(dataset.getDimensions() == std::vector<std::size_t>{2, 2, 2});
			dataset.getAttribute("units").read(units);
			REQUIRE(units == (quantity == "Resistance" ? "Ohm" : "H"));
			std::vector<double> flat(8);
			dataset.read_raw(flat.data());
			const double offset = quantity == "Resistance" ? 0.0 : 100.0;
			REQUIRE(flat == std::vector<double>{offset, offset + 1, offset + 10,
				offset + 11, offset + 200, offset + 201, offset + 210, offset + 211});
		}
	}
}

TEST_CASE("Coupling HDF5 rejects inconsistent matrix axes and frequencies", "[hdf5][io][coupling]") {
	const TemporaryHdf5File temporary;
	mfem::DenseMatrix matrix(1), wrong_size(2);
	matrix = 1.0;
	matrix_io::CouplingMatrixWriter writer(
		temporary.path, {"Port"}, "magnetoquasistatics", "planar");
	SECTION("dimensions") {
		REQUIRE_THROWS(writer.WriteMatrix("CustomQuantity", wrong_size, "H/m"));
	}
	SECTION("duplicate frequency") {
		REQUIRE_THROWS(writer.WriteFrequencies({100.0, 100.0}));
	}
	SECTION("non-finite frequency") {
		REQUIRE_THROWS(writer.WriteFrequencies({std::numeric_limits<double>::infinity()}));
	}
	SECTION("empty sweep") {
		REQUIRE_THROWS(writer.WriteFrequencies({}));
	}
	SECTION("matrix count") {
		writer.WriteFrequencies({100.0, 200.0});
		REQUIRE_THROWS(writer.WriteMatrixSeries("CustomQuantity", {&matrix}, "custom units"));
	}
	SECTION("null matrix") {
		writer.WriteFrequencies({100.0});
		REQUIRE_THROWS(writer.WriteMatrixSeries("CustomQuantity", {nullptr}, "custom units"));
	}
}

TEST_CASE("Field HDF5 writes numeric primary and projected derived data", "[hdf5][io][fields]") {
	const TemporaryHdf5File temporary;
	auto mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL);
	mfem::H1_FECollection collection(2, 2);
	mfem::FiniteElementSpace space(&mesh, &collection);
	mfem::GridFunction primary(&space);
	for (int i = 0; i < primary.Size(); ++i) primary(i) = i + 0.25;
	FieldExportSet fields;
	fields.AddPrimary("potential", primary);
	fields.AddScalar("magnitude", std::make_unique<mfem::ConstantCoefficient>(3.5));
	mfem::Vector constant(3);
	constant(0) = 1.25;
	constant(1) = -2.5;
	constant(2) = 4.0;
	fields.AddVector("vector_field", std::make_unique<mfem::VectorConstantCoefficient>(constant));
	ProblemConfig config;
	config.Order = 2;
	Hdf5ResultsWriter writer(temporary.path, mesh, config);
	Scenario scenario;
	scenario.Frequency = 50.0;
	writer.WriteScenario("scenario_000000", "Drive/A", scenario, fields);
	writer.WriteScenario("scenario_000001", "Drive/A", scenario, fields);
	mfem::DenseMatrix matrix(1);
	matrix = 2.5;
	matrix_io::CouplingMatrixWriter coupling(writer.File(), {"Drive"}, "electrostatics", "planar");
	coupling.WriteMatrix("Capacitance", matrix, "F/m");

	HighFive::File file(temporary.path.string(), HighFive::File::ReadOnly);
	REQUIRE(file.exist("coupling/Capacitance/values"));
	REQUIRE(file.getGroup("scenarios").getNumberObjects() == 2);
	std::string mesh_text;
	file.getDataSet("mesh/mfem").read(mesh_text);
	std::istringstream mesh_stream(mesh_text);
	mfem::Mesh restored_mesh(mesh_stream, 1, 0, false);
	REQUIRE(restored_mesh.GetNE() == mesh.GetNE());
	REQUIRE(restored_mesh.GetNV() == mesh.GetNV());
	const auto output_fields = file.getGroup("scenarios/scenario_000000/fields");
	std::vector<double> values;
	output_fields.getDataSet("potential/values").read(values);
	REQUIRE(values.size() == primary.Size());
	for (int i = 0; i < primary.Size(); ++i) REQUIRE(values[i] == primary(i));
	std::string primary_collection;
	output_fields.getDataSet("potential/values").getAttribute("finite_element_collection").read(primary_collection);
	std::unique_ptr<mfem::FiniteElementCollection> restored_collection(
		mfem::FiniteElementCollection::New(primary_collection.c_str()));
	mfem::FiniteElementSpace restored_space(&restored_mesh, restored_collection.get());
	mfem::GridFunction restored(&restored_space);
	REQUIRE(restored.Size() == values.size());
	for (int index = 0; index < restored.Size(); ++index) restored(index) = values[index];
	mfem::IntegrationPoint point;
	point.Set2(0.37, 0.61);
	REQUIRE(std::abs(restored.GetValue(0, point) - primary.GetValue(0, point)) < 1.0e-12);

	mfem::L2_FECollection l2_collection(1, 2);
	mfem::FiniteElementSpace scalar_space(&mesh, &l2_collection);
	output_fields.getDataSet("magnitude/values").read(values);
	REQUIRE(values == std::vector<double>(scalar_space.GetVSize(), 3.5));

	mfem::FiniteElementSpace vector_space(&mesh, &l2_collection, 3);
	auto dataset = output_fields.getDataSet("vector_field/values");
	dataset.read(values);
	REQUIRE(values.size() == vector_space.GetVSize());
	for (int dof = 0; dof < vector_space.GetNDofs(); ++dof) {
		for (int component = 0; component < 3; ++component) {
			REQUIRE(values[vector_space.DofToVDof(dof, component)] == constant(component));
		}
	}
	int dimension = 0, ordering = -1;
	std::string collection_name;
	dataset.getAttribute("vector_dimension").read(dimension);
	dataset.getAttribute("ordering").read(ordering);
	dataset.getAttribute("finite_element_collection").read(collection_name);
	REQUIRE(dimension == 3);
	REQUIRE(ordering == static_cast<int>(vector_space.GetOrdering()));
	REQUIRE(collection_name == l2_collection.Name());
}

TEST_CASE("Field HDF5 spaces and curved mesh nodes describe each element's DOFs", "[hdf5][io][fields]") {
	const TemporaryHdf5File temporary;
	auto mesh = mfem::Mesh::MakeCartesian2D(3, 2, mfem::Element::TRIANGLE);
	// Curve the mesh so the geometry is genuinely high order.
	mesh.SetCurvature(3);
	mfem::GridFunction& geometry = *mesh.GetNodes();
	mfem::VectorFunctionCoefficient bend(2, [](const mfem::Vector& x, mfem::Vector& y) {
		y(0) = x(0) + 0.05 * std::sin(3.0 * x(1));
		y(1) = x(1) + 0.05 * std::sin(2.0 * x(0));
	});
	geometry.ProjectCoefficient(bend);

	// Order 3 has two DOFs per edge, so a wrong edge orientation would show.
	mfem::H1_FECollection collection(3, 2);
	mfem::FiniteElementSpace space(&mesh, &collection);
	mfem::GridFunction primary(&space);
	mfem::FunctionCoefficient smooth([](const mfem::Vector& x) { return std::cos(x(0)) * std::exp(x(1)); });
	primary.ProjectCoefficient(smooth);
	FieldExportSet fields;
	fields.AddPrimary("potential", primary);
	ProblemConfig config;
	config.Order = 3;
	Hdf5ResultsWriter writer(temporary.path, mesh, config);
	Scenario scenario;
	writer.WriteScenario("scenario_000000", "Base", scenario, fields);

	HighFive::File file(temporary.path.string(), HighFive::File::ReadOnly);

	// Checks the documented contract: the coefficient that element e's DOF k names
	// is the field's value at reference node k of element e.
	auto check = [&](const std::string& values_path, const mfem::GridFunction& field) {
		auto dataset = file.getDataSet(values_path);
		std::vector<double> values;
		dataset.read(values);
		std::string space_path;
		int vdim = 0, ordering = -1;
		dataset.getAttribute("space").read(space_path);
		dataset.getAttribute("vector_dimension").read(vdim);
		dataset.getAttribute("ordering").read(ordering);
		REQUIRE(vdim == field.VectorDim());
		const auto space_group = file.getGroup(space_path);
		std::vector<int> offsets, dofs;
		space_group.getDataSet("element_dof_offsets").read(offsets);
		space_group.getDataSet("element_dofs").read(dofs);
		std::vector<std::vector<double>> nodes;
		space_group.getDataSet("reference_nodes/" + std::to_string(mfem::Geometry::TRIANGLE)).read(nodes);
		const int ndofs = static_cast<int>(values.size()) / vdim;
		REQUIRE(offsets.size() == static_cast<std::size_t>(mesh.GetNE() + 1));
		mfem::Vector expected;
		for (int element = 0; element < mesh.GetNE(); ++element) {
			REQUIRE(offsets[element + 1] - offsets[element] == static_cast<int>(nodes.size()));
			for (int k = 0; k < static_cast<int>(nodes.size()); ++k) {
				mfem::IntegrationPoint point;
				point.Set2(nodes[k][0], nodes[k][1]);
				field.GetVectorValue(element, point, expected);
				const int dof = dofs[offsets[element] + k];
				REQUIRE(dof >= 0);
				for (int c = 0; c < vdim; ++c) {
					const double stored = ordering == mfem::Ordering::byNODES
						? values[c * ndofs + dof] : values[dof * vdim + c];
					REQUIRE(std::abs(stored - expected(c)) < 1.0e-12);
				}
			}
		}
	};

	check("scenarios/scenario_000000/fields/potential/values", primary);
	REQUIRE(file.exist("mesh/nodes/values"));
	check("mesh/nodes/values", geometry);
}
