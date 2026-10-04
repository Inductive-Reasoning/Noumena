// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "linalg/complex_block_layout.hpp"
#include "solvers/mqs_massive_port_operator.hpp"

namespace {

std::unique_ptr<mfem::SparseMatrix> DiagonalMatrix(
	std::initializer_list<mfem::real_t> diagonal)
{
	auto matrix = std::make_unique<mfem::SparseMatrix>(static_cast<int>(diagonal.size()));
	int row = 0;
	for (const mfem::real_t value : diagonal) {
		matrix->Set(row, row, value);
		++row;
	}
	matrix->Finalize();
	return matrix;
}

} // namespace

TEST_CASE("Complex port layout names every packed block", "[mqs][layout]")
{
	ComplexPortLayout layout(3, 2);

	REQUIRE(layout.NDofs() == 3);
	REQUIRE(layout.NPorts() == 2);
	REQUIRE(layout.HalfSize() == 5);
	REQUIRE(layout.FullSize() == 10);

	REQUIRE(layout.ReMeshIndex(2) == 2);
	REQUIRE(layout.RePortIndex(1) == 4);
	REQUIRE(layout.ImMeshIndex(2) == 7);
	REQUIRE(layout.ImPortIndex(1) == 9);

	mfem::Vector packed(layout.FullSize());
	packed = 0.0;
	ComplexPortVectorView values(packed, layout.NDofs(), layout.NPorts());
	values.ReMesh(2) = 1.0;
	values.RePort(1) = 2.0;
	values.ImMesh(2) = 3.0;
	values.ImPort(1) = 4.0;

	ConstComplexPortVectorView read_only(packed, layout.NDofs(), layout.NPorts());
	REQUIRE(read_only.ReMesh(2) == 1.0);
	REQUIRE(read_only.RePort(1) == 2.0);
	REQUIRE(read_only.ImMesh(2) == 3.0);
	REQUIRE(read_only.ImPort(1) == 4.0);
}

TEST_CASE("Complex port layout supports a field without ports", "[mqs][layout]")
{
	ComplexPortLayout layout(3, 0);

	REQUIRE(layout.NPorts() == 0);
	REQUIRE(layout.HalfSize() == 3);
	REQUIRE(layout.FullSize() == 6);
	REQUIRE(layout.ImMeshIndex(2) == 5);
}

TEST_CASE("Complex essential DOFs constrain both field copies", "[mqs][layout]")
{
	mfem::Array<int> field_tdofs({ 1, 4 });
	const mfem::Array<int> packed_tdofs = ComplexEssentialTDofs(field_tdofs, 7);

	REQUIRE(packed_tdofs.Size() == 4);
	REQUIRE(packed_tdofs[0] == 1);
	REQUIRE(packed_tdofs[1] == 8);
	REQUIRE(packed_tdofs[2] == 4);
	REQUIRE(packed_tdofs[3] == 11);
}

TEST_CASE("MQS massive-port operator preserves the coupled block equation", "[mqs][operator]")
{
	auto K = DiagonalMatrix({ 2.0, 3.0 });
	auto M = DiagonalMatrix({ 5.0, 7.0 });

	std::vector<std::unique_ptr<mfem::Vector>> port_loads;
	auto load = std::make_unique<mfem::Vector>(2);
	(*load)(0) = 11.0;
	(*load)(1) = 13.0;
	port_loads.push_back(std::move(load));

	MqsMassivePortOperator coupled(
		2, *K, *M, std::move(port_loads), { 0.25 }, 1.0);

	mfem::Vector x(coupled.Layout().FullSize());
	x(0) = 1.0;
	x(1) = 2.0;
	x(2) = 3.0;
	x(3) = 4.0;
	x(4) = 5.0;
	x(5) = 6.0;

	mfem::Vector y(coupled.Layout().FullSize());
	coupled.Operator().Mult(x, y);

	REQUIRE(y(0) == Catch::Approx(-51.0));
	REQUIRE(y(1) == Catch::Approx(-68.0));
	REQUIRE(y(2) == Catch::Approx(-35.5));
	REQUIRE(y(3) == Catch::Approx(-53.0));
	REQUIRE(y(4) == Catch::Approx(-49.0));
	REQUIRE(y(5) == Catch::Approx(-109.75));
	mfem::Array<int> essential;
	REQUIRE_NOTHROW(coupled.RequireAccepted(y, y, x, essential, 1e-12, "port test"));
	mfem::Vector physical_rhs(y);
	coupled.View(physical_rhs).ImPort(0) += 1.0;
	REQUIRE_THROWS(coupled.RequireAccepted(y, physical_rhs, x, essential, 1e-12, "port test"));
}

TEST_CASE("MQS massive-port operator degenerates to the complex field operator", "[mqs][operator]")
{
	auto K = DiagonalMatrix({ 2.0, 3.0 });
	auto M = DiagonalMatrix({ 5.0, 7.0 });
	std::vector<std::unique_ptr<mfem::Vector>> no_port_loads;

	MqsMassivePortOperator coupled(2, *K, *M, std::move(no_port_loads), {}, 1.0);

	mfem::Vector x(coupled.Layout().FullSize());
	x(0) = 1.0;
	x(1) = 2.0;
	x(2) = 4.0;
	x(3) = 5.0;

	mfem::Vector y(coupled.Layout().FullSize());
	coupled.Operator().Mult(x, y);

	REQUIRE(y(0) == Catch::Approx(-18.0));
	REQUIRE(y(1) == Catch::Approx(-29.0));
	REQUIRE(y(2) == Catch::Approx(13.0));
	REQUIRE(y(3) == Catch::Approx(29.0));
}

// The direct solver factors an explicitly assembled copy of a system that is
// otherwise only ever applied matrix-free, so the two must agree exactly. A
// sign error in the packed assembly would otherwise surface as a plausible but
// wrong solution rather than as a failure.
TEST_CASE("MQS packed matrix reproduces the matrix-free operator", "[mqs][operator]")
{
	auto K = DiagonalMatrix({ 2.0, 3.0 });
	auto M = DiagonalMatrix({ 5.0, 7.0 });

	std::vector<std::unique_ptr<mfem::Vector>> port_loads;
	auto load = std::make_unique<mfem::Vector>(2);
	(*load)(0) = 11.0;
	(*load)(1) = 13.0;
	port_loads.push_back(std::move(load));

	MqsMassivePortOperator coupled(
		2, *K, *M, std::move(port_loads), { 0.25 }, 1.0);

	// A frequency other than the construction value, to confirm the packed
	// assembly picks up both omega-dependent blocks after SetOmega.
	coupled.SetOmega(3.0);

	auto packed = coupled.AssemblePackedMatrix();
	const int n = coupled.Layout().FullSize();
	REQUIRE(packed->Height() == n);

	// Compare column by column: the j-th column of each operator is its action
	// on the j-th basis vector.
	for (int j = 0; j < n; ++j) {
		mfem::Vector e(n);
		e = 0.0;
		e(j) = 1.0;

		mfem::Vector expected(n), actual(n);
		coupled.Operator().Mult(e, expected);
		packed->Mult(e, actual);

		for (int i = 0; i < n; ++i) {
			REQUIRE(actual(i) == Catch::Approx(expected(i)).margin(1e-12));
		}
	}
}
