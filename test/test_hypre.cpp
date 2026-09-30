// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// The MPI/HYPRE toolchain, end to end, on a single-rank ParMesh: BoomerAMG on
// a 3D Laplacian and AMS on the singular 3D magnetostatic curl-curl problem.
// These exercise exactly what the parallel solver paths will build on. In a
// serial build the file compiles to one placeholder test.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "mfem.hpp"
#include "parallel/mpi_runtime.hpp"

#ifdef MFEM_USE_MPI

#include <cmath>

#include "linalg/sparse_direct_solver.hpp"

namespace {

// Dirichlet data g on every face; interior Poisson source 1.
struct PoissonResult { double energy; int iterations; };

PoissonResult SolveBoomerAmgPoisson(int n) {
	mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(n, n, n, mfem::Element::TETRAHEDRON);
	auto mesh = parallel::MakeSingleRankParMesh(serial);
	mfem::H1_FECollection fec(2, 3);
	mfem::ParFiniteElementSpace fes(mesh.get(), &fec);

	mfem::Array<int> ess_bdr(mesh->bdr_attributes.Max());
	ess_bdr = 1;
	mfem::Array<int> ess_tdofs;
	fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);

	mfem::ConstantCoefficient one(1.0);
	mfem::ParLinearForm b(&fes);
	b.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
	b.Assemble();
	mfem::ParBilinearForm a(&fes);
	a.AddDomainIntegrator(new mfem::DiffusionIntegrator);
	a.Assemble();

	mfem::ParGridFunction x(&fes);
	x = 0.0;
	mfem::HypreParMatrix A;
	mfem::Vector X, B;
	a.FormLinearSystem(ess_tdofs, x, b, A, X, B);

	mfem::HypreBoomerAMG amg(A);
	amg.SetPrintLevel(0);
	mfem::CGSolver cg(MPI_COMM_WORLD);
	cg.SetOperator(A);
	cg.SetPreconditioner(amg);
	cg.SetRelTol(1e-12);
	cg.SetMaxIter(500);
	cg.Mult(B, X);
	REQUIRE(cg.GetConverged());
	return { B * X, cg.GetNumIterations() };
}

} // namespace

// BoomerAMG-preconditioned CG must reproduce a serial direct solve of the same
// problem (compared through the energy b.x, which does not depend on DOF
// numbering), with an iteration count that does not grow with refinement.
TEST_CASE("HYPRE BoomerAMG solves a 3D Laplacian", "[hypre][mpi]") {
	const PoissonResult coarse = SolveBoomerAmgPoisson(4);
	const PoissonResult fine = SolveBoomerAmgPoisson(8);
	INFO("iterations: " << coarse.iterations << " (n=4), " << fine.iterations << " (n=8)");
	REQUIRE(fine.iterations < 40);
	REQUIRE(fine.iterations <= coarse.iterations + 10);

	// Serial reference on the coarse mesh.
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::TETRAHEDRON);
	mfem::H1_FECollection fec(2, 3);
	mfem::FiniteElementSpace fes(&mesh, &fec);
	mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
	ess_bdr = 1;
	mfem::Array<int> ess_tdofs;
	fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);
	mfem::ConstantCoefficient one(1.0);
	mfem::LinearForm b(&fes);
	b.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
	b.Assemble();
	mfem::BilinearForm a(&fes);
	a.AddDomainIntegrator(new mfem::DiffusionIntegrator);
	a.Assemble();
	mfem::GridFunction x(&fes);
	x = 0.0;
	mfem::OperatorPtr A;
	mfem::Vector X, B;
	a.FormLinearSystem(ess_tdofs, x, b, A, X, B);
	SparseDirectSolver direct(*dynamic_cast<mfem::SparseMatrix*>(A.Ptr()));
	direct.Mult(B, X);
	REQUIRE(coarse.energy == Catch::Approx(B * X).epsilon(1e-9));
}

// The 3D magnetostatic case: curl(nu curl A) = J with NO regularization -- singular, since
// every gradient is in the null space -- preconditioned by AMS, with a
// divergence-free source so the system is consistent. The quadratic
// manufactured A = (y^2, z^2, x^2) lies in the third-order Nedelec space, so B
// must be exact to solver precision (the regularized direct path is limited to
// ~1e-7 by its 1e-6 mass term).
TEST_CASE("HYPRE AMS solves the singular 3D magnetostatic problem", "[hypre][mpi][3d]") {
	const double nu = 1.0;
	mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(3, 3, 3, mfem::Element::TETRAHEDRON);
	auto mesh = parallel::MakeSingleRankParMesh(serial);
	mfem::ND_FECollection fec(3, 3);
	mfem::ParFiniteElementSpace fes(mesh.get(), &fec);

	mfem::VectorFunctionCoefficient A_exact(3, [](const mfem::Vector& p, mfem::Vector& A) {
		A.SetSize(3);
		A(0) = p(1) * p(1);
		A(1) = p(2) * p(2);
		A(2) = p(0) * p(0);
	});
	mfem::VectorFunctionCoefficient B_exact(3, [](const mfem::Vector& p, mfem::Vector& B) {
		B.SetSize(3);
		B(0) = -2.0 * p(2);
		B(1) = -2.0 * p(0);
		B(2) = -2.0 * p(1);
	});
	mfem::Vector j(3);
	j = -2.0 * nu;
	mfem::VectorConstantCoefficient J(j);

	mfem::Array<int> ess_bdr(mesh->bdr_attributes.Max());
	ess_bdr = 1;
	mfem::Array<int> ess_tdofs;
	fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);

	mfem::ParLinearForm b(&fes);
	b.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(J));
	b.Assemble();
	mfem::ConstantCoefficient nu_coeff(nu);
	mfem::ParBilinearForm a(&fes);
	a.AddDomainIntegrator(new mfem::CurlCurlIntegrator(nu_coeff));
	a.Assemble();

	mfem::ParGridFunction A(&fes);
	A = 0.0;
	A.ProjectBdrCoefficientTangent(A_exact, ess_bdr);
	mfem::HypreParMatrix K;
	mfem::Vector X, B;
	a.FormLinearSystem(ess_tdofs, A, b, K, X, B);

	mfem::HypreAMS ams(K, &fes);
	ams.SetSingularProblem();
	ams.SetPrintLevel(0);
	mfem::CGSolver cg(MPI_COMM_WORLD);
	cg.SetOperator(K);
	cg.SetPreconditioner(ams);
	cg.SetRelTol(1e-12);
	cg.SetMaxIter(500);
	cg.Mult(B, X);
	INFO("AMS-CG iterations: " << cg.GetNumIterations());
	REQUIRE(cg.GetConverged());
	REQUIRE(cg.GetNumIterations() < 100);
	a.RecoverFEMSolution(X, b, A);

	mfem::CurlGridFunctionCoefficient curl(&A);
	double err = 0.0;
	mfem::Vector b_h, b_ex;
	for (int e = 0; e < mesh->GetNE(); ++e) {
		mfem::ElementTransformation* T = mesh->GetElementTransformation(e);
		mfem::IntegrationPoint ip;
		ip.Set3(0.2, 0.3, 0.1);
		T->SetIntPoint(&ip);
		curl.Eval(b_h, *T, ip);
		B_exact.Eval(b_ex, *T, ip);
		b_h -= b_ex;
		err = std::max(err, b_h.Normlinf());
	}
	INFO("max |B - B_exact| = " << err);
	REQUIRE(err < 1e-8);
}

TEST_CASE("The MPI runtime reports a single rank", "[hypre][mpi]") {
	REQUIRE(parallel::Enabled());
	REQUIRE(parallel::WorldSize() == 1);
	REQUIRE(parallel::IsRoot());
}

#else

TEST_CASE("The serial runtime reports a single process", "[hypre][mpi]") {
	REQUIRE_FALSE(parallel::Enabled());
	REQUIRE(parallel::WorldSize() == 1);
	REQUIRE(parallel::IsRoot());
}

#endif
