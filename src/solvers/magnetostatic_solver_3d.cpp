// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include "magnetostatic_solver_3d.hpp"

double MagnetostaticSolver3D::MagneticEnergy() const {
	mfem::BilinearForm k(fespace.get());
	k.AddDomainIntegrator(new mfem::CurlCurlIntegrator(*nu_coeff));
	k.Assemble();
	k.Finalize();
	return 0.5 * k.InnerProduct(*A, *A);
}

std::vector<double> MagnetostaticSolver3D::FluxLinkages() const {
	std::vector<double> lambda;
	for (const auto& load : terminal_loads) { lambda.push_back(load * *A); }
	return lambda;
}

void MagnetostaticSolver3D::Setup()  {
	InitializeVectorPotential();
	BuildOperators();
	ValidateVectorPotentialBoundaries();
}

void MagnetostaticSolver3D::BuildOperators()  {
	BuildSpaceAndConductors();
	A = std::make_unique<mfem::GridFunction>(fespace.get());
	*A = 0.0;

	const bool direct = config.LinearSolver == LinearSolverType::Direct;
	a = std::make_unique<mfem::BilinearForm>(fespace.get());
	a->AddDomainIntegrator(new mfem::CurlCurlIntegrator(*nu_coeff));
	a->Assemble();
	a->FormSystemMatrix(ess_tdof_list, A_op);

	auto* matrix = dynamic_cast<mfem::SparseMatrix*>(A_op.Ptr());
	MFEM_VERIFY(matrix, "Expected a SparseMatrix operator from FormSystemMatrix.");
	direct_solver.reset();
#ifdef MFEM_USE_MPI
	ams.reset();
#endif
	if (direct) {
		WarnOnLargeDirectSolve(fespace->GetTrueVSize());
		auto operation = Reporter().Start("sparse direct factorization");
		const auto gauge = projector->GaugeConstraint(mfem::Array<int>());
		direct_solver = std::make_unique<GaugedDirectSolver>(*matrix, *gauge, ess_tdof_list);
	}
	else {
#ifdef MFEM_USE_MPI
		auto operation = Reporter().Start("AMS preconditioner setup");
		ams = std::make_unique<SerialAmsPreconditioner>(*matrix, *fespace, /*singular=*/true);
#endif
	}

	terminal_loads.clear();
	for (const TerminalConductor& c : conductors) {
		terminal_loads.push_back(ProjectedUnitCurrentLoad(c));
	}
}

void MagnetostaticSolver3D::RunOnCurrentMesh()  {
	const bool coupling = config.AnalysisType == AnalysisType::CouplingMatrix;
	const int n = static_cast<int>(config.Terminals.size());
	if (coupling) {
		L = std::make_unique<mfem::DenseMatrix>(n, n);
		*L = 0.0;
	}
	int column = 0;
	for (const auto& [name, scenario] : BuildSolveScenarios()) {
		auto operation = Reporter().Start("scenario '" + name + "'");
		ImprintScenario(scenario);
		SolveSystem();
		if (coupling) {
			const std::vector<double> lambda = FluxLinkages();
			for (int row = 0; row < n; ++row) { (*L)(row, column) = lambda[row]; }
			++column;
		}
		SaveScenario(name, scenario,
			coupling ? scenario.Excitations.front().TerminalName : "");
	}
}

FieldExportSet MagnetostaticSolver3D::CollectExportFields() const  {
	FieldExportSet fields;
	fields.AddPrimary("A", *A);
	fields.AddVector("B", std::make_unique<mfem::CurlGridFunctionCoefficient>(A.get()));
	return fields;
}

void MagnetostaticSolver3D::SaveAnalysisResults()  {
	if (config.AnalysisType != AnalysisType::CouplingMatrix) return;
	if (!L) {
		Reporter().Warning("WriteCouplingMatrix: coupling matrix not computed.");
		return;
	}
	SaveCouplingMatrix(*L, "Inductance Matrix " + CouplingUnitLabel("H"),
		"Inductance", "H");
}

void MagnetostaticSolver3D::ImprintScenario(const Scenario& scenario) {
	const bool background = config.AnalysisType != AnalysisType::CouplingMatrix;
	*A = 0.0;
	if (background && boundary_value) {
		A->ProjectBdrCoefficientTangent(*boundary_value, ess_bdr);
	}
	b = std::make_unique<mfem::LinearForm>(fespace.get());
	if (background && source) {
		b->AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(*source));
	}
	b->Assemble();
	if (background && source) { projector->Project(*b); }

	for (size_t k = 0; k < conductors.size(); ++k) {
		const double current = ExcitationFor(scenario, conductors[k].Name).real();
		if (current != 0.0) { b->Add(current, terminal_loads[k]); }
	}
}

void MagnetostaticSolver3D::SolveSystem() {
	auto operation = Reporter().Start("linear system solve");
	mfem::Vector X, B;
	a->FormLinearSystem(ess_tdof_list, *A, *b, A_op, X, B);
	if (direct_solver) {
		direct_solver->Mult(B, X);
		a->RecoverFEMSolution(X, *b, *A);
		return;
	}
#ifdef MFEM_USE_MPI
	SolveSpdIteratively(*A_op, *ams, B, X);
	a->RecoverFEMSolution(X, *b, *A);
	projector->RemoveGradient(*A);
#endif
}
