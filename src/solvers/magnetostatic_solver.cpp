// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include "magnetostatic_solver.hpp"

void MagnetostaticSolver::Setup() 
{
	int order = config.Order;
	const int dim = mesh.Dimension();

	InitializeMagneticGeometry();
	for (const auto& [term_name, term] : config.Terminals) {
		MFEM_VERIFY(term.DriveQuantity == Quantity::Current,
			"Magnetostatic terminal '" + term_name +
			"' must use a current excitation.");
	}

	// Reject negative radii, record whether the domain reaches r = 0, and
	// report under-resolved near-axis curl-curl quadrature.
	ValidateMagneticAxisymmetricGeometry();

	// FE collection
	fec = std::make_unique<mfem::H1_FECollection>(order, dim);

	// Material Properties (Reluctivity nu = 1/mu), keyed by mesh DOMAIN attribute.
	BuildReluctivity();
	BuildConductivity();
	for (const auto& [term_name, term] : config.Terminals) {
		if (term.Conductor != ConductorType::Massive) continue;
		ValidateMassiveConductor(term_name, config.EntityGroups.at(term.EntityGroupName).AttributeIds);
	}

	boundary_conditions = BuildBoundaryConditions();
	BuildEssentialBoundaryMarker();
	RequireReferencePotential();

	// Build the FE space and everything bound to it for the starting mesh.
	BuildOperators();
	ValidateMagneticAxisBoundaryValues();

	BoundaryConditionValidator validator(mesh, *fespace);
	validator.ValidateBoundaryConditions(
		boundary_conditions.Entries(), /*terminals=*/{}, /*allow_overlap=*/false);

}

void MagnetostaticSolver::BuildOperators()  {
	fespace = std::make_unique<mfem::FiniteElementSpace>(&mesh, fec.get());

	massive_sources.clear();
	for (const auto& [term_name, term] : config.Terminals) {
		if (term.Conductor != ConductorType::Massive) continue;
		const auto& attributes = config.EntityGroups.at(term.EntityGroupName).AttributeIds;
		massive_sources[term_name] = { MassiveConductorLoad(term_name, attributes),
									   MassiveConductance(term_name, attributes) };
	}

	A = std::make_unique<mfem::GridFunction>(fespace.get());
	*A = 0.0;
	neumann_rhs = AssembleNaturalBoundaryLoad();

	a = std::make_unique<mfem::BilinearForm>(fespace.get());
	a->AddDomainIntegrator(MakeStiffnessIntegrator()); // a takes ownership
	a->Assemble();

	fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
	AddAxisTrueDofs(ess_tdof_list);

	// Form the constrained system operator. The eliminated-column part
	// (mat_e, used to build each scenario's RHS) is bound to A_op, which is
	// reused for every scenario's solve on this mesh.
	a->FormSystemMatrix(ess_tdof_list, A_op);

	// Factor here rather than in SolveSystem() so the (dominant) factorization
	// cost is paid once per mesh instead of once per scenario. AMR rebuilds it
	// implicitly by re-running BuildOperators() after each refinement.
	direct_solver.reset();
	amg.reset();
	if (config.LinearSolver == LinearSolverType::Direct) {
		WarnOnLargeDirectSolve(fespace->GetTrueVSize());
		auto operation = Reporter().Start("sparse direct factorization");
		direct_solver = std::make_unique<SparseDirectSolver>(SystemMatrix());
	}
	else {
		// Built once per mesh like the factorization, and reused for every
		// scenario's right-hand side.
		auto operation = Reporter().Start("algebraic multigrid setup");
		amg = std::make_unique<AmgPreconditioner>(SystemMatrix());
	}
}

mfem::SparseMatrix& MagnetostaticSolver::SystemMatrix() const {
	auto* sp = dynamic_cast<mfem::SparseMatrix*>(A_op.Ptr());
	MFEM_VERIFY(sp, "Expected a SparseMatrix operator from FormSystemMatrix.");
	return *sp;
}

double MagnetostaticSolver::ComputePeakFieldMagnitude() const  {
	if (!A) { return 0.0; }

	std::optional<MagneticFieldCoefficient> B_axi;
	if (axis_geometry) { B_axi.emplace(A.get(), axis_geometry->tolerance); }

	double peak = 0.0;
	mfem::Vector B;
	for (int e = 0; e < fespace->GetNE(); ++e) {
		const mfem::FiniteElement* fe = fespace->GetFE(e);
		mfem::ElementTransformation* T = fespace->GetElementTransformation(e);
		const mfem::IntegrationRule& nodes = fe->GetNodes();
		for (int i = 0; i < nodes.GetNPoints(); ++i) {
			const mfem::IntegrationPoint& ip = nodes.IntPoint(i);
			T->SetIntPoint(&ip);
			if (B_axi) { B_axi->Eval(B, *T, ip); }  // true |B| incl. A/r term
			else { A->GetGradient(*T, B); }   // |B| == |grad(A)| (planar)
			const double mag = B.Norml2();
			if (mag > peak) { peak = mag; }
		}
	}
	return peak;
}

void MagnetostaticSolver::ImprintScenario(const Scenario& sc, ImprintMode mode) {
	*A = 0.0; // Reset solution for new scenario

	// Re-apply non-zero essential BC values on this mesh's A.
	// ess_tdof values are lifted into the RHS by FormLinearSystem at solve time,
	// so they must be set AFTER the *A = 0.0 reset, every scenario.
	if (mode == ImprintMode::Field) {
		ForEachNonzeroDirichlet([&](mfem::Array<int>& marker, double value) {
			mfem::ConstantCoefficient c(value);
			A->ProjectBdrCoefficient(c, marker);
		});
	}
	// Axis stays at A=0 (already zero from the reset; no projection needed).

	auto j_src = BuildCurrentDensity(sc);
	j_coeff = std::make_unique<mfem::PWConstCoefficient>(j_src);
	// RHS
	b = std::make_unique<mfem::LinearForm>(fespace.get());

	// Integrates J * v under the geometry's measure (2*pi*r for axisymmetric).
	b->AddDomainIntegrator(Geometry().NewDomainLFIntegrator(*j_coeff));
	b->Assemble();
	for (const auto& [term_name, source] : massive_sources) {
		const double I = ExcitationFor(sc, term_name).real();
		if (I != 0.0) { b->Add(I / source.conductance, source.load); }
	}
	if (mode == ImprintMode::Field) {
		*b += neumann_rhs;
	}
}

void MagnetostaticSolver::RunOnCurrentMesh() 
{
	if (config.AnalysisType == AnalysisType::CouplingMatrix) {
		L = std::make_unique<mfem::DenseMatrix>(config.Terminals.size());
		// CouplingMatrix synthesizes a unit-current scenario per terminal, so
		// every terminal must be current-driven for the drive to be meaningful.
		for (const auto& [term_name, term] : config.Terminals) {
			MFEM_VERIFY(term.DriveQuantity == Quantity::Current,
				"CouplingMatrix terminal '" + term_name +
				"' must be a Current terminal for the magnetostatic solver.");

		}
	}

	for (const auto& [sc_name, sc] : BuildSolveScenarios()) {
		auto operation = Reporter().Start("scenario '" + sc_name + "'");
		ImprintScenario(sc,
			config.AnalysisType == AnalysisType::CouplingMatrix
				? ImprintMode::CouplingPerturbation
				: ImprintMode::Field);
		SolveSystem();
		AccumulateScenarioError();
		SaveScenario(sc_name, sc, config.AnalysisType == AnalysisType::CouplingMatrix
			? sc.Excitations.front().TerminalName : "");
		if (config.AnalysisType == AnalysisType::CouplingMatrix) {
			// Each scenario is a unit-current drive on one terminal, so the
			// solution's flux linkage / inductance is the corresponding column
			// of the coupling matrix.
			const int col = std::distance(config.Terminals.begin(),
				config.Terminals.find(sc.Excitations[0].TerminalName));
			for (int row = 0; row < L->Height(); ++row) {
				const std::string& row_term = std::next(config.Terminals.begin(), row)->first;
				(*L)(row, col) = ComputeFluxLinkage(row_term);
			}
		}
	}
}

void MagnetostaticSolver::SolveSystem() {
	auto operation = Reporter().Start("linear system solve");
	// Form and solve
	mfem::Vector X, B;

	// Reuses the constrained operator cached in BuildOperators(); only this
	// scenario's eliminated RHS is re-derived here.
	a->FormLinearSystem(ess_tdof_list, *A, *b, A_op, X, B);

	if (B.Norml2() < 1e-12 && X.Norml2() < 1e-12)
	{
		mfem::out << "WARNING: Linear system RHS is ~zero. "
			<< "Check that 'sources' in JSON match mesh attributes.\n";
	}

	if (direct_solver) {
		// Back-substitution only: the factorization was done in BuildOperators().
		direct_solver->Mult(B, X);
	}
	else {
		SolveSpdIteratively(*A_op, *amg, B, X);
	}

	a->RecoverFEMSolution(X, *b, *A);

	std::ostringstream statistics;
	statistics << "=== A Statistics ===\n"
		<< "  A min:     " << A->Min() << "\n"
		<< "  A max:     " << A->Max() << "\n"
		<< "  A L2 norm: " << A->Norml2();
	Reporter().Diagnostic(statistics.str());
}

FieldExportSet MagnetostaticSolver::CollectExportFields() const 
{
	FieldExportSet fields;
	fields.AddPrimary("A", *A);

	if (geometry == GeometryType::Axisymmetric) {
		fields.AddVector("B", std::make_unique<MagneticFieldCoefficient>(
			A.get(), axis_geometry->tolerance));
	}
	else {
		fields.AddVector("B", std::make_unique<PlanarMagneticFieldCoefficient>(A.get()));
	}
	return fields;
}

void MagnetostaticSolver::SaveAnalysisResults() 
{
	if (config.AnalysisType == AnalysisType::CouplingMatrix) {
		WriteCouplingMatrix();
	}
}

void MagnetostaticSolver::WriteCouplingMatrix() {
	if (!L) {
		Reporter().Warning("WriteCouplingMatrix: coupling matrix not computed.");
		return;
	}

	SaveCouplingMatrix(*L, "Inductance Matrix " + CouplingUnitLabel("H"),
		"Inductance", "H");
}

double MagnetostaticSolver::ComputeFluxLinkage(const std::string& terminal_name) const
{
	const auto massive = massive_sources.find(terminal_name);
	if (massive != massive_sources.end()) {
		return (massive->second.load * *A) / massive->second.conductance;
	}
	// The integrator carries the full geometric measure, so this is webers.
	return WindingFunctional(terminal_name) * *A;
}

mfem::Vector MagnetostaticSolver::BuildCurrentDensity(const Scenario& sc) const {
	mfem::Vector j_re, j_im;
	MagneticSolver::BuildCurrentDensity(sc, [](const Terminal& term) {
		return term.Conductor == ConductorType::Stranded;
	}, j_re, j_im);
	return j_re;
}
