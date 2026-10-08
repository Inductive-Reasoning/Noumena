// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include "electrostatic_solver.hpp"

void ElectrostaticSolver::Setup()  {
	int order = config.Order;
	const int dim = mesh.Dimension();

	InitializeGeometry();
	for (const auto& [term_name, term] : config.Terminals) {
		MFEM_VERIFY(term.DriveQuantity == Quantity::Voltage,
			"Electrostatic terminal '" + term_name +
			"' must use a voltage excitation.");
	}

	// Reject negative radii. Electrostatics needs no axis condition: the
	// natural condition dV/dr = 0 on r = 0 is already the correct one.
	ValidateAxisymmetricGeometry();

	// FE collection
	fec = std::make_unique<mfem::H1_FECollection>(order, dim);

	// Material Properties (Permittivity). epsilon_coeff is a PWConstCoefficient
	// keyed by mesh DOMAIN attribute. AMR refinement subdivides elements but
	// preserves their attributes, so this mapping is refinement-invariant.

	epsilon_coeff = MaterialCoefficient(0.0, [](const Material& m) {
		return m.RelPermittivity * Constants::EPSILON_0; });

	boundary_conditions = BuildBoundaryConditions();
	for (const auto& bc : boundary_conditions) {
		MFEM_VERIFY(!bc.IsRobin() || bc.Condition.RobinCoeff >= 0.0,
			"Robin boundary group '" + bc.Condition.EntityGroupName +
			"' has a negative robin_coefficient; electrostatics requires "
			"a non-negative one (eps/R for a spherical far-field boundary "
			"of radius R).");
	}

	// Voltage terminals are realized as essential constraints in this
	// formulation. They are not boundary conditions: the value is
	// scenario-dependent and is projected in ImprintScenario(). See
	// docs/boundary_and_terminal_model.md.
	terminal_markers.clear();
	for (const auto& [term_name, term] : config.Terminals) {
		terminal_markers[term_name] = MarkerFromGroup(term.EntityGroupName);
	}

	BuildEssentialBoundaryMarker();
	RequireReferencePotential();

	// Build the FE space and everything bound to it for the starting mesh.
	BuildOperators();

	// Validate boundary conditions once. The check is over the (refinement-
	// invariant) mesh topology / attributes; it merely needs an FE space for
	// DOF queries, so running it after the first BuildOperators() is correct.
	BoundaryConditionValidator validator(mesh, *fespace);
	validator.ValidateBoundaryConditions(boundary_conditions.Entries(),
		terminal_markers, /*allow_overlap=*/false);
}

void ElectrostaticSolver::BuildEssentialBoundaryMarker()  {
	PhysicsSolver::BuildEssentialBoundaryMarker();

	for (const auto& [term_name, marker] : terminal_markers) {
		MergeMarker(ess_bdr, marker);
	}
}

void ElectrostaticSolver::BuildOperators()  {
	fespace = std::make_unique<mfem::FiniteElementSpace>(&mesh, fec.get());
	
	x = std::make_unique<mfem::GridFunction>(fespace.get());
	*x = 0.0;

	robin_coeffs.clear();
	robin_markers.clear();
	for (const auto& bc : boundary_conditions) {
		if (!bc.IsRobin() || bc.Condition.RobinCoeff == 0.0) continue;
		robin_coeffs.push_back(
			std::make_unique<mfem::ConstantCoefficient>(bc.Condition.RobinCoeff));
		robin_markers.push_back(std::make_unique<mfem::Array<int>>(bc.Marker));
	}
	a = std::make_unique<mfem::BilinearForm>(fespace.get());
	AddOperatorIntegrators(*a);
	a->Assemble();

	// The same operator, unconstrained, for the charge extraction Q = K*x.
	// Essential elimination rewrites `a`'s matrix, so K is assembled by a
	// form of its own from the same integrators. It must be the whole
	// operator, Robin terms included: a terminal DOF's basis function also
	// reaches onto any Robin boundary next to the terminal, and the charge
	// on the terminal is that DOF's full residual. With the domain
	// stiffness alone, the flux through the adjoining Robin faces would be
	// counted as terminal charge. Only coupling runs extract charge, so
	// only they pay for it.
	K.reset();
	if (config.AnalysisType == AnalysisType::CouplingMatrix) {
		mfem::BilinearForm unconstrained(fespace.get());
		AddOperatorIntegrators(unconstrained);
		unconstrained.Assemble();
		unconstrained.Finalize();
		K.reset(unconstrained.LoseMat());
	}

	// Linear Form (RHS)
	b = std::make_unique<mfem::LinearForm>(fespace.get());
	natural_rhs = AssembleNaturalBoundaryLoad();

	fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

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

mfem::SparseMatrix& ElectrostaticSolver::SystemMatrix() const {
	auto* sp = dynamic_cast<mfem::SparseMatrix*>(A_op.Ptr());
	MFEM_VERIFY(sp, "Expected a SparseMatrix operator from FormSystemMatrix.");
	return *sp;
}

void ElectrostaticSolver::AddOperatorIntegrators(mfem::BilinearForm& form) const {
	form.AddDomainIntegrator(MakeStiffnessIntegrator());
	for (size_t i = 0; i < robin_coeffs.size(); ++i) {
		form.AddBoundaryIntegrator(
			Geometry().NewBoundaryMassIntegrator(*robin_coeffs[i]), *robin_markers[i]);
	}
}

double ElectrostaticSolver::ComputePeakFieldMagnitude() const  {
	if (!x) { return 0.0; }
	double peak = 0.0;
	mfem::Vector grad;
	for (int e = 0; e < fespace->GetNE(); ++e) {
		const mfem::FiniteElement* fe = fespace->GetFE(e);
		mfem::ElementTransformation* T = fespace->GetElementTransformation(e);
		const mfem::IntegrationRule& nodes = fe->GetNodes();
		for (int i = 0; i < nodes.GetNPoints(); ++i) {
			const mfem::IntegrationPoint& ip = nodes.IntPoint(i);
			T->SetIntPoint(&ip);
			x->GetGradient(*T, grad); // grad(V); |E| = |grad(V)|
			const double mag = grad.Norml2();
			if (mag > peak) { peak = mag; }
		}
	}
	return peak;
}

void ElectrostaticSolver::ImprintScenario(const Scenario& sc, ImprintMode mode) {
	*x = 0.0; // Reset solution for new scenario
	*b = 0.0;

	if (mode == ImprintMode::Field) {
		*b = natural_rhs;
		ForEachNonzeroDirichlet([&](mfem::Array<int>& marker, double value) {
			mfem::ConstantCoefficient c(value);
			x->ProjectBdrCoefficient(c, marker);
		});
	}
	for (const auto& [term_name, term] : config.Terminals) {
		if (term.DriveQuantity == Quantity::Voltage) {
			mfem::Array<int> marker(terminal_markers.at(term_name));
			mfem::ConstantCoefficient c(ExcitationFor(sc, term_name).real());
			x->ProjectBdrCoefficient(c, marker);
		}
		else
		{
			Reporter().Warning("Excitation type not supported in ElectrostaticSolver. "
				"Skipping terminal " + term_name + ".");
			continue;
		}
	}
}

void ElectrostaticSolver::RunOnCurrentMesh()  {
	const bool coupling = (config.AnalysisType == AnalysisType::CouplingMatrix);
	if (coupling) {
		const int n = static_cast<int>(config.Terminals.size());
		C = std::make_unique<mfem::DenseMatrix>(n, n);
		*C = 0.0;
	}

	int col = 0;
	for (const auto& [name, sc] : BuildSolveScenarios()) {
		auto operation = Reporter().Start("scenario '" + name + "'");
		ImprintScenario(sc, coupling ? ImprintMode::CouplingPerturbation
										 : ImprintMode::Field);
		SolveSystem();
		AccumulateScenarioError();
		if (coupling) { GatherChargeColumn(col++); }
		SaveScenario(name, sc, coupling ? sc.Excitations.front().TerminalName : "");
	}
}

void ElectrostaticSolver::SolveSystem() {
	auto operation = Reporter().Start("linear system solve");
	// The system matrix and its essential-DOF elimination (mat_e) were built for
	// the current mesh in BuildOperators(). FormLinearSystem here re-derives ONLY
	// this scenario's eliminated RHS from the freshly imprinted x/b (b was zeroed
	// in ImprintScenario, so no previous scenario's load survives in it); it
	// reuses the same already-eliminated operator. Each scenario is therefore
	// solved independently, and AMR re-runs BuildOperators() after refinement to
	// rebuild on the new mesh.
	mfem::Vector B, X;
	a->FormLinearSystem(ess_tdof_list, *x, *b, A_op, X, B);
	if (direct_solver) {
		// Back-substitution only: the factorization was done in BuildOperators().
		direct_solver->Mult(B, X);
	}
	else {
		SolveSpdIteratively(*A_op, *amg, B, X);
	}
	a->RecoverFEMSolution(X, *b, *x);
}

FieldExportSet ElectrostaticSolver::CollectExportFields() const  {
	FieldExportSet fields;
	fields.AddPrimary("V", *x);

	auto grad = std::make_unique<mfem::GradientGridFunctionCoefficient>(x.get());
	auto& grad_ref = fields.Own(std::move(grad));
	fields.AddVector("E",
		std::make_unique<mfem::ScalarVectorProductCoefficient>(-1.0, grad_ref));

	fields.AddScalar("Permittivity", *epsilon_coeff);
	return fields;
}

void ElectrostaticSolver::SaveAnalysisResults()  {
	// Writes the computed Maxwell (short-circuit) capacitance matrix.
	// C(k,i) is the charge induced on conductor k when conductor i is held at
	// 1 V and all other conductors at 0 V. The matrix is symmetric; diagonals
	// are positive and off-diagonals negative. Each row sums to that
	// conductor's capacitance to the grounded boundary.
	if (config.AnalysisType == AnalysisType::CouplingMatrix) {
		if (!C) {
			Reporter().Warning("WriteCouplingMatrix: coupling matrix not computed.");
			return;
		}

		SaveCouplingMatrix(*C, "Capacitance Matrix " + CouplingUnitLabel("F"),
			"Capacitance", "F");
	}
}

void ElectrostaticSolver::GatherChargeColumn(int col) {
	std::vector<const std::pair<const std::string, Terminal>*> terms;
	terms.reserve(config.Terminals.size());
	for (const auto& kv : config.Terminals) terms.push_back(&kv);

	mfem::Vector Q(fespace->GetVSize());
	K->Mult(*x, Q);

	for (int k = 0; k < static_cast<int>(terms.size()); ++k) {
		mfem::Array<int> vdofs_k;
		fespace->GetEssentialVDofs(terminal_markers.at(terms[k]->first), vdofs_k);
		double Qk = 0.0;
		for (int n = 0; n < vdofs_k.Size(); ++n) {
			if (vdofs_k[n]) Qk += Q(n);
		}
		(*C)(k, col) = Qk;
	}
}
