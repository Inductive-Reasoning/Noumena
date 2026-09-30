// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once
#include <memory>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>
#include "mfem.hpp"
#include "physics_solver.hpp"
#include "../config/boundary_validation.hpp"
#include "../io/gmsh_results_writer.hpp"
#include "amr_support.hpp"
#include "../linalg/amg_preconditioner.hpp"
#include "../linalg/sparse_direct_solver.hpp"

class ElectrostaticSolver : public PhysicsSolver {
	enum class ImprintMode { Field, CouplingPerturbation };

	// Primary Spaces
	std::unique_ptr<mfem::GridFunction> x; // Electric Potential (V)

	// Physics
	std::unique_ptr<mfem::PWConstCoefficient> epsilon_coeff;

	std::unique_ptr<mfem::LinearForm> b;
	mfem::Vector natural_rhs; // Neumann flux + Robin data load (boundary data)
	std::unique_ptr<mfem::BilinearForm> a;

	// Robin coefficients and markers bound into `a`. MFEM keeps pointers to
	// both, so they live as long as the form does.
	std::vector<std::unique_ptr<mfem::ConstantCoefficient>> robin_coeffs;
	std::vector<std::unique_ptr<mfem::Array<int>>> robin_markers;

	// Domain stiffness alone, for the coupling charge extraction Q = K0*x.
	// Assembled by its own form rather than snapshotted from `a`, because `a`
	// also carries Robin boundary terms. See BuildOperators().
	std::unique_ptr<mfem::SparseMatrix> K0;
	std::unique_ptr<mfem::DenseMatrix> C; // Coupling Matrix for terminals

	// Terminal name -> boundary marker, resolved once at setup. This solver
	// realizes every voltage terminal as an essential constraint, so these feed
	// ess_bdr; they are also how per-terminal charge is gathered.
	std::unordered_map<std::string, mfem::Array<int>> terminal_markers;

	// Cached constrained system for the CURRENT mesh. The matrix is identical
	// for every solve on a given mesh (same bilinear form and essential DOFs),
	// so it is assembled once per mesh in BuildOperators() and reused for all
	// scenarios / coupling columns. AMR refinement rebuilds it via BuildOperators().
	mfem::OperatorPtr A_op;

	// Factorization of A_op, built once per mesh when the direct solver is
	// selected and reused for every scenario's RHS. Null when solving iteratively.
	std::unique_ptr<SparseDirectSolver> direct_solver;

	// Multigrid preconditioner for the iterative path, built once per mesh like
	// direct_solver and reused for every scenario. Null when solving directly.
	std::unique_ptr<AmgPreconditioner> amg;

public:
	ElectrostaticSolver(mfem::Mesh& m, const ProblemConfig& c) : PhysicsSolver(m, c) {}

	void Setup() override {
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

		// Build the FE space and everything bound to it for the starting mesh.
		BuildOperators();

		// Validate boundary conditions once. The check is over the (refinement-
		// invariant) mesh topology / attributes; it merely needs an FE space for
		// DOF queries, so running it after the first BuildOperators() is correct.
		BoundaryConditionValidator validator(mesh, *fespace);
		validator.ValidateBoundaryConditions(boundary_conditions.Entries(),
			terminal_markers, /*allow_overlap=*/false);
	}

	// Robin conditions ((eps dV/dn) + RobinCoeff * V = Value, outward normal)
	// are assembled into the operator; see BuildOperators(). The coefficient
	// must be non-negative: a negative one makes the operator indefinite, which
	// both the Cholesky factorization and PCG require it not to be.
	bool SupportsRobin() const override { return true; }

	// Driven electrodes pin their DOFs exactly as a Dirichlet condition does, so
	// every terminal marker joins the prescribed Dirichlet ones. Only the value
	// differs in origin, and that is supplied per scenario in ImprintScenario().
	void BuildEssentialBoundaryMarker() override {
		PhysicsSolver::BuildEssentialBoundaryMarker();

		for (const auto& [term_name, marker] : terminal_markers) {
			MergeMarker(ess_bdr, marker);
		}
	}

	// (Re)build the FE space and every object bound to it for the CURRENT mesh.
	// Called once from Setup() and again after each AMR refinement. The
	// refinement-invariant data (fec, epsilon_coeff, ess_bdr, boundary_conditions,
	// terminal_markers) persists across calls and is reused.
	//
	// The constrained system matrix is assembled here; within a single mesh that
	// matrix is reused for every scenario / coupling column (the bilinear form
	// and essential-DOF set do not change between solves). AMR refinement
	// invalidates the mesh, so this is re-run to rebuild on the new mesh.
	void BuildOperators() override {
		fespace = std::make_unique<mfem::FiniteElementSpace>(&mesh, fec.get());
		
		x = std::make_unique<mfem::GridFunction>(fespace.get());
		*x = 0.0;

		// System operator: domain stiffness plus the Robin boundary terms
		// (RobinCoeff * V, v) over each Robin boundary, all under the
		// geometry's measure.
		a = std::make_unique<mfem::BilinearForm>(fespace.get());
		a->AddDomainIntegrator(MakeStiffnessIntegrator()); // a takes ownership
		robin_coeffs.clear();
		robin_markers.clear();
		for (const auto& bc : boundary_conditions) {
			if (!bc.IsRobin() || bc.Condition.RobinCoeff == 0.0) continue;
			robin_coeffs.push_back(
				std::make_unique<mfem::ConstantCoefficient>(bc.Condition.RobinCoeff));
			robin_markers.push_back(std::make_unique<mfem::Array<int>>(bc.Marker));
			a->AddBoundaryIntegrator(
				Geometry().NewBoundaryMassIntegrator(*robin_coeffs.back()),
				*robin_markers.back());
		}
		a->Assemble();

		// UNCONSTRAINED domain stiffness for the charge extraction Q = K0*x.
		//
		// INVARIANT: K0 must contain the DOMAIN stiffness and nothing else. The
		// charge extraction is Gauss's law over the volume; the Robin terms in
		// `a` are part of the operator but not part of that relation, so
		// including them would silently shift every extracted charge and
		// capacitance with no error and no failing assertion. K0 is therefore
		// assembled by its own domain-only form, which makes the invariant
		// structural instead of an assembly-ordering rule. Only coupling runs
		// extract charge, so only they pay for it.
		K0.reset();
		if (config.AnalysisType == AnalysisType::CouplingMatrix) {
			mfem::BilinearForm domain(fespace.get());
			domain.AddDomainIntegrator(MakeStiffnessIntegrator());
			domain.Assemble();
			domain.Finalize();
			K0.reset(domain.LoseMat());
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

	// The constrained system matrix behind A_op. FormSystemMatrix always yields a
	// SparseMatrix for this serial build.
	mfem::SparseMatrix& SystemMatrix() const {
		auto* sp = dynamic_cast<mfem::SparseMatrix*>(A_op.Ptr());
		MFEM_VERIFY(sp, "Expected a SparseMatrix operator from FormSystemMatrix.");
		return *sp;
	}

	// Create the domain diffusion integrator matching the active geometry. Used
	// both by the solve (owned by 'a') and the AMR error estimator (a separate,
	// independently-owned instance), so the estimated error is consistent with
	// the assembled operator: Div(eps Grad V) = 0 under the geometry's measure
	// (2*pi*r for axisymmetric, the plain Cartesian Laplacian in 2D and 3D).
	mfem::BilinearFormIntegrator* MakeStiffnessIntegrator() const {
		return Geometry().NewDiffusionIntegrator(*epsilon_coeff);
	}

	// Estimate per-element error on the CURRENT mesh. The scenario-wide fold (a
	// running RMS over every scenario / coupling column) lives in the base class
	// AccumulateScenarioError(), so one shared mesh is refined for all scenarios
	// (spec: identical $Nodes/$Elements across every <scenario>.results.msh).
	//
	// Uses the serial recovery-based ZienkiewiczZhuEstimator (the L2 variant is
	// MPI-only). A dedicated integrator instance (separate from a's) and an H1
	// vector flux space are constructed here per call; SetFluxAveraging(1) keeps
	// the recovered flux from smoothing across material-attribute interfaces so
	// per-region permittivity discontinuities are respected. The recovered field
	// is grad(V); ComputeFluxEnergy applies eps once to form the physical norm.
	//
	// The raw indicator is then normalized by the total electrostatic field
	// energy Et = 0.5 * x^T K0 x, making it a dimensionless RELATIVE error. The
	// ZZ indicator has units of sqrt(energy) and so scales with the applied
	// voltage; without this the base-class fold would be dominated by whichever
	// scenario is driven hardest rather than by mesh quality. That is the normal
	// case for a capacitance extraction, where each terminal contributes its own
	// unit-excitation scenario.
	//
	// @param errors  Output: per-element error indicator (sized to NE).
	void EstimateCurrentSolutionError(mfem::Vector& errors) override {
		const int sdim = mesh.SpaceDimension();
		std::unique_ptr<mfem::BilinearFormIntegrator> flux_integ(MakeStiffnessIntegrator());
		mfem::FiniteElementSpace flux_fes(&mesh, fec.get(), sdim);
		mfem::ZienkiewiczZhuEstimator estimator(*flux_integ, *x, flux_fes);
		estimator.SetWithCoeff(false);    // field = grad(V); energy applies eps
		estimator.SetFluxAveraging(1);    // do not average across attribute interfaces
		errors = estimator.GetLocalErrors();

		// A zero/near-zero solution carries no energy and no meaningful relative
		// error; leave the indicator unscaled rather than dividing by ~0.
		const double energy = amr::FieldEnergy(*fespace, MakeStiffnessIntegrator(), *x);
		if (energy > 0.0) { errors /= std::sqrt(energy); }
	}

	// Peak field magnitude |E| = |grad(V)| over the current solution *x, sampled
	// at element nodes. Used as an AMR convergence diagnostic (peak |E| at a
	// conductor corner should settle as refinement resolves the singularity).
	// Reflects whichever solution currently lives in *x (the last one solved).
	double ComputePeakFieldMagnitude() const override {
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

	// Field analysis imprints the configured boundary data (nonzero Dirichlet
	// values and the Neumann load). CouplingPerturbation analysis omits it:
	// a coupling coefficient is a derivative with respect to terminal drive,
	// so each column must be the homogeneous response to a unit terminal
	// excitation. Because the problem is linear, homogenizing here is exactly
	// equivalent to subtracting a background baseline solve, at one fewer solve.
	void ImprintScenario(const Scenario& sc, ImprintMode mode) {
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
				mfem::ConstantCoefficient c(ExcitationFor(sc, term_name));
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

	// Solve + save on the CURRENT mesh/operators. Both analysis types flow through
	// ONE imprint -> solve loop over BuildSolveScenarios(); only the intrinsic
	// post-solve action differs: CouplingMatrix gathers an induced-charge column
	// into C. Shared output policy controls field serialization for either analysis.
	// Each AMR pass replaces the previous mesh's results.
	void RunOnCurrentMesh() override {
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

	void SolveSystem() {
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

	// Post-solve field recovery: the potential V, the field E = -grad(V), and the
	// per-region permittivity. Serialization is handled by the base class.
	FieldExportSet CollectExportFields() const override {
		FieldExportSet fields;
		fields.AddPrimary("V", *x);

		auto grad = std::make_unique<mfem::GradientGridFunctionCoefficient>(x.get());
		auto& grad_ref = fields.Own(std::move(grad));
		fields.AddVector("E",
			std::make_unique<mfem::ScalarVectorProductCoefficient>(-1.0, grad_ref));

		fields.AddScalar("Permittivity", *epsilon_coeff);
		return fields;
	}

	void SaveAnalysisResults() override {
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

private:
	// CouplingMatrix post-solve action for one column: with the just-solved
	// potential in *x (terminal `col` driven at 1 V, the rest grounded by the
	// synthesized scenario), gather the reaction charge Q = K0*x onto every
	// conductor's boundary DOFs and write column `col` of C. Off-diagonals are
	// negative, diagonals positive. K0 carries the full geometric measure in
	// both planar and axisymmetric mode, so Q is already in coulombs and needs
	// no geometry-dependent scaling here.
	// The coupling solve is imprinted in CouplingPerturbation mode, so the load
	// vector is identically zero and K0*x is the full reaction, with no RHS
	// contribution left to subtract.
	// Terminal order matches BuildSolveScenarios() / WriteCouplingMatrix()
	// (config.Terminals order).
	void GatherChargeColumn(int col) {
		std::vector<const std::pair<const std::string, Terminal>*> terms;
		terms.reserve(config.Terminals.size());
		for (const auto& kv : config.Terminals) terms.push_back(&kv);

		mfem::Vector Q(fespace->GetVSize());
		K0->Mult(*x, Q);

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
};