// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <cmath>
#include <algorithm>
#include <sstream>

#include "mfem.hpp"
#include "magnetic_solver.hpp"
#include "../axisym/axisymmetric_curl_curl_integrator.hpp"
#include "../axisym/axisymmetric_lf_integrator.hpp"
#include "../coefficients/magnetic_field_coefficient.hpp"
#include "../config/boundary_validation.hpp"
#include "../core/constants.hpp"
#include "../io/gmsh_results_writer.hpp"
#include "../linalg/amg_preconditioner.hpp"
#include "../linalg/sparse_direct_solver.hpp"

class MagnetostaticSolver : public MagneticSolver
{
private:
	enum class ImprintMode { Field, CouplingPerturbation };

	// Resources (order of declaration = order of destruction)
	std::unique_ptr<mfem::GridFunction> A; // A_phi (axisym) or A_z (planar scalar)

	std::unique_ptr<mfem::PWConstCoefficient> j_coeff; // J_phi (axisym) or J (planar scalar src)

	std::unique_ptr<mfem::LinearForm> b;
	std::unique_ptr<mfem::BilinearForm> a;
	mfem::Vector neumann_rhs;

	// Cached constrained system for the CURRENT mesh. The matrix is identical
	// for every solve on a given mesh (same bilinear form and essential DOFs),
	// so it is assembled once per mesh in BuildOperators() and reused for all
	// scenarios / coupling columns. AMR refinement rebuilds it via BuildOperators().
	mfem::OperatorHandle A_op;

	// Factorization of the cached constrained matrix, valid for the same lifetime
	// as A_op. Null when the iterative solver is configured.
	std::unique_ptr<SparseDirectSolver> direct_solver;

	// Multigrid preconditioner for the iterative path; see ElectrostaticSolver.
	std::unique_ptr<AmgPreconditioner> amg;

	std::unique_ptr<mfem::DenseMatrix> L; // Inductance matrix (coupling matrix) for the current mesh

public:
	MagnetostaticSolver(mfem::Mesh& m, const ProblemConfig& c) : MagneticSolver(m, c) {}

	void Setup() override
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

		boundary_conditions = BuildBoundaryConditions();
		BuildEssentialBoundaryMarker();

		// Build the FE space and everything bound to it for the starting mesh.
		BuildOperators();
		ValidateMagneticAxisBoundaryValues();

		BoundaryConditionValidator validator(mesh, *fespace);
		validator.ValidateBoundaryConditions(
			boundary_conditions.Entries(), /*terminals=*/{}, /*allow_overlap=*/false);

	}

	void BuildOperators() override {
		fespace = std::make_unique<mfem::FiniteElementSpace>(&mesh, fec.get());

		A = std::make_unique<mfem::GridFunction>(fespace.get());
		*A = 0.0;
		neumann_rhs = AssembleNaturalBoundaryLoad();

		a = std::make_unique<mfem::BilinearForm>(fespace.get());
		a->AddDomainIntegrator(MakeStiffnessIntegrator()); // a takes ownership
		a->Assemble();

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

	// Estimate per-element error on the CURRENT mesh. The scenario-wide fold (a
	// running RMS over every scenario / coupling column) lives in the base class
	// AccumulateScenarioError(), so one shared mesh is refined for all scenarios
	// (spec: identical $Nodes/$Elements across every <scenario>.results.msh).
	//
	// Uses the serial recovery-based ZienkiewiczZhuEstimator (the L2 variant is
	// MPI-only). A dedicated integrator instance (separate from a's) and an H1
	// vector flux space are constructed here per call; SetFluxAveraging(1) keeps
	// the recovered flux from smoothing across material-attribute interfaces so
	// per-region reluctivity discontinuities are respected. The recovered field
	// is grad(A) (planar) or B (axisymmetric); ComputeFluxEnergy applies
	// reluctivity once to form the physical norm.
	//
	// The raw indicator is then normalized by the total magnetic field energy
	// Et = 0.5 * A^T K A, making it a dimensionless RELATIVE error. The ZZ
	// indicator has units of sqrt(energy) and so scales with the driving current;
	// without this the base-class fold would be dominated by whichever scenario
	// is driven hardest rather than by mesh quality.
	//
	// @param errors  Output: per-element error indicator (sized to NE).
	void EstimateCurrentSolutionError(mfem::Vector& errors) override {
		const int sdim = mesh.SpaceDimension();
		std::unique_ptr<mfem::BilinearFormIntegrator> flux_integ(MakeStiffnessIntegrator());
		mfem::FiniteElementSpace flux_fes(&mesh, fec.get(), sdim);
		mfem::ZienkiewiczZhuEstimator estimator(*flux_integ, *A, flux_fes);
		estimator.SetWithCoeff(false);    // field = grad(A) or B; energy applies nu
		estimator.SetFluxAveraging(1);    // do not average across attribute interfaces
		errors = estimator.GetLocalErrors();

		// A zero/near-zero solution carries no energy and no meaningful relative
		// error; leave the indicator unscaled rather than dividing by ~0.
		const double energy = amr::FieldEnergy(*fespace, MakeStiffnessIntegrator(), *A);
		if (energy > 0.0) { errors /= std::sqrt(energy); }
	}

	// Peak flux density |B| over the current solution *A, sampled at element
	// nodes. AMR convergence diagnostic: the peak near a conductor corner or
	// high-permeability edge should settle as refinement resolves it.
	//
	// Planar: B = (dA/dy, -dA/dx), so |B| == |grad(A)| exactly.
	// Axisymmetric: B_r=-dA/dz, B_z=A/r+dA/dr, so the A/r term matters; reuse
	// MagneticFieldCoefficient (same B reconstruction as the exporters, incl.
	// the r->0 limit). Reflects whichever solution currently lives in *A.
	double ComputePeakFieldMagnitude() const override {
		if (!A) { return 0.0; }

		MagneticFieldCoefficient B_axi(A.get(), axisymmetric_mesh.tolerance);
		const bool axi = (geometry == GeometryType::Axisymmetric);

		double peak = 0.0;
		mfem::Vector B;
		for (int e = 0; e < fespace->GetNE(); ++e) {
			const mfem::FiniteElement* fe = fespace->GetFE(e);
			mfem::ElementTransformation* T = fespace->GetElementTransformation(e);
			const mfem::IntegrationRule& nodes = fe->GetNodes();
			for (int i = 0; i < nodes.GetNPoints(); ++i) {
				const mfem::IntegrationPoint& ip = nodes.IntPoint(i);
				T->SetIntPoint(&ip);
				if (axi) { B_axi.Eval(B, *T, ip); }  // true |B| incl. A/r term
				else { A->GetGradient(*T, B); }   // |B| == |grad(A)| (planar)
				const double mag = B.Norml2();
				if (mag > peak) { peak = mag; }
			}
		}
		return peak;
	}

	void ImprintScenario(const Scenario& sc, ImprintMode mode) {
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
		if (mode == ImprintMode::Field) {
			*b += neumann_rhs;
		}
	}

	// Solve + save on the CURRENT mesh/operators. Both analysis types flow through
	// ONE imprint -> solve -> save loop over BuildSolveScenarios() (prescribed
	// scenarios for Field; synthetic per-terminal unit-current drives for
	// CouplingMatrix). Shared output policy controls field serialization and each
	// AMR pass replaces the previous mesh's results.
	void RunOnCurrentMesh() override
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

	void SolveSystem() {
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

	// Post-solve field recovery: the vector potential A and the flux density B.
	// B = curl(A): the axisymmetric form (B_r = -dA/dz, B_z = dA/dr + A/r) needs
	// MagneticFieldCoefficient; the planar form is the standard curl of A_z.
	// Serialization is handled by the base class.
	FieldExportSet CollectExportFields() const override
	{
		FieldExportSet fields;
		fields.AddPrimary("A", *A);

		if (geometry == GeometryType::Axisymmetric) {
			fields.AddVector("B", std::make_unique<MagneticFieldCoefficient>(
				A.get(), axisymmetric_mesh.tolerance));
		}
		else {
			fields.AddVector("B", std::make_unique<PlanarMagneticFieldCoefficient>(A.get()));
		}
		return fields;
	}

	void SaveAnalysisResults() override
	{
		if (config.AnalysisType == AnalysisType::CouplingMatrix) {
			WriteCouplingMatrix();
		}
	}

	void WriteCouplingMatrix() {
		if (!L) {
			Reporter().Warning("WriteCouplingMatrix: coupling matrix not computed.");
			return;
		}

		SaveCouplingMatrix(*L, "Inductance Matrix " + CouplingUnitLabel("H"),
			"Inductance", "H");
	}

private:

	// Flux linkage of terminal `terminal_name` for the solution currently in *A.
	//
	//   lambda_k = integral over terminal k of (N_k/area_k) * A dV
	//
	// i.e. the winding functional of the MEASURED terminal applied to the field,
	// NOT the source of the driving scenario. With a unit-current drive on
	// terminal i, lambda_k is directly L(k,i) - the mutual inductance for k != i.
	// Mirrors MagnetoquasistaticSolver::ComputeStrandedFluxLinkage.
	double ComputeFluxLinkage(const std::string& terminal_name) const
	{
		mfem::Vector unit_density = BuildTerminalCurrentDensity(terminal_name, 1.0);
		mfem::PWConstCoefficient unit_density_coeff(unit_density);

		mfem::LinearForm winding_functional(fespace.get());
		winding_functional.AddDomainIntegrator(
			Geometry().NewDomainLFIntegrator(unit_density_coeff));
		winding_functional.Assemble();

		// The integrator carries the full geometric measure, so this is webers.
		return winding_functional * *A;
	}

	// Source current density for a scenario. Magnetostatics has no conductor-type
	// distinction, so every current terminal contributes.
	mfem::Vector BuildCurrentDensity(const Scenario& sc) const {
		return MagneticSolver::BuildCurrentDensity(
			sc, [](const Terminal&) { return true; });
	}
};
