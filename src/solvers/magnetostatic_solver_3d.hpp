// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "vector_potential_solver_3d.hpp"
#include "../linalg/serial_ams.hpp"
#include "../linalg/sparse_direct_solver.hpp"

/**
 * @brief 3D magnetostatics in the full vector potential A (H(curl)).
 *
 * Solves curl(nu curl A) = J on a 3D mesh; see VectorPotentialSolver3D for the
 * discretization, boundary conditions, conductors and gauge.
 *
 * @par Sources
 * Each terminal's 1 A current density is assembled once per mesh and made
 * discretely divergence-free (DivergenceFreeProjector); a scenario's load is
 * the excitation-weighted sum. A stranded conductor carries I / A_cs along its
 * path; an azimuthal one's is exactly the 2D axisymmetric model's I/area
 * source revolved. A massive conductor carries its DC distribution
 * sigma w I / G. A programmatic source (SetSourceCurrentDensity) is added to
 * the load and projected the same way.
 *
 * @par Coupling matrix
 * Column j drives terminal j with 1 A; entry (k, j) is the flux linkage
 * lambda_k = integral(A . J_k) with J_k terminal k's unit-current density,
 * evaluated as b'_k . A with the projected load b'_k. That is the same winding
 * functional the 2D solvers use, it is gauge-invariant because b'_k is
 * orthogonal to the gradients, and it makes L = B'^T K^-1 B' symmetric by
 * construction. For a massive conductor it is the DC inductance, whose
 * energy-weighted average over the conductor accounts for the nonuniform
 * current. Units are henries.
 *
 * @par Linear solvers
 *  - "iterative" (MPI/HYPRE build only): CG preconditioned by hypre's AMS
 *    (SerialAmsPreconditioner) on the singular system itself, which is
 *    consistent because every load is projected. The iteration count stays
 *    roughly constant under refinement. CG leaves A's gradient part
 *    arbitrary, so it is removed afterwards
 *    (DivergenceFreeProjector::RemoveGradient), putting A in the discrete
 *    Coulomb gauge.
 *  - "direct": the regularized system (see VectorPotentialSolver3D), which
 *    lands in the same gauge. Its fill-in limits it to small 3D problems.
 */
class MagnetostaticSolver3D : public VectorPotentialSolver3D {
public:
	MagnetostaticSolver3D(mfem::Mesh& m, const ProblemConfig& c)
		: VectorPotentialSolver3D(m, c) {}

	/// Source current density J [A/m^2], applied to every scenario. Not owned;
	/// must outlive the solve. nullptr (the default) means no source.
	void SetSourceCurrentDensity(mfem::VectorCoefficient* J) { source = J; }

	/// Tangential boundary data g, imposing n x A = n x g on every Dirichlet
	/// boundary. Not owned. nullptr (the default) means n x A = 0. Used by the
	/// manufactured-solution tests; the configured conditions are homogeneous.
	void SetTangentialBoundaryValue(mfem::VectorCoefficient* g) { boundary_value = g; }

	/// The solved vector potential (Nedelec grid function).
	const mfem::GridFunction& GetSolution() const { return *A; }

	/// Magnetic energy W = 1/2 integral(nu |curl A|^2) [J] of the current
	/// solution, excluding the regularization term.
	double MagneticEnergy() const {
		mfem::BilinearForm k(fespace.get());
		k.AddDomainIntegrator(new mfem::CurlCurlIntegrator(*nu_coeff));
		k.Assemble();
		k.Finalize();
		return 0.5 * k.InnerProduct(*A, *A);
	}

	/// Flux linkage lambda_k [Wb] of every terminal for the current solution,
	/// in config.Terminals (name) order.
	std::vector<double> FluxLinkages() const {
		std::vector<double> lambda;
		for (const auto& load : terminal_loads) { lambda.push_back(load * *A); }
		return lambda;
	}

	/// The projected unit-current load of each terminal, in config.Terminals
	/// order (exposed for verification).
	const std::vector<mfem::Vector>& TerminalLoads() const { return terminal_loads; }

	void Setup() override {
		InitializeVectorPotential();
		BuildOperators();
		ValidateVectorPotentialBoundaries();
	}

	void BuildOperators() override {
		BuildSpaceAndConductors();
		A = std::make_unique<mfem::GridFunction>(fespace.get());
		*A = 0.0;

		const bool direct = config.LinearSolver == LinearSolverType::Direct;
		a = std::make_unique<mfem::BilinearForm>(fespace.get());
		a->AddDomainIntegrator(new mfem::CurlCurlIntegrator(*nu_coeff));
		if (direct) {
			regularization = std::make_unique<mfem::ConstantCoefficient>(RegularizationWeight());
			a->AddDomainIntegrator(new mfem::VectorFEMassIntegrator(*regularization));
		}
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
			direct_solver = std::make_unique<SparseDirectSolver>(*matrix);
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

	void RunOnCurrentMesh() override {
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

	// Post-solve fields: the potential A (a vector Nedelec field) and the flux
	// density B = curl A, evaluated exactly from the element basis.
	FieldExportSet CollectExportFields() const override {
		FieldExportSet fields;
		fields.AddPrimary("A", *A);
		fields.AddVector("B", std::make_unique<mfem::CurlGridFunctionCoefficient>(A.get()));
		return fields;
	}

	double ComputePeakFieldMagnitude() const override {
		return A ? PeakCurlMagnitude({ A.get() }) : 0.0;
	}

protected:
	void SaveAnalysisResults() override {
		if (config.AnalysisType != AnalysisType::CouplingMatrix) return;
		if (!L) {
			Reporter().Warning("WriteCouplingMatrix: coupling matrix not computed.");
			return;
		}
		SaveCouplingMatrix(*L, "Inductance Matrix " + CouplingUnitLabel("H"),
			"Inductance", "H");
	}

private:
	std::unique_ptr<mfem::GridFunction> A;
	std::unique_ptr<mfem::ConstantCoefficient> regularization;
	std::unique_ptr<mfem::BilinearForm> a;
	std::unique_ptr<mfem::LinearForm> b;
	mfem::OperatorHandle A_op;
	std::unique_ptr<SparseDirectSolver> direct_solver;
#ifdef MFEM_USE_MPI
	std::unique_ptr<SerialAmsPreconditioner> ams;  // iterative path
#endif

	mfem::VectorCoefficient* source = nullptr;          // not owned
	mfem::VectorCoefficient* boundary_value = nullptr;  // not owned

	std::vector<mfem::Vector> terminal_loads;  // projected 1 A loads, terminal order
	std::unique_ptr<mfem::DenseMatrix> L;      // inductance matrix (coupling runs)

	void ImprintScenario(const Scenario& scenario) {
		*A = 0.0;
		if (boundary_value) {
			A->ProjectBdrCoefficientTangent(*boundary_value, ess_bdr);
		}
		b = std::make_unique<mfem::LinearForm>(fespace.get());
		if (source) {
			b->AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(*source));
		}
		b->Assemble();
		if (source) { projector->Project(*b); }

		for (size_t k = 0; k < conductors.size(); ++k) {
			const double current = ExcitationFor(scenario, conductors[k].Name);
			if (current != 0.0) { b->Add(current, terminal_loads[k]); }
		}
	}

	void SolveSystem() {
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
};
