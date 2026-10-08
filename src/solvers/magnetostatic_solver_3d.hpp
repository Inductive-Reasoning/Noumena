// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "vector_potential_solver_3d.hpp"
#include "../linalg/serial_ams.hpp"
#include "../linalg/gauged_direct_solver.hpp"

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
 *  - "direct": the system gauged by a Lagrange multiplier
 *    (GaugedDirectSolver), which imposes the same discrete Coulomb gauge
 *    exactly. Its fill-in limits it to moderate 3D problems.
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
	/// solution.
	double MagneticEnergy() const;

	/// Flux linkage lambda_k [Wb] of every terminal for the current solution,
	/// in config.Terminals (name) order.
	std::vector<double> FluxLinkages() const;

	/// The projected unit-current load of each terminal, in config.Terminals
	/// order (exposed for verification).
	const std::vector<mfem::Vector>& TerminalLoads() const { return terminal_loads; }

	void Setup() override;

	void BuildOperators() override;

	void RunOnCurrentMesh() override;

	// Post-solve fields: the potential A (a vector Nedelec field) and the flux
	// density B = curl A, evaluated exactly from the element basis.
	FieldExportSet CollectExportFields() const override;

	double ComputePeakFieldMagnitude() const override {
		return A ? PeakCurlMagnitude({ A.get() }) : 0.0;
	}

protected:
	void SaveAnalysisResults() override;

private:
	std::unique_ptr<mfem::GridFunction> A;
	std::unique_ptr<mfem::BilinearForm> a;
	std::unique_ptr<mfem::LinearForm> b;
	mfem::OperatorHandle A_op;
	std::unique_ptr<GaugedDirectSolver> direct_solver;
#ifdef MFEM_USE_MPI
	std::unique_ptr<SerialAmsPreconditioner> ams;  // iterative path
#endif

	mfem::VectorCoefficient* source = nullptr;          // not owned
	mfem::VectorCoefficient* boundary_value = nullptr;  // not owned

	std::vector<mfem::Vector> terminal_loads;  // projected 1 A loads, terminal order
	std::unique_ptr<mfem::DenseMatrix> L;      // inductance matrix (coupling runs)

	// A coupling column is the response to its terminal alone: the
	// programmatic source and tangential boundary data are background, left
	// out as in the 2D solvers' coupling runs, or every column of L would
	// carry the same background flux linkage.
	void ImprintScenario(const Scenario& scenario);

	void SolveSystem();
};
