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

	// The system operator before essential elimination, for the coupling
	// charge extraction Q = K*x. See BuildOperators().
	std::unique_ptr<mfem::SparseMatrix> K;
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

	void Setup() override;

	// Robin conditions ((eps dV/dn) + RobinCoeff * V = Value, outward normal)
	// are assembled into the operator; see BuildOperators(). The coefficient
	// must be non-negative: a negative one makes the operator indefinite, which
	// both the Cholesky factorization and PCG require it not to be.
	bool SupportsRobin() const override { return true; }

	// Driven electrodes pin their DOFs exactly as a Dirichlet condition does, so
	// every terminal marker joins the prescribed Dirichlet ones. Only the value
	// differs in origin, and that is supplied per scenario in ImprintScenario().
	void BuildEssentialBoundaryMarker() override;

	// (Re)build the FE space and every object bound to it for the CURRENT mesh.
	// Called once from Setup() and again after each AMR refinement. The
	// refinement-invariant data (fec, epsilon_coeff, ess_bdr, boundary_conditions,
	// terminal_markers) persists across calls and is reused.
	//
	// The constrained system matrix is assembled here; within a single mesh that
	// matrix is reused for every scenario / coupling column (the bilinear form
	// and essential-DOF set do not change between solves). AMR refinement
	// invalidates the mesh, so this is re-run to rebuild on the new mesh.
	void BuildOperators() override;

	// The constrained system matrix behind A_op. FormSystemMatrix always yields a
	// SparseMatrix for this serial build.
	mfem::SparseMatrix& SystemMatrix() const;

	// Create the domain diffusion integrator matching the active geometry. Used
	// both by the solve (owned by 'a') and the AMR error estimator (a separate,
	// independently-owned instance), so the estimated error is consistent with
	// the assembled operator: Div(eps Grad V) = 0 under the geometry's measure
	// (2*pi*r for axisymmetric, the plain Cartesian Laplacian in 2D and 3D).
	mfem::BilinearFormIntegrator* MakeStiffnessIntegrator() const {
		return Geometry().NewDiffusionIntegrator(*epsilon_coeff);
	}

	// The system operator's integrators: the domain stiffness plus the Robin
	// terms (RobinCoeff * V, v) over each Robin boundary, all under the
	// geometry's measure. The form takes ownership of the integrators; the
	// Robin coefficients and markers stay with this solver.
	void AddOperatorIntegrators(mfem::BilinearForm& form) const;

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
		EstimateRelativeZZError({ x.get() }, [this] { return MakeStiffnessIntegrator(); }, errors);
	}

	// Peak field magnitude |E| = |grad(V)| over the current solution *x, sampled
	// at element nodes. Used as an AMR convergence diagnostic (peak |E| at a
	// conductor corner should settle as refinement resolves the singularity).
	// Reflects whichever solution currently lives in *x (the last one solved).
	double ComputePeakFieldMagnitude() const override;

	// Field analysis imprints the configured boundary data (nonzero Dirichlet
	// values and the Neumann load). CouplingPerturbation analysis omits it:
	// a coupling coefficient is a derivative with respect to terminal drive,
	// so each column must be the homogeneous response to a unit terminal
	// excitation. Because the problem is linear, homogenizing here is exactly
	// equivalent to subtracting a background baseline solve, at one fewer solve.
	void ImprintScenario(const Scenario& sc, ImprintMode mode);

	// Solve + save on the CURRENT mesh/operators. Both analysis types flow through
	// ONE imprint -> solve loop over BuildSolveScenarios(); only the intrinsic
	// post-solve action differs: CouplingMatrix gathers an induced-charge column
	// into C. Shared output policy controls field serialization for either analysis.
	// Each AMR pass replaces the previous mesh's results.
	void RunOnCurrentMesh() override;

	void SolveSystem();

	// Post-solve field recovery: the potential V, the field E = -grad(V), and the
	// per-region permittivity. Serialization is handled by the base class.
	FieldExportSet CollectExportFields() const override;

	void SaveAnalysisResults() override;

private:
	// CouplingMatrix post-solve action for one column: with the just-solved
	// potential in *x (terminal `col` driven at 1 V, the rest grounded by the
	// synthesized scenario), gather the reaction charge Q = K*x onto every
	// conductor's boundary DOFs and write column `col` of C. Off-diagonals are
	// negative, diagonals positive. K carries the full geometric measure in
	// both planar and axisymmetric mode, so Q is already in coulombs and needs
	// no geometry-dependent scaling here.
	// The coupling solve is imprinted in CouplingPerturbation mode, so the load
	// vector is identically zero and K*x is the full reaction, with no RHS
	// contribution left to subtract.
	// Terminal order matches BuildSolveScenarios() / WriteCouplingMatrix()
	// (config.Terminals order).
	void GatherChargeColumn(int col);
};