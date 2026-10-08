// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <cmath>
#include <algorithm>
#include <sstream>

#include <map>
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

	std::unique_ptr<mfem::PWConstCoefficient> j_coeff; // stranded J_phi (axisym) or J (planar)

	// Each massive conductor's DC load for a unit voltage and its conductance
	// G on the current mesh (MagneticSolver::MassiveConductorLoad); a current
	// I loads the field with (I / G) * load.
	struct MassiveSource {
		mfem::Vector load;
		double conductance = 0.0;
	};
	std::map<std::string, MassiveSource> massive_sources;

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

	void Setup() override;

	void BuildOperators() override;

	// The constrained system matrix behind A_op. FormSystemMatrix always yields a
	// SparseMatrix for this serial build.
	mfem::SparseMatrix& SystemMatrix() const;

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
		EstimateRelativeZZError({ A.get() }, [this] { return MakeStiffnessIntegrator(); }, errors);
	}

	// Peak flux density |B| over the current solution *A, sampled at element
	// nodes. AMR convergence diagnostic: the peak near a conductor corner or
	// high-permeability edge should settle as refinement resolves it.
	//
	// Planar: B = (dA/dy, -dA/dx), so |B| == |grad(A)| exactly.
	// Axisymmetric: B_r=-dA/dz, B_z=A/r+dA/dr, so the A/r term matters; reuse
	// MagneticFieldCoefficient (same B reconstruction as the exporters, incl.
	// the r->0 limit). Reflects whichever solution currently lives in *A.
	double ComputePeakFieldMagnitude() const override;

	void ImprintScenario(const Scenario& sc, ImprintMode mode);

	// Solve + save on the CURRENT mesh/operators. Both analysis types flow through
	// ONE imprint -> solve -> save loop over BuildSolveScenarios() (prescribed
	// scenarios for Field; synthetic per-terminal unit-current drives for
	// CouplingMatrix). Shared output policy controls field serialization and each
	// AMR pass replaces the previous mesh's results.
	void RunOnCurrentMesh() override;

	void SolveSystem();

	// Post-solve field recovery: the vector potential A and the flux density B.
	// B = curl(A): the axisymmetric form (B_r = -dA/dz, B_z = dA/dr + A/r) needs
	// MagneticFieldCoefficient; the planar form is the standard curl of A_z.
	// Serialization is handled by the base class.
	FieldExportSet CollectExportFields() const override;

	void SaveAnalysisResults() override;

	void WriteCouplingMatrix();

private:

	// Flux linkage of terminal `terminal_name` for the solution currently in *A.
	//
	//   lambda_k = integral over terminal k of (N_k/area_k) * A dV
	//
	// i.e. the winding functional of the MEASURED terminal applied to the field,
	// NOT the source of the driving scenario. With a unit-current drive on
	// terminal i, lambda_k is directly L(k,i) - the mutual inductance for k != i.
	// Mirrors MagnetoquasistaticSolver::ComputeStrandedFluxLinkage.
	double ComputeFluxLinkage(const std::string& terminal_name) const;

	// Stranded conductors' uniform source current density for a scenario.
	// Massive conductors carry their DC distribution instead, loaded from
	// massive_sources in ImprintScenario(). Static excitations have no phase
	// (validated), so the density is real.
	mfem::Vector BuildCurrentDensity(const Scenario& sc) const;
};
