// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <cctype>
#include <complex>
#include <iomanip>
#include <limits>
#include <map>
#include <memory> // Required for smart pointers
#include <set>
#include <sstream>
#include "mfem.hpp"
#include "magnetic_solver.hpp"
#include "../axisym/axisymmetric_curl_curl_integrator.hpp"
#include "../axisym/axisymmetric_mass_integrator.hpp"
#include "../axisym/axisymmetric_lf_integrator.hpp"
#include "../coefficients/magnetic_field_coefficient.hpp"
#include "../coefficients/complex_vector_magnitude_coefficient.hpp"
#include "../coefficients/mqs_loss_density_coefficient.hpp"
#include "../core/constants.hpp"
#include "../config/boundary_validation.hpp"
#include "../core/problem_config.hpp"
#include "mqs_block_preconditioner.hpp"
#include "mqs_massive_port_operator.hpp"
#include "../linalg/amg_preconditioner.hpp"
#include "../linalg/complex_block_layout.hpp"
#include "../io/gmsh_results_writer.hpp"
#include "amr_support.hpp"
#include "../linalg/complex_direct_solver.hpp"

class MagnetoquasistaticSolver : public MagneticSolver {
    enum class ImprintMode { Field, CouplingPerturbation };

    double frequency = 0.0;
    mfem::real_t omega = 0.0;
    
    // Complex system objects. S_AA owns the real/imag field matrices referenced
    // by port_operator, so it is declared first and outlives that operator.
	std::unique_ptr<mfem::SesquilinearForm> S_AA;
	std::unique_ptr<mfem::ComplexGridFunction> A; // Complex field solution
	std::unique_ptr<mfem::Vector> Re_port_values; // Real part of port voltages
	std::unique_ptr<mfem::Vector> Im_port_values; // Imag part of port voltages
	std::unique_ptr<MqsMassivePortOperator> port_operator;
	std::unique_ptr<mfem::Vector> x_combined;
	std::unique_ptr<mfem::Vector> b_combined;

	// Direct-solve state for the packed real system. Both are rebuilt whenever
	// the mesh or the active frequency changes; factored_omega records which
	// frequency the current factors belong to.
	std::unique_ptr<mfem::SparseMatrix> packed_matrix;
	std::unique_ptr<ComplexDirectSolver> direct_solver;
	// Preconditioner of the iterative path, for one frequency.
	std::unique_ptr<MqsBlockPreconditioner> preconditioner;
	mfem::real_t preconditioned_omega = 0.0;
	mfem::real_t factored_omega = 0.0;

    // Coefficients
    mfem::Vector neumann_rhs;
    std::vector<mfem::real_t> port_conductances;
    
    // Two DISTINCT index spaces, both essential-DOF lists. Keeping them
    // separately named is deliberate: mixing them silently eliminates the wrong
    // rows, because the packed system is larger than the FE space.
    mfem::Array<int> ess_mesh_tdofs;    // indices into the FE space, [0, N_DOFs)
    mfem::Array<int> ess_packed_tdofs;  // indices into the packed block system,
                                        // [0, N_DOFs + N_Ports); each scalar
                                        // mesh DOF constrains its Re and Im copy.
                                        // Used INSTEAD of PhysicsSolver::ess_tdof_list,
                                        // which is an FE-space list and does not
                                        // apply to this formulation.

    struct MassivePortDefinition {
        std::string Name;
        std::vector<int> AttributeIds;
    };

    // The ordered list of ports that own a voltage unknown.
    //
    // The ordering is a contract, not an implementation detail: the solved
    // vectors Re_port_values / Im_port_values are indexed by position here, so
    // anything mapping a solved voltage back to a region must agree with the
    // order the operator was built from. It lives in one function for that
    // reason; a second copy of this loop would be free to drift.
    //
    // Explicit massive terminals come first so their solved voltage indices
    // retain terminal order. Passive open-current regions follow and receive an
    // identically zero current RHS in every scenario.
    //
    // Conductive regions with no current constraint (a flux shield, a steel
    // brace) are deliberately absent. They still carry eddy currents through the
    // sigma mass term, but nothing constrains their net current, so they have no
    // voltage unknown and their drive field is zero.
    std::vector<MassivePortDefinition> CollectMassivePorts() const;

    std::vector<ImpedancePoint> coupling_results;

    // The complex block system is solved as a single real vector laid out
    // [Re_Mesh, Re_Port, Im_Mesh, Im_Port]; ComplexPortVectorView and
    // ConstComplexPortVectorView (complex_block_layout.hpp) name the four slots
    // so callers never compute packed indices by hand.

    // Lift the essential boundary values currently projected into *A onto the
    // matching slots of the packed solution vector. FormLinearSystem constrains
    // essential DOFs to whatever it finds there, so without this the projected
    // non-zero Dirichlet values would be forced back to zero.
    void LiftEssentialInto(mfem::Vector& x_packed) const;

	void RecoverSolvedUnknowns(mfem::Vector& x_packed);

	// Per-attribute drive amplitudes from the solved port voltages, indexed by
	// (attribute - 1) and sized to the mesh.
	//
	// The tables are zero-initialised and written only where a port unknown
	// exists. That default is load-bearing rather than defensive: conductive
	// regions with no current constraint (a flux shield, a steel brace) carry
	// eddy currents through the sigma mass term but own no voltage unknown, so
	// zero is their physically correct drive and the general loss expression
	// collapses to 0.5*sigma*omega^2*|A|^2 for them with no special case.
	//
	// Ordering comes from CollectMassivePorts(), the same function the operator
	// was built from, so voltage index p always refers to the same region here
	// as it does there.
	void BuildDriveTables(std::vector<double>& drive_re,
						  std::vector<double>& drive_im) const;

public:
	// Solved complex vector potential. Exposed const so verification code can
	// recompute derived quantities independently of the solver's own paths.
	const mfem::GridFunction& GetSolutionReal() const { return A->real(); }
	const mfem::GridFunction& GetSolutionImag() const { return A->imag(); }

	// Solved complex voltage of a named massive port, as (real, imaginary).
	//
	// Looked up through CollectMassivePorts() so the index always refers to the
	// same region the operator was assembled from.
	std::pair<double, double> GetPortVoltage(const std::string& port_name) const;

	// Time-averaged dissipation of every region that can dissipate. Public
	// because it is a result of the analysis in its own right.
	std::vector<RegionLoss> ComputeRegionLosses() const;

public:
	// Constructor deals only with initialization, no manual nullptr assignment needed
	MagnetoquasistaticSolver(mfem::Mesh &m, const ProblemConfig &c) : MagneticSolver(m, c) {}

	void Setup() override;

	void BuildOperators() override;

	mfem::BilinearFormIntegrator* MakeMassIntegrator() {
		return Geometry().NewMassIntegrator(*sigma_coeff);
	}

    void ActivateFrequency(const Scenario& sc);

    void ImprintScenario(const Scenario& sc, ImprintMode mode);

    // Solve + save on the CURRENT mesh/operators. Field analysis performs one
    // solve per prescribed scenario. Coupling analysis uses each unique frequency
    // and synthesizes one unit-current solve per terminal.
    void RunOnCurrentMesh() override;

    void SolveSystem();

    // Assemble and factor the packed real system for the currently active
    // frequency, reusing the existing factors if the frequency has not moved.
    //
    // Unlike the static solvers, the MQS matrix is NOT constant over a run: both
    // the sigma-mass block (omega*M_sigma) and the port corner (G_dc/omega)
    // depend on frequency. It is, however, constant across the terminal columns
    // of a single frequency point, which is where the reuse pays off: one
    // factorization serves every terminal at that frequency.
    void EnsurePreconditionerForActiveFrequency();

    void EnsureFactorizationForActiveFrequency();

    // Estimate per-element error on the CURRENT mesh. The scenario-wide fold (a
    // running RMS over every scenario / frequency point) lives in the base class
    // AccumulateScenarioError(), so one shared mesh is refined for all scenarios
    // (spec: identical $Nodes/$Elements across every <scenario>.results.msh).
    //
    // Uses the serial recovery-based ZienkiewiczZhuEstimator (the L2 variant is
    // MPI-only). A dedicated integrator instance (separate from a's) and an H1
    // vector flux space are constructed here per call; SetFluxAveraging(1) keeps
    // the recovered flux from smoothing across material-attribute interfaces so
    // per-region reluctivity discontinuities are respected. The real and
    // imaginary indicators are combined as a complex magnitude.
    //
    // That magnitude is then normalized by the total field energy of the phasor,
    // Et = 0.5*(Re^T K Re) + 0.5*(Im^T K Im), making it a dimensionless RELATIVE
    // error. This matters more here than in the static solvers: an MQS run
    // sweeps frequency, and the field energy of a fixed excitation varies by
    // orders of magnitude across a decade sweep as eddy currents screen the
    // conductor. Folding raw indicators would let the highest-energy frequency
    // point set the mesh for the whole sweep.
    //
    // @param errors  Output: per-element error indicator (sized to NE).
    void EstimateCurrentSolutionError(mfem::Vector& errors) override;

    // Peak flux density |B| over the current solution *A, sampled at element
    // nodes. AMR convergence diagnostic: the peak near a conductor corner or
    // high-permeability edge should settle as refinement resolves it.
    //
    // Planar: B = (dA/dy, -dA/dx), so |B| == |grad(A)| exactly.
    // Axisymmetric: B_r=-dA/dz, B_z=A/r+dA/dr, so the A/r term matters; reuse
    // MagneticFieldCoefficient (same B reconstruction as the exporters, incl.
    // the r->0 limit). Reflects whichever solution currently lives in *A.
    double ComputePeakFieldMagnitude() const override;


	// Post-solve field recovery for the complex solution: the real/imaginary
	// vector-potential parts (primaries), the real/imaginary flux densities
	// B = curl(A) (derived vectors, geometry-dependent), and the complex flux
	// magnitude |B| = sqrt(|Re B|^2 + |Im B|^2). Serialization is handled by the
	// base class.
	FieldExportSet CollectExportFields() const override;

	void GatherCouplingColumn(const Scenario& sc);

	void PrepareAnalysis();

    void BeginCouplingPoint(double point_frequency);

    void SaveAnalysisResults() override;

    std::pair<double, double> ComputeStrandedFluxLinkage(
        const std::string& terminal_name) const;

    // Stranded-conductor source current density for a scenario. Massive
    // conductors are driven through the port block instead, so they are
    // excluded here.
    void BuildCurrentDensity(const Scenario& sc, mfem::Vector& j_re,
                             mfem::Vector& j_im) const;
};