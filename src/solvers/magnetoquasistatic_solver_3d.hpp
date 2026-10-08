// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <map>
#include <set>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "vector_potential_solver_3d.hpp"
#include "mqs_block_preconditioner.hpp"
#include "mqs_massive_port_operator.hpp"
#include "../coefficients/complex_vector_magnitude_coefficient.hpp"
#include "../coefficients/mqs_vector_electric_field.hpp"
#include "../core/constants.hpp"
#include "../linalg/serial_ams.hpp"
#include "../linalg/complex_direct_solver.hpp"

/**
 * @brief 3D time-harmonic magnetoquasistatics (eddy currents) in the vector
 *        potential A (H(curl)).
 *
 * Solves curl(nu curl A) + j omega sigma A = J_s + sigma V w for the phasor A
 * at each scenario frequency; see VectorPotentialSolver3D for the
 * discretization, boundary conditions, conductors and gauge.
 *
 * @par Conductors
 *  - Stranded terminals impose their current as a source, I / A_cs along the
 *    path, exactly as in 3D magnetostatics; no eddy currents are modelled in
 *    their strands.
 *  - Massive terminals are ports. The conductor's field is
 *    E = V w - j omega A, with w its DC conduction path (found with its own
 *    conductivity) and V a voltage unknown that makes its net current
 *    integral(sigma E . w) equal the imposed I:
 *        G V - j omega c^T A = I,   c_i = integral sigma w . N_i,
 *        G = integral sigma |w|^2  (the DC conductance).
 *    The block system is MqsMassivePortOperator, the same one the 2D solver
 *    uses. At low frequency the current returns to the DC distribution and
 *    R -> 1/G.
 *  - Any other conducting region (a shield, a tank wall) carries the eddy
 *    currents -j omega sigma A that the sigma mass term induces. A is the
 *    modified potential of the A-formulation: in a conductor its gradient part
 *    carries the electric scalar potential, so the eddy current is
 *    divergence-free with no normal component at the surface, weakly and with
 *    no extra unknown. A closed loop of it carries whatever net current the
 *    induction drives; 2D's "open" regions, which force that to zero, have no
 *    3D counterpart and are rejected.
 *
 * @par Coupling matrix
 * Per frequency, column j drives terminal j with 1 A. A massive row reads
 * its solved port voltage, R = Re V and L = Im V / omega; a stranded row its
 * flux linkage lambda = b'_k . A (b'_k its projected unit load), with
 * V = j omega lambda. Written as one R and one L matrix per frequency.
 *
 * @par Gauge
 * In the nonconducting regions curl-curl alone is singular. The direct
 * solver imposes the Coulomb gauge there by a Lagrange multiplier, with the
 * multiplier constant on each conductor so that the eddy-current equations
 * are left exactly as they are (see DivergenceFreeProjector::GaugeConstraint).
 * The iterative solver regularizes instead. Tested with a gradient grad(psi),
 * the regularized field equation reads
 *     integral (beta + j omega sigma) A . grad(psi) = 0,
 * so beta enters charge conservation in, and at the surface of, every
 * conductor: the eddy current is off by a relative beta / (omega sigma). The
 * static weight beta = kRegularization nu_min / L^2 is harmless for good
 * conductors, but not for weak ones. At 50 Hz it overstated the loss in a
 * sigma = 1 S/m block with ends by 68%. Confining beta to the nonconducting
 * regions does not help: the surface term remains. So beta is scaled to the
 * weakest conductor instead,
 *     beta = kRegularization min(nu_min / L^2, omega_min sigma_min),
 * over every scenario frequency and every conducting attribute, which keeps
 * beta / (omega sigma) <= kRegularization everywhere. It is floored at
 * kRegularization times the static weight to keep the null-space pivots
 * above round-off, with a warning if the floor binds.
 *
 * @par Linear solvers
 *  - "direct": the gauged complex system factored once per frequency (see
 *    ComplexDirectSolver) and reused for every terminal column.
 *  - "iterative" (MPI/HYPRE build only): FGMRES preconditioned by PRESB on
 *    the field blocks, with hypre's AMS on K + omega M_sigma, and the exact
 *    inverse of the port corner (see MqsBlockPreconditioner). The
 *    rank-N_ports border is left to FGMRES.
 */
class MagnetoquasistaticSolver3D : public VectorPotentialSolver3D {
public:
	MagnetoquasistaticSolver3D(mfem::Mesh& m, const ProblemConfig& c)
		: VectorPotentialSolver3D(m, c) {}

	const mfem::GridFunction& GetSolutionReal() const { return A->real(); }
	const mfem::GridFunction& GetSolutionImag() const { return A->imag(); }

	/// Solved complex voltage of massive terminal @p name for the current
	/// solution.
	std::complex<double> GetPortVoltage(const std::string& name) const;

	/// Time-averaged dissipation of every region that can dissipate.
	std::vector<RegionLoss> ComputeRegionLosses() const;

	void Setup() override;

	void BuildOperators() override;

	void RunOnCurrentMesh() override;

	// Real and imaginary parts of A and of B = curl A, |B| of the phasor and
	// the loss density.
	FieldExportSet CollectExportFields() const override;

	double ComputePeakFieldMagnitude() const override {
		return A ? PeakCurlMagnitude({ &A->real(), &A->imag() }) : 0.0;
	}

protected:
	void SaveAnalysisResults() override;

private:
	double frequency = 0.0;
	double omega = 0.0;

	std::unique_ptr<mfem::ConstantCoefficient> regularization;
	std::unique_ptr<mfem::BilinearForm> stiffness;   // curl-curl + beta mass
	std::unique_ptr<mfem::BilinearForm> sigma_mass;  // referenced by port_operator
	std::unique_ptr<MqsMassivePortOperator> port_operator;
	mfem::Array<int> ess_packed_tdofs;  // essential DOFs of the packed system

	std::vector<mfem::Vector> stranded_loads;  // by conductor; empty for massive
	std::vector<int> port_of;                  // conductor -> port index, or -1

	std::unique_ptr<mfem::ComplexGridFunction> A;
	std::vector<std::complex<double>> port_voltage;
	std::vector<ImpedancePoint> coupling_results;

	// Solver state for the active frequency (prepared_omega).
	double prepared_omega = 0.0;
	std::unique_ptr<mfem::SparseMatrix> packed_matrix;
	std::unique_ptr<GaugeConstraintRows> gauge;  // direct path; see BuildOperators()
	std::unique_ptr<ComplexDirectSolver> direct_solver;

#ifdef MFEM_USE_MPI
	std::unique_ptr<MqsBlockPreconditioner> preconditioner;  // AMS on the field blocks
#endif

	void ActivateFrequency(double f);

	// n x A = 0 forces the tangential E = -j omega A to zero on the wall, so
	// the wall is a perfect electrical contact: a conductor touching it can
	// pass eddy current into it and back out elsewhere. Right on a symmetry
	// plane that current crosses normally; on an outer box it shorts the
	// conductor's surface to the box. The electrodes of a terminal are meant
	// to touch such a wall and are not reported.
	void WarnOnConductorsTouchingContacts() const;

	// beta scaled to the weakest conductor; see the class comment.
	double EddyCurrentRegularization() const;

	std::complex<double> FluxLinkage(size_t k) const {
		return { stranded_loads[k] * A->real(), stranded_loads[k] * A->imag() };
	}

	// Solve for @p scenario's excitations at the active frequency.
	void Solve(const Scenario& scenario);

	// Factor, or build the preconditioner, for the active frequency; both are
	// reused across every terminal column at one frequency.
	void PrepareSolver();

	void SolveIteratively(const mfem::Vector& rhs, mfem::Vector& x);

	std::shared_ptr<const MqsVectorElectricField> MakeElectricField() const;
};
