// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "magnetic_solver.hpp"
#include "divergence_free_projector.hpp"
#include "../coefficients/conductor_path.hpp"
#include "../config/boundary_validation.hpp"
#include "../parallel/mpi_runtime.hpp"

/**
 * @brief What the 3D magnetic solvers share: the vector potential A in a
 *        Nedelec (H(curl)) space, its boundary conditions and gauge, and the
 *        current-carrying conductors that drive it.
 *
 * The tangential continuity of a Nedelec field across element faces -- and
 * therefore the normal continuity of B = curl A -- holds exactly, so material
 * interfaces need no special treatment.
 *
 * @par Conductors
 * Every current terminal is a conductor: a volume group plus a "direction"
 * (CurrentDirection) that says where its current flows -- azimuthally about an
 * axis, between electrodes, or around a closed loop through a cut. Its
 * ConductorPath (conductor_path.hpp) turns that into the direction field w;
 * how the current spreads over the conductor then follows from its type:
 * uniformly along the path (stranded) or as the conduction current sigma E of
 * a massive conductor, whose path is found with its own conductivity.
 *
 * @par Boundary conditions
 * A 3D vector potential has no meaningful scalar boundary value, so the
 * configured conditions are restricted to their homogeneous forms:
 *  - Dirichlet (value 0): n x A = 0, "flux tangent" -- B has no normal
 *    component (magnetic insulation, or a symmetry plane B cannot cross);
 *  - Neumann (value 0) or no entry: the natural condition n x H = 0,
 *    "flux normal" -- B crosses the boundary normally.
 * Current can enter or leave the model only through n x A = 0 walls, so
 * electrodes must lie on one.
 *
 * @par Gauge and regularization
 * The curl-curl operator annihilates gradients, so it is singular even with
 * n x A fixed on the whole boundary: every gradient of a nodal function that
 * vanishes on the n x A = 0 walls is in its null space. Sources are made
 * orthogonal to those gradients (DivergenceFreeProjector), so the singular
 * system is consistent. The direct solvers impose the discrete Coulomb gauge
 * exactly instead, by a Lagrange multiplier
 * (DivergenceFreeProjector::GaugeConstraint), which leaves the field
 * equations untouched. The iterative MQS solver, which needs a nonsingular
 * matrix, adds a small mass term beta (A, w),
 *     beta = kRegularization * nu_min / L^2,
 * with L the mesh bounding-box diagonal and nu_min the smallest reluctivity.
 * With an orthogonal source this selects the Coulomb-gauged solution and
 * perturbs B by a relative O(kRegularization) in every material, while keeping
 * the null-space pivots (relative size kRegularization * (nu_min/nu_max) *
 * (h/L)^2) above round-off. In an eddy-current solve beta also enters charge
 * conservation in the conductors, so there it is scaled down to the weakest
 * conductor (MagnetoquasistaticSolver3D).
 *
 * Not yet available, and rejected in Setup(): adaptive refinement.
 */
class VectorPotentialSolver3D : public MagneticSolverBase {
public:
	/// Relative size of the regularizing mass term; see the class comment.
	static constexpr double kRegularization = 1e-6;


	/// Largest fraction of a conductor's current density (L2 norm) the
	/// divergence-free projection may remove before it is reported; see
	/// ProjectedUnitCurrentLoad. The removed part is orthogonal to the kept
	/// one, so its effect on energies and inductances is about the fraction
	/// squared.
	static constexpr double kMaxProjectedFraction = 0.02;

	const DivergenceFreeProjector& Projector() const { return *projector; }

protected:
	/// A current terminal's conductor on the current mesh.
	struct TerminalConductor {
		std::string Name;
		ConductorType Type = ConductorType::Stranded;
		CurrentDirection::Kind Direction = CurrentDirection::Kind::Azimuthal;
		mfem::Array<int> Marker;              // domain attributes
		std::unique_ptr<ConductorPath> Path;
		/// Stranded: the cross-section A_cs = integral |w|. Massive: the DC
		/// conductance G = integral sigma |w|^2.
		double PathIntegral = 0.0;
		double Turns = 1.0;                   // stranded only; massive is 1
	};

	/// Terminal conductors in config.Terminals (name) order. Rebuilt per mesh
	/// by BuildConductors().
	std::vector<TerminalConductor> conductors;
	std::unique_ptr<DivergenceFreeProjector> projector;

	VectorPotentialSolver3D(mfem::Mesh& m, const ProblemConfig& c) : MagneticSolverBase(m, c) {}

	// The part of Setup() before the FE space: checks, material tables, the
	// Nedelec collection and the essential-boundary marker.
	void InitializeVectorPotential();

	// The part of Setup() after BuildOperators().
	void ValidateVectorPotentialBoundaries();

	// The Nedelec space, its essential DOFs, the gradient projector and the
	// terminal conductors for the current mesh.
	void BuildSpaceAndConductors();

	/// sigma for a massive conductor, nullptr for a stranded one (the
	/// convention of ConductorCurrentCoefficient and ConductorPathIntegral).
	mfem::Coefficient* ConductivityOf(const TerminalConductor& c) const {
		return c.Type == ConductorType::Massive ? sigma_coeff.get() : nullptr;
	}

	/// Load vector of the current density ConductorCurrentCoefficient(scale)
	/// of @p c: stranded, scale * w / |w|; massive, scale * sigma * w.
	mfem::Vector AssembleConductorLoad(const TerminalConductor& c, double scale);

	/// The divergence-free load of 1 A in each turn of @p c: J = N w / (|w| A_cs)
	/// stranded with N turns, the DC distribution sigma w / G massive. Its
	/// product with A is the flux linkage of all N turns.
	///
	/// The load is projected within the conductor
	/// (DivergenceFreeProjector::ProjectWithin), so the result is the nearest
	/// divergence-free current that stays inside it. A well-posed current is
	/// divergence-free in the continuum and has no normal component on its
	/// conductor's surface (except at electrodes), so the projection removes
	/// only discretization-level imbalance from it. A sizable removal means
	/// the current as given does not balance in its conductor and the
	/// projection has redistributed it, which is reported. That happens with an azimuthal direction about
	/// the wrong axis, or on a conductor that is not a body of revolution
	/// about it (including, to a few percent, a coarsely faceted round one),
	/// and with a stranded current -- uniform along its path -- in a
	/// conductor whose cross-section varies along it or that has a dead-end
	/// branch. It is reported as the fraction |grad psi| / |J| of the
	/// current's L2 norm that was removed.
	///
	/// Projecting within the conductor keeps the current: the removed
	/// gradient carries none along the path, since integral grad(psi) . w = 0
	/// for the harmonic path w (no lateral flux, psi = 0 at electrodes).
	mfem::Vector ProjectedUnitCurrentLoad(const TerminalConductor& c);

	double RegularizationWeight() const;

	/// Peak of sqrt(sum_i |curl a_i|^2) over quadrature points: |B| of a real
	/// field ({A}) or of a phasor ({Re A, Im A}).
	double PeakCurlMagnitude(std::initializer_list<const mfem::GridFunction*> parts) const;

	void EstimateCurrentSolutionError(mfem::Vector&) override {
		MFEM_ABORT("3D magnetics does not yet support adaptive refinement.");
	}

private:
	TerminalConductor BuildConductor(const std::string& name, const Terminal& term);

	// The angular extent of an azimuthal conductor, over which its
	// conduction potential falls by 1: 2 pi for a full ring, less for a
	// sector whose ends lie on n x A = 0 walls -- a symmetry model, cut by
	// meridian planes the current crosses normally. The ends are the
	// conductor's faces on such walls that phi-hat crosses (|n . phi-hat| >
	// 1/2; a ring merely lying against a wall is crossed by none), current
	// entering where phi-hat points in and leaving where it points out. The
	// extent is the arc from the one to the other along phi-hat.
	double AzimuthalExtent(const std::string& name, const AzimuthalPath& frame,
						   const mfem::Array<int>& conductor) const;

	// The connected pieces of the n x A = 0 boundary: a piece number for each
	// boundary element on it (-1 elsewhere). Elements sharing a vertex are in
	// one piece, since a continuous potential cannot differ between them.
	std::vector<int> WallPieces() const;

	std::unique_ptr<ConductorPath> MakeConductorPath(
		const std::string& name, const CurrentDirection& d,
		const mfem::Array<int>& conductor, mfem::Coefficient* conductivity);
};
