// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <set>
#include <vector>

#include "mfem.hpp"
#include "../linalg/amg_preconditioner.hpp"
#include "../linalg/subset_solve.hpp"

/**
 * @brief Removes the gradient part of a Nedelec load vector, making the source
 *        of a curl-curl problem discretely divergence-free.
 *
 * The curl-curl operator annihilates the discrete gradients G psi (psi in the
 * matching H1 space, vanishing where A has an essential boundary). A load b
 * can only be balanced if it is orthogonal to all of them, G^T b = 0. A source
 * that is divergence-free in the continuum generally is not, discretely: a
 * faceted conductor surface, or a source direction sampled by quadrature,
 * leaves a small gradient component. An iterative solve then drifts in the null space,
 * and a regularized direct solve amplifies it by 1/beta into A, which pollutes
 * every flux linkage computed from A.
 *
 * The projection is purely algebraic, so it is exact whatever quadrature
 * produced b: with M the Nedelec mass matrix and K = G^T M G (which is exactly
 * the H1 stiffness matrix, by the de Rham sequence),
 *     K psi = G^T b,     b' = b - M G psi,
 * which gives G^T b' = 0 to solver precision. It is the M-orthogonal
 * projection of the source onto the complement of the gradients, i.e. the
 * discrete counterpart of J' = J - grad(psi) with div J' = 0.
 *
 * psi vanishes on the essential (n x A = 0) boundary, but only up to a
 * constant on each of its connected pieces: a psi that is 1 on one piece and
 * 0 on the others has a gradient with zero tangential trace on every wall,
 * which curl-curl annihilates as well. psi therefore lives in the reduced
 * space P (every DOF off the essential boundary, plus one value per wall
 * piece but the first), and the system solved is P^T K P.
 *
 * The H1 system is solved by multigrid-preconditioned CG to a tight
 * tolerance; the hierarchy is built once per mesh and reused for every load.
 *
 * A conductor's load can instead be projected within the conductor
 * (ProjectWithin), which keeps the correction current inside it.
 */
class DivergenceFreeProjector {
public:
	/// Relative residual of the H1 solve. What is left of the gradient part
	/// after the projection is invisible to B (curl grad = 0) and orthogonal,
	/// to this tolerance, to every flux-linkage functional, so 1e-10 is ample;
	/// much tighter targets hit the round-off floor of G^T b.
	static constexpr double kTolerance = 1e-10;

	/// @param nd       Nedelec space of the vector potential.
	/// @param ess_bdr  Boundary attributes where A is essential (n x A given):
	///                 psi is constant on each connected piece of it, and zero
	///                 on the first. With none, psi is fixed at one DOF to
	///                 remove the constant from K's null space.
	DivergenceFreeProjector(mfem::FiniteElementSpace& nd, const mfem::Array<int>& ess_bdr);

	/// Replace @p b by its divergence-free part (G^T b = 0 afterwards).
	///
	/// @return The squared L2 norm of what was removed, |grad psi|^2 =
	///         psi . G^T b: for a load b of a current density J, the part of
	///         J that was not divergence-free, to compare with |J|^2.
	double Project(mfem::Vector& b) const;

	/// Replace a conductor's load @p b by its divergence-free part within the
	/// conductor (domain-attribute marker @p conductor), on which @p b must be
	/// supported.
	///
	/// The same projection as Project(), with the Nedelec mass taken over the
	/// conductor's elements only, M_c, and psi free on its DOFs except where
	/// it touches an essential (n x A = 0) boundary:
	///     (G^T M_c G) psi = G^T b,     b' = b - M_c G psi.
	/// The correction -grad psi then lives in the conductor and the projected
	/// current has no normal component on its surface (psi is free there: the
	/// natural condition), so it stays inside instead of being made up through
	/// the surroundings. G^T b' = 0 still holds for every gradient of the mesh:
	/// b' is supported on the conductor's elements, so it sees a global
	/// gradient only through the conductor's own DOFs, which the local problem
	/// balances. A conductor part that touches no essential boundary has psi
	/// fixed at one DOF, removing the constant.
	///
	/// @return The squared L2 norm of what was removed, as for Project().
	double ProjectWithin(mfem::Vector& b, const mfem::Array<int>& conductor) const;

	/// Remove the gradient part of a solved potential @p A, leaving it
	/// M-orthogonal to every discrete gradient (the discrete Coulomb gauge,
	/// div A = 0 weakly). B = curl A is unchanged, as is every flux linkage
	/// computed with a projected load, and A's tangential trace on the
	/// essential boundary is kept (psi vanishes there).
	///
	/// The regularized direct solve lands in this gauge on its own; a solve
	/// of the singular system (AMS-preconditioned CG) leaves the gradient part
	/// arbitrary, so this makes both paths export the same A.
	void RemoveGradient(mfem::Vector& A) const;

	/// The discrete Coulomb gauge as a constraint, for a direct solve: the
	/// matrix C = M G P (Nedelec DOFs x multipliers) whose columns span the
	/// gradients the curl-curl system leaves undetermined, so that
	///
	///     [ K    C ]
	///     [ C^T  0 ]
	///
	/// is nonsingular and its solution satisfies C^T A = 0, div A = 0 weakly.
	/// For a balanced load the multiplier is zero, so K A = b is unchanged.
	///
	/// P maps the multiplier space into H1: functions that are constant on
	/// each connected piece of the essential boundary (zero on the first) and
	/// one constant on every conductor (domain attributes of @p conducting).
	/// There the eddy-current term already determines A, and charge
	/// conservation makes sigma A, not A, divergence-free, so the constraint
	/// must leave A free: a conductor part that touches no essential boundary
	/// gets a single multiplier (the gradients that are zero on it remain
	/// undetermined), one that does shares the value of the wall pieces it
	/// touches, which it joins into one. With no essential boundary at all one
	/// multiplier is dropped, removing the constant. The rows at essential
	/// Nedelec DOFs are included; see GaugeConstraintRows for how a solver uses
	/// them.
	std::unique_ptr<mfem::SparseMatrix> GaugeConstraint(const mfem::Array<int>& conducting) const;

	/// ||P^T G^T b||: zero for a balanced load.
	double GradientResidual(const mfem::Vector& b) const;

private:
	// The H1 DOFs of the elements with a marked domain attribute, grouped into
	// parts connected through shared elements; a part is grounded if one of
	// its DOFs lies on the essential boundary.
	struct DofParts {
		std::vector<int> part;       // by H1 DOF, -1 if not marked
		std::vector<bool> grounded;  // by part
	};
	DofParts ConnectedParts(const mfem::Array<int>& marker) const;

	// psi = P y with (P^T G^T M G P) y = P^T rhs.
	mfem::Vector SolvePotential(const mfem::Vector& rhs) const;

	// The connected pieces of the essential boundary, as a piece index by H1
	// DOF (-1 off it); pieces that share a DOF, even at one vertex, are one.
	std::vector<int> WallPieces(const mfem::Array<int>& marker, int& count) const;

	// The H1 x columns matrix with a one at (d, column[d]) where column[d] >= 0.
	std::unique_ptr<mfem::SparseMatrix> Columns(const std::vector<int>& column, int columns) const;

	mfem::FiniteElementSpace& nd;
	mfem::H1_FECollection h1_fec;
	mfem::FiniteElementSpace h1;
	std::unique_ptr<mfem::SparseMatrix> G;  // H1 -> Nedelec discrete gradient
	std::unique_ptr<mfem::SparseMatrix> M;  // Nedelec mass
	std::unique_ptr<mfem::SparseMatrix> P;  // reduced psi space -> H1
	std::unique_ptr<mfem::SparseMatrix> K;  // P^T G^T M G P
	std::set<int> essential;                // H1 DOFs on the essential boundary
	std::vector<int> wall_piece;            // by H1 DOF: piece of the essential boundary, or -1
	int wall_pieces = 0;
	std::unique_ptr<AmgPreconditioner> amg;
};
