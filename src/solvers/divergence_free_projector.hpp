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
	///                 psi vanishes there. With none, psi is fixed at one DOF
	///                 to remove the constant from K's null space.
	DivergenceFreeProjector(mfem::FiniteElementSpace& nd, const mfem::Array<int>& ess_bdr)
		: nd(nd), h1_fec(nd.GetMaxElementOrder(), nd.GetMesh()->Dimension()),
		  h1(nd.GetMesh(), &h1_fec) {
		mfem::DiscreteLinearOperator gradient(&h1, &nd);
		gradient.AddDomainInterpolator(new mfem::GradientInterpolator);
		gradient.Assemble();
		gradient.Finalize();
		G.reset(gradient.LoseMat());

		mfem::ConstantCoefficient one(1.0);
		mfem::BilinearForm mass(&nd);
		mass.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(one));
		mass.Assemble();
		mass.Finalize();
		M.reset(mass.LoseMat());

		K.reset(mfem::RAP(*G, *M, *G));  // G^T M G

		mfem::Array<int> marker(ess_bdr);
		h1.GetEssentialTrueDofs(marker, fixed);
		for (int i = 0; i < fixed.Size(); ++i) { essential.insert(fixed[i]); }
		if (fixed.Size() == 0) { fixed.Append(0); }
		for (int i = 0; i < fixed.Size(); ++i) {
			K->EliminateRowCol(fixed[i], mfem::Operator::DIAG_ONE);
		}
		K->Finalize();
		amg = std::make_unique<AmgPreconditioner>(*K);
	}

	/// Replace @p b by its divergence-free part (G^T b = 0 afterwards).
	///
	/// @return The squared L2 norm of what was removed, |grad psi|^2 =
	///         psi . G^T b: for a load b of a current density J, the part of
	///         J that was not divergence-free, to compare with |J|^2.
	double Project(mfem::Vector& b) const {
		mfem::Vector rhs(G->Width());
		G->MultTranspose(b, rhs);
		for (int i = 0; i < fixed.Size(); ++i) { rhs(fixed[i]) = 0.0; }
		const mfem::Vector psi = SolvePotential(rhs);

		mfem::Vector grad_psi(G->Height()), correction(M->Height());
		G->Mult(psi, grad_psi);
		M->Mult(grad_psi, correction);
		b -= correction;
		return psi * rhs;
	}

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
	double ProjectWithin(mfem::Vector& b, const mfem::Array<int>& conductor) const {
		mfem::ConstantCoefficient one(1.0);
		mfem::Array<int> marker(conductor);
		mfem::BilinearForm mass(&nd);
		mass.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(one), marker);
		mass.Assemble();
		mass.Finalize();
		const mfem::SparseMatrix& M_c = mass.SpMat();
		std::unique_ptr<mfem::SparseMatrix> K_c(mfem::RAP(*G, M_c, *G));

		// psi is free on the conductor's H1 DOFs off the essential boundary,
		// except one per connected part that touches no essential boundary.
		const DofParts parts = ConnectedParts(conductor);
		std::vector<int> free;
		std::vector<bool> pinned(parts.grounded.size(), false);
		for (int d = 0; d < h1.GetVSize(); ++d) {
			const int c = parts.part[d];
			if (c < 0 || essential.count(d)) continue;
			if (!parts.grounded[c] && !pinned[c]) { pinned[c] = true; continue; }
			free.push_back(d);
		}

		mfem::Vector rhs(G->Width()), psi(G->Width());
		G->MultTranspose(b, rhs);
		psi = 0.0;
		SolveOnSubset(*K_c, rhs, free, psi, kTolerance, "The divergence-free projection");

		mfem::Vector grad_psi(G->Height()), correction(M_c.Height());
		G->Mult(psi, grad_psi);
		M_c.Mult(grad_psi, correction);
		b -= correction;
		double removed = 0.0;
		for (int d : free) { removed += psi(d) * rhs(d); }
		return removed;
	}

	/// Remove the gradient part of a solved potential @p A, leaving it
	/// M-orthogonal to every discrete gradient (the discrete Coulomb gauge,
	/// div A = 0 weakly). B = curl A is unchanged, as is every flux linkage
	/// computed with a projected load, and A's tangential trace on the
	/// essential boundary is kept (psi vanishes there).
	///
	/// The regularized direct solve lands in this gauge on its own; a solve
	/// of the singular system (AMS-preconditioned CG) leaves the gradient part
	/// arbitrary, so this makes both paths export the same A.
	void RemoveGradient(mfem::Vector& A) const {
		mfem::Vector MA(M->Height()), rhs(G->Width());
		M->Mult(A, MA);
		G->MultTranspose(MA, rhs);
		const mfem::Vector psi = SolvePotential(rhs);

		mfem::Vector grad_psi(G->Height());
		G->Mult(psi, grad_psi);
		A -= grad_psi;
	}

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
	/// P maps the multiplier space into H1: functions that vanish on the
	/// essential boundary and are one constant on every conductor (domain
	/// attributes of @p conducting). There the eddy-current term already
	/// determines A, and charge conservation makes sigma A, not A,
	/// divergence-free, so the constraint must leave A free: a conductor
	/// part that touches no essential boundary gets a single multiplier
	/// (the gradients that are zero on it remain undetermined), one that does
	/// gets none. With no essential boundary at all one multiplier is
	/// dropped, removing the constant. The rows at essential Nedelec DOFs are
	/// included; see GaugeConstraintRows for how a solver uses them.
	std::unique_ptr<mfem::SparseMatrix> GaugeConstraint(const mfem::Array<int>& conducting) const {
		const DofParts parts = ConnectedParts(conducting);
		const int n_h1 = h1.GetVSize();
		std::vector<int> column(n_h1, -1), part_column(parts.grounded.size(), -1);
		int multipliers = 0;
		for (int d = 0; d < n_h1; ++d) {
			if (essential.count(d)) continue;
			const int c = parts.part[d];
			if (c < 0) { column[d] = multipliers++; continue; }
			if (parts.grounded[c]) continue;
			if (part_column[c] < 0) { part_column[c] = multipliers++; }
			column[d] = part_column[c];
		}
		if (essential.empty() && multipliers > 0) {
			for (int& j : column) { j = j == 0 ? -1 : (j > 0 ? j - 1 : j); }
			--multipliers;
		}

		mfem::SparseMatrix P(n_h1, multipliers);
		for (int d = 0; d < n_h1; ++d) {
			if (column[d] >= 0) { P.Set(d, column[d], 1.0); }
		}
		P.Finalize();
		std::unique_ptr<mfem::SparseMatrix> MG(mfem::Mult(*M, *G));
		return std::unique_ptr<mfem::SparseMatrix>(mfem::Mult(*MG, P));
	}

	/// ||G^T b|| (excluding the fixed DOFs): zero for a balanced load.
	double GradientResidual(const mfem::Vector& b) const {
		mfem::Vector r(G->Width());
		G->MultTranspose(b, r);
		for (int i = 0; i < fixed.Size(); ++i) { r(fixed[i]) = 0.0; }
		return r.Norml2();
	}

private:
	// The H1 DOFs of the elements with a marked domain attribute, grouped into
	// parts connected through shared elements; a part is grounded if one of
	// its DOFs lies on the essential boundary.
	struct DofParts {
		std::vector<int> part;       // by H1 DOF, -1 if not marked
		std::vector<bool> grounded;  // by part
	};
	DofParts ConnectedParts(const mfem::Array<int>& marker) const {
		const mfem::Mesh& mesh = *nd.GetMesh();
		DofParts parts;
		parts.part.assign(h1.GetVSize(), -1);
		std::vector<std::vector<int>> dof_elements(h1.GetVSize());
		mfem::Array<int> dofs;
		for (int e = 0; e < mesh.GetNE(); ++e) {
			const int a = mesh.GetAttribute(e);
			if (a < 1 || a > marker.Size() || !marker[a - 1]) continue;
			h1.GetElementDofs(e, dofs);
			for (int d : dofs) { dof_elements[d].push_back(e); }
		}
		for (int start = 0; start < h1.GetVSize(); ++start) {
			if (dof_elements[start].empty() || parts.part[start] >= 0) continue;
			const int c = static_cast<int>(parts.grounded.size());
			parts.grounded.push_back(false);
			std::vector<int> stack{ start };
			parts.part[start] = c;
			while (!stack.empty()) {
				const int d = stack.back();
				stack.pop_back();
				if (essential.count(d)) { parts.grounded[c] = true; }
				for (int e : dof_elements[d]) {
					h1.GetElementDofs(e, dofs);
					for (int m : dofs) {
						if (parts.part[m] < 0) { parts.part[m] = c; stack.push_back(m); }
					}
				}
			}
		}
		return parts;
	}

	// psi with K psi = rhs, psi = 0 at the fixed DOFs.
	mfem::Vector SolvePotential(mfem::Vector rhs) const {
		for (int i = 0; i < fixed.Size(); ++i) { rhs(fixed[i]) = 0.0; }
		mfem::Vector psi(G->Width());
		psi = 0.0;
		mfem::CGSolver cg;
		cg.SetOperator(*K);
		cg.SetPreconditioner(*amg);
		cg.SetRelTol(kTolerance);
		cg.SetAbsTol(0.0);
		cg.SetMaxIter(1000);
		cg.SetPrintLevel(0);
		cg.Mult(rhs, psi);
		MFEM_VERIFY(cg.GetConverged() || rhs.Norml2() == 0.0,
			"Divergence-free projection did not converge.");
		return psi;
	}

	mfem::FiniteElementSpace& nd;
	mfem::H1_FECollection h1_fec;
	mfem::FiniteElementSpace h1;
	std::unique_ptr<mfem::SparseMatrix> G;  // H1 -> Nedelec discrete gradient
	std::unique_ptr<mfem::SparseMatrix> M;  // Nedelec mass
	std::unique_ptr<mfem::SparseMatrix> K;  // G^T M G, constrained
	mfem::Array<int> fixed;                 // H1 DOFs held at psi = 0
	std::set<int> essential;                // H1 DOFs on the essential boundary
	std::unique_ptr<AmgPreconditioner> amg;
};
