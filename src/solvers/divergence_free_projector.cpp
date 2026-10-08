// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include "divergence_free_projector.hpp"

DivergenceFreeProjector::DivergenceFreeProjector(mfem::FiniteElementSpace& nd, const mfem::Array<int>& ess_bdr)
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

	mfem::Array<int> marker(ess_bdr);
	mfem::Array<int> ess_dofs;
	h1.GetEssentialTrueDofs(marker, ess_dofs);
	for (int d : ess_dofs) { essential.insert(d); }
	wall_piece = WallPieces(marker, wall_pieces);

	// The reduced space: one column per DOF off the essential boundary
	// (all but the first if there is none) and one per wall piece but the
	// first.
	const int n_h1 = h1.GetVSize();
	std::vector<int> column(n_h1, -1);
	int columns = 0;
	for (int d = 0; d < n_h1; ++d) {
		if (essential.count(d) == 0 && (!essential.empty() || d > 0)) { column[d] = columns++; }
	}
	std::vector<int> piece_column(wall_pieces, -1);
	for (int p = 1; p < wall_pieces; ++p) { piece_column[p] = columns++; }
	for (int d : essential) { column[d] = piece_column[wall_piece[d]]; }
	P = Columns(column, columns);

	std::unique_ptr<mfem::SparseMatrix> GtMG(mfem::RAP(*G, *M, *G));
	K.reset(mfem::RAP(*P, *GtMG, *P));  // P^T G^T M G P
	amg = std::make_unique<AmgPreconditioner>(*K);
}

double DivergenceFreeProjector::Project(mfem::Vector& b) const {
	mfem::Vector rhs(G->Width());
	G->MultTranspose(b, rhs);
	const mfem::Vector psi = SolvePotential(rhs);

	mfem::Vector grad_psi(G->Height()), correction(M->Height());
	G->Mult(psi, grad_psi);
	M->Mult(grad_psi, correction);
	b -= correction;
	return psi * rhs;
}

double DivergenceFreeProjector::ProjectWithin(mfem::Vector& b, const mfem::Array<int>& conductor) const {
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

void DivergenceFreeProjector::RemoveGradient(mfem::Vector& A) const {
	mfem::Vector MA(M->Height()), rhs(G->Width());
	M->Mult(A, MA);
	G->MultTranspose(MA, rhs);
	const mfem::Vector psi = SolvePotential(rhs);

	mfem::Vector grad_psi(G->Height());
	G->Mult(psi, grad_psi);
	A -= grad_psi;
}

std::unique_ptr<mfem::SparseMatrix> DivergenceFreeProjector::GaugeConstraint(const mfem::Array<int>& conducting) const {
	const DofParts parts = ConnectedParts(conducting);
	const int n_h1 = h1.GetVSize();

	// Wall pieces joined through a grounded conductor share one value.
	std::vector<int> group(wall_pieces);
	for (int p = 0; p < wall_pieces; ++p) { group[p] = p; }
	const auto find = [&](int p) {
		while (group[p] != p) { p = group[p] = group[group[p]]; }
		return p;
	};
	std::vector<int> part_piece(parts.grounded.size(), -1);
	for (int d : essential) {
		const int c = parts.part[d];
		if (c < 0) continue;
		const int p = find(wall_piece[d]);
		if (part_piece[c] < 0) { part_piece[c] = p; }
		else { group[find(part_piece[c])] = p; }
	}

	std::vector<int> column(n_h1, -1), part_column(parts.grounded.size(), -1);
	std::vector<int> group_column(wall_pieces, -1);
	int multipliers = 0;
	const auto wall_column = [&](int piece) {
		const int g = find(piece);
		if (g == find(0)) { return -1; }  // the reference value, zero
		if (group_column[g] < 0) { group_column[g] = multipliers++; }
		return group_column[g];
	};
	for (int d = 0; d < n_h1; ++d) {
		const int c = parts.part[d];
		if (essential.count(d)) { column[d] = wall_column(wall_piece[d]); continue; }
		if (c < 0) { column[d] = multipliers++; continue; }
		if (parts.grounded[c]) { column[d] = wall_column(part_piece[c]); continue; }
		if (part_column[c] < 0) { part_column[c] = multipliers++; }
		column[d] = part_column[c];
	}
	if (essential.empty() && multipliers > 0) {
		for (int& j : column) { j = j == 0 ? -1 : (j > 0 ? j - 1 : j); }
		--multipliers;
	}

	const std::unique_ptr<mfem::SparseMatrix> P_gauge = Columns(column, multipliers);
	std::unique_ptr<mfem::SparseMatrix> MG(mfem::Mult(*M, *G));
	return std::unique_ptr<mfem::SparseMatrix>(mfem::Mult(*MG, *P_gauge));
}

double DivergenceFreeProjector::GradientResidual(const mfem::Vector& b) const {
	mfem::Vector gtb(G->Width()), r(P->Width());
	G->MultTranspose(b, gtb);
	P->MultTranspose(gtb, r);
	return r.Norml2();
}

DivergenceFreeProjector::DofParts DivergenceFreeProjector::ConnectedParts(const mfem::Array<int>& marker) const {
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

mfem::Vector DivergenceFreeProjector::SolvePotential(const mfem::Vector& rhs) const {
	mfem::Vector reduced(P->Width()), y(P->Width()), psi(G->Width());
	P->MultTranspose(rhs, reduced);
	y = 0.0;
	mfem::CGSolver cg;
	cg.SetOperator(*K);
	cg.SetPreconditioner(*amg);
	cg.SetRelTol(kTolerance);
	cg.SetAbsTol(0.0);
	cg.SetMaxIter(1000);
	cg.SetPrintLevel(0);
	cg.Mult(reduced, y);
	MFEM_VERIFY(cg.GetConverged() || reduced.Norml2() == 0.0,
		"Divergence-free projection did not converge.");
	P->Mult(y, psi);
	return psi;
}

std::vector<int> DivergenceFreeProjector::WallPieces(const mfem::Array<int>& marker, int& count) const {
	const mfem::Mesh& mesh = *nd.GetMesh();
	std::vector<int> parent(h1.GetVSize());
	for (int d = 0; d < h1.GetVSize(); ++d) { parent[d] = d; }
	const auto find = [&](int d) {
		while (parent[d] != d) { d = parent[d] = parent[parent[d]]; }
		return d;
	};
	mfem::Array<int> dofs;
	for (int be = 0; be < mesh.GetNBE(); ++be) {
		const int a = mesh.GetBdrAttribute(be);
		if (a < 1 || a > marker.Size() || !marker[a - 1]) continue;
		h1.GetBdrElementDofs(be, dofs);
		for (int d : dofs) { parent[find(d)] = find(dofs[0]); }
	}
	std::vector<int> piece(h1.GetVSize(), -1), root_piece(h1.GetVSize(), -1);
	count = 0;
	for (int d : essential) {
		const int r = find(d);
		if (root_piece[r] < 0) { root_piece[r] = count++; }
		piece[d] = root_piece[r];
	}
	return piece;
}

std::unique_ptr<mfem::SparseMatrix> DivergenceFreeProjector::Columns(const std::vector<int>& column, int columns) const {
	auto matrix = std::make_unique<mfem::SparseMatrix>(static_cast<int>(column.size()), columns);
	for (size_t d = 0; d < column.size(); ++d) {
		if (column[d] >= 0) { matrix->Set(static_cast<int>(d), column[d], 1.0); }
	}
	matrix->Finalize();
	return matrix;
}
