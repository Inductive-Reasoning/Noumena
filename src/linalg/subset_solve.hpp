// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "mfem.hpp"
#include "amg_preconditioner.hpp"

/**
 * @brief Solve the SPD system K x = rhs for the unknowns marked free, the
 *        others keeping their values in @p x.
 *
 * Only the free unknowns enter the solve, by extracting their submatrix and
 * lifting the fixed values into its right-hand side. That matters when the
 * free set is a small part of a large space -- one conductor's DOFs in a mesh
 * of air -- because pinning everything else as identity rows would leave
 * those rows unconnected, which smoothed-aggregation AMG turns into empty
 * aggregates.
 *
 * @param free     Unknowns to solve for (indices into K, x and rhs).
 * @param x        In: the fixed values (and anything at the free entries);
 *                 out: the free entries solved.
 * @param what     Names the solve in the non-convergence error.
 */
inline void SolveOnSubset(const mfem::SparseMatrix& K, const mfem::Vector& rhs,
						  const std::vector<int>& free, mfem::Vector& x,
						  double tolerance, const std::string& what) {
	std::vector<int> index(K.Height(), -1);
	for (size_t r = 0; r < free.size(); ++r) { index[free[r]] = static_cast<int>(r); }
	const int n = static_cast<int>(free.size());
	if (n == 0) return;

	mfem::SparseMatrix Kff(n, n);
	mfem::Vector b(n), y(n);
	for (int r = 0; r < n; ++r) {
		const int row = free[r];
		b(r) = rhs(row);
		for (int p = K.GetI()[row]; p < K.GetI()[row + 1]; ++p) {
			const int col = K.GetJ()[p];
			const double a = K.GetData()[p];
			if (index[col] >= 0) { Kff.Add(r, index[col], a); }
			else { b(r) -= a * x(col); }  // lift the fixed values
		}
	}
	Kff.Finalize();
	AmgPreconditioner amg(Kff);
	mfem::CGSolver cg;
	cg.SetOperator(Kff);
	cg.SetPreconditioner(amg);
	cg.SetRelTol(tolerance);
	cg.SetAbsTol(0.0);
	cg.SetMaxIter(2000);
	cg.SetPrintLevel(0);
	y = 0.0;
	cg.Mult(b, y);
	MFEM_VERIFY(cg.GetConverged() || b.Norml2() == 0.0, what + " did not converge.");
	for (int r = 0; r < n; ++r) { x(free[r]) = y(r); }
}
