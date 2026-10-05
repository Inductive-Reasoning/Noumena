// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include "mfem.hpp"
#include "sparse_lu.hpp"

// Direct solver of a curl-curl system K A = b, K singular on the discrete
// gradients, gauged by a Lagrange multiplier: it factors
//
//     [ K    G ]
//     [ G^T  0 ]
//
// with G the gauge constraint (DivergenceFreeProjector::GaugeConstraint), so
// that A satisfies the discrete Coulomb gauge exactly. For a balanced load
// (G^T b = 0, which the projected loads are) the multiplier comes out zero
// and K A = b holds exactly. The matrix is symmetric indefinite; SparseLU
// pivots past its zero block.
class GaugedDirectSolver : public mfem::Solver {
public:
	/// @param K          The system matrix with the essential rows and
	///                   columns @p ess_tdofs eliminated (unit diagonal).
	/// @param gauge      The gauge constraint, every row.
	GaugedDirectSolver(const mfem::SparseMatrix& K, const mfem::SparseMatrix& gauge,
					   const mfem::Array<int>& ess_tdofs)
		: mfem::Solver(K.Height()), constraint(gauge, ess_tdofs) {
		const CsrMatrix<double> csr = ToCsr(K);
		diagonal = Diagonal(csr);
		lu.Factor(BorderWithConstraint(csr, constraint.Interior()));
	}

	void Mult(const mfem::Vector& b, mfem::Vector& x) const override {
		MFEM_ASSERT(b.Size() == Height() && x.Size() == Height(),
			"GaugedDirectSolver: vector size does not match the factored matrix.");
		const int n = Height();
		std::vector<double> rhs(lu.Size()), sol(lu.Size());
		for (int i = 0; i < n; ++i) { rhs[i] = b(i); }
		constraint.RightHandSide(rhs.data(), diagonal.data(), rhs.data() + n);
		lu.Solve(rhs.data(), sol.data(), 1);
		for (int i = 0; i < Height(); ++i) { x(i) = sol[i]; }
	}

	void SetOperator(const mfem::Operator&) override {
		MFEM_ABORT("GaugedDirectSolver is bound to the matrix it factored.");
	}

private:
	GaugeConstraintRows constraint;
	std::vector<double> diagonal;  // of K, for the essential values
	SparseLU<double> lu;
};
