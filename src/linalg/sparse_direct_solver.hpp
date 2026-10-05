// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include "mfem.hpp"

// Sparse direct solver (supernodal-free simplicial LDL^T) for the symmetric
// positive-definite systems the static solvers assemble.
//
// Why this exists alongside PCG: a CouplingMatrix run solves N right-hand sides
// against ONE matrix (one unit excitation per terminal). A direct solver factors
// that matrix once and back-substitutes per RHS, so the per-terminal marginal
// cost is a pair of triangular sweeps instead of a full Krylov solve. It also
// removes the preconditioner's convergence behaviour from the results: the
// factorization is exact up to round-off, so weak off-diagonal coupling terms do
// not depend on a residual tolerance.
//
// Eigen (MPL2 subset) is used because it is header-only: no SuiteSparse/MPI
// dependency, and no GPL-licensed component, unlike UMFPACK/CHOLMOD.
class SparseDirectSolver : public mfem::Solver {
public:
	// Factor `A`. The matrix must be finalized (CSR) and SPD.
	explicit SparseDirectSolver(mfem::SparseMatrix& A)
		: mfem::Solver(A.Height(), A.Width()) {
		MFEM_VERIFY(A.Height() == A.Width(),
			"SparseDirectSolver requires a square matrix.");
		MFEM_VERIFY(A.Finalized(),
			"SparseDirectSolver requires a finalized (CSR) matrix.");

		// Eigen requires each row's column indices to be ascending; MFEM makes no
		// such guarantee for an assembled matrix. Sorting in place leaves A
		// semantically unchanged (CSR is order-independent) and is a no-op on
		// subsequent calls.
		A.SortColumnIndices();

		// Map MFEM's CSR arrays onto Eigen without copying. MFEM stores rows
		// contiguously with sorted column indices per row, which is exactly Eigen's
		// RowMajor sparse layout, so the map is a reinterpretation rather than a
		// conversion. The Map is a view over A's storage and must not outlive A;
		// assigning it into `matrix` both copies and transposes into the ColMajor
		// layout SimplicialLDLT factors in.
		const Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor>> view(
			A.Height(), A.Width(), A.NumNonZeroElems(),
			A.GetI(), A.GetJ(), A.GetData());
		matrix = view;

		// SimplicialLDLT reads only the selected triangle, so a structurally
		// asymmetric stored matrix would be silently accepted; the symmetry of the
		// FE operator is the caller's responsibility.
		solver.compute(matrix);
		MFEM_VERIFY(solver.info() == Eigen::Success,
			"Sparse direct factorization failed: the system matrix is singular "
			"or not positive definite. A common cause is a domain region with "
			"no material assigned, which contributes zero stiffness and leaves "
			"its interior DOFs unconstrained.");
	}

	// Back-substitute a single right-hand side against the stored factors.
	void Mult(const mfem::Vector& b, mfem::Vector& x) const override {
		MFEM_ASSERT(b.Size() == matrix.rows() && x.Size() == matrix.cols(),
			"SparseDirectSolver::Mult size mismatch.");
		const Eigen::Map<const Eigen::VectorXd> rhs(b.GetData(), b.Size());
		Eigen::Map<Eigen::VectorXd> sol(x.GetData(), x.Size());
		sol = solver.solve(rhs);
		MFEM_VERIFY(solver.info() == Eigen::Success,
			"Sparse direct back-substitution failed.");
	}

	// The operator is fixed at construction: this solver owns its factorization
	// and there is no meaningful way to retarget it without refactoring.
	void SetOperator(const mfem::Operator&) override {
		MFEM_ABORT("SparseDirectSolver's operator is set at construction.");
	}

private:
	using SpMat = Eigen::SparseMatrix<double>;  // ColMajor
	SpMat matrix;
	Eigen::SimplicialLDLT<SpMat, Eigen::Lower> solver;
};

// Sparse direct solver (LU with column-pivoting-free COLAMD ordering) for
// general square systems.
//
// Why this exists alongside SparseDirectSolver: the time-harmonic MQS system is
// the packed real form [R, -I; I, R] of a complex operator, bordered by the
// massive-port coupling. That packed matrix is not symmetric even where R and I
// are (its off-diagonal blocks are -I and +I), nor definite, so a Cholesky-type
// factorization does not apply. LU assumes neither.
//
// The reuse argument is the same as for the SPD case: a coupling sweep solves
// one right-hand side per terminal against the matrix for a single frequency,
// so factoring once per frequency turns each terminal into a pair of triangular
// sweeps.
class SparseLUSolver : public mfem::Solver {
public:
	// Factor `A`. The matrix must be finalized (CSR) and square.
	explicit SparseLUSolver(mfem::SparseMatrix& A)
		: mfem::Solver(A.Height(), A.Width()) {
		MFEM_VERIFY(A.Height() == A.Width(),
			"SparseLUSolver requires a square matrix.");
		MFEM_VERIFY(A.Finalized(),
			"SparseLUSolver requires a finalized (CSR) matrix.");

		// See SparseDirectSolver for why the indices are sorted and why the
		// RowMajor map into a ColMajor matrix is the transpose-copy we want.
		A.SortColumnIndices();
		const Eigen::Map<const Eigen::SparseMatrix<double, Eigen::RowMajor>> view(
			A.Height(), A.Width(), A.NumNonZeroElems(),
			A.GetI(), A.GetJ(), A.GetData());
		matrix = view;

		solver.analyzePattern(matrix);
		solver.factorize(matrix);
		MFEM_VERIFY(solver.info() == Eigen::Success,
			"Sparse LU factorization failed: the system matrix is singular. "
			"A common cause is a region left without material properties, or a "
			"massive port whose conductance is zero.");
	}

	// Back-substitute a single right-hand side against the stored factors.
	void Mult(const mfem::Vector& b, mfem::Vector& x) const override {
		MFEM_ASSERT(b.Size() == matrix.rows() && x.Size() == matrix.cols(),
			"SparseLUSolver::Mult size mismatch.");
		const Eigen::Map<const Eigen::VectorXd> rhs(b.GetData(), b.Size());
		Eigen::Map<Eigen::VectorXd> sol(x.GetData(), x.Size());
		sol = solver.solve(rhs);
		MFEM_VERIFY(solver.info() == Eigen::Success,
			"Sparse LU back-substitution failed.");
	}

	// The operator is fixed at construction, as for SparseDirectSolver.
	void SetOperator(const mfem::Operator&) override {
		MFEM_ABORT("SparseLUSolver's operator is set at construction.");
	}

private:
	using SpMat = Eigen::SparseMatrix<double>;  // ColMajor
	SpMat matrix;
	Eigen::SparseLU<SpMat, Eigen::COLAMDOrdering<int>> solver;
};
