// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>

#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/chebyshev.hpp>
#include <amgcl/util.hpp>

#include "mfem.hpp"

/**
 * @brief Algebraic multigrid preconditioner (AMGCL) for the SPD systems of the
 *        static solvers, applied inside CG.
 *
 * Why multigrid: a one-level preconditioner (Gauss-Seidel, incomplete
 * Cholesky) only damps the highest-frequency error, so the CG iteration count
 * grows with mesh refinement; in 3D that growth, not the per-iteration cost,
 * dominates. One multigrid V-cycle reduces error at every scale, so the
 * iteration count grows slowly, if at all, as the mesh is refined. Measured on
 * a 3D Laplacian in the unit cube (CG to 1e-10): order-3 tetrahedra, 13-14
 * iterations from 16k to 389k unknowns; order-2 hexahedra, 23-63 from 36k to
 * 913k. Gauss-Seidel PCG took 92-201 iterations on a P2 problem of 36k-531k
 * and was 6x slower at the top, even including the AMG setup.
 *
 * Why AMGCL: header-only, MIT-licensed, and threaded with OpenMP rather than
 * MPI (HYPRE's BoomerAMG requires an MPI build of MFEM).
 *
 * Configuration: smoothed-aggregation coarsening with Chebyshev relaxation
 * (degree 5), for scalar elliptic operators such as the (axisymmetric or
 * Cartesian) diffusion operators assembled here.
 *
 * Why Chebyshev: a pointwise smoother with a fixed weight -- SPAI(0), damped
 * Jacobi -- is stable only while the weight is below 2 / rho(D^-1 A).
 * Higher-order Lagrange stiffness matrices push rho(D^-1 A) past that: at
 * order 3 on tetrahedra (even a uniform cube) the V-cycle stopped being
 * positive definite and CG broke down within four iterations, under every
 * coarsening tried. Chebyshev sizes itself from an estimate of the spectrum,
 * so it stays stable at any order. Measured: the same iteration count as
 * SPAI(0) at order 2 (or fewer), at 1.0-1.6x the time per solve. It is NOT suitable for the
 * 3D H(curl) curl-curl operator, whose gradient null space defeats nodal AMG;
 * that needs an auxiliary-space (AMS-type) preconditioner instead.
 *
 * The hierarchy is built once from @p A at construction (AMGCL copies what it
 * needs, so @p A need not outlive this object) and reused for every right-hand
 * side, which is what lets a coupling-matrix run amortize the setup.
 */
class AmgPreconditioner : public mfem::Solver {
	using Backend = amgcl::backend::builtin<double>;
	using Amg = amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation,
						   amgcl::relaxation::chebyshev>;

public:
	explicit AmgPreconditioner(const mfem::SparseMatrix& A)
		: mfem::Solver(A.Height(), A.Width()) {
		MFEM_VERIFY(A.Height() == A.Width(),
			"AmgPreconditioner requires a square matrix.");
		MFEM_VERIFY(A.Finalized(),
			"AmgPreconditioner requires a finalized (CSR) matrix.");

		// MFEM's CSR arrays are passed as ranges; AMGCL copies them into its own
		// backend format while building the hierarchy.
		const int n = A.Height();
		const int nnz = A.NumNonZeroElems();
		auto matrix = std::make_tuple(n,
			amgcl::make_iterator_range(A.GetI(), A.GetI() + n + 1),
			amgcl::make_iterator_range(A.GetJ(), A.GetJ() + nnz),
			amgcl::make_iterator_range(A.GetData(), A.GetData() + nnz));
		amg = std::make_unique<Amg>(matrix);

		rhs.resize(n);
		sol.resize(n);
	}

	// One V-cycle from a zero initial guess: x = M^{-1} b.
	void Mult(const mfem::Vector& b, mfem::Vector& x) const override {
		MFEM_ASSERT(b.Size() == Height() && x.Size() == Width(),
			"AmgPreconditioner::Mult size mismatch.");
		std::copy(b.GetData(), b.GetData() + b.Size(), rhs.begin());
		std::fill(sol.begin(), sol.end(), 0.0);
		amg->apply(rhs, sol);
		std::copy(sol.begin(), sol.end(), x.GetData());
	}

	// The hierarchy is fixed at construction, as for SparseDirectSolver.
	void SetOperator(const mfem::Operator&) override {
		MFEM_ABORT("AmgPreconditioner's operator is set at construction.");
	}

private:
	std::unique_ptr<Amg> amg;
	// AMGCL's builtin backend works on its own vector type (std::vector is
	// accepted); scratch storage is kept here so Mult() does not allocate.
	mutable std::vector<double> rhs;
	mutable std::vector<double> sol;
};
