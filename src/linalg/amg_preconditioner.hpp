// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <ostream>
#include <tuple>
#include <vector>

#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
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
 * iteration count stays roughly constant as the mesh is refined. Measured on a
 * 3D P2 Laplacian: 18-22 iterations from 36k to 531k unknowns, against 92-201
 * for Gauss-Seidel PCG, and 6x faster at 531k even including the AMG setup.
 *
 * Why AMGCL: header-only, MIT-licensed, and threaded with OpenMP rather than
 * MPI (HYPRE's BoomerAMG requires an MPI build of MFEM).
 *
 * Configuration: smoothed-aggregation coarsening with SPAI(0) relaxation, the
 * robust default for scalar elliptic operators such as the (axisymmetric or
 * Cartesian) diffusion operators assembled here. It is NOT suitable for the
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
						   amgcl::relaxation::spai0>;

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

	/// Memory held by the hierarchy, in bytes (diagnostics).
	[[nodiscard]] std::size_t Bytes() const { return amg->bytes(); }

	/// AMGCL's own per-level summary (rows, nonzeros, operator complexity).
	friend std::ostream& operator<<(std::ostream& os, const AmgPreconditioner& p) {
		return os << *p.amg;
	}

private:
	std::unique_ptr<Amg> amg;
	// AMGCL's builtin backend works on its own vector type (std::vector is
	// accepted); scratch storage is kept here so Mult() does not allocate.
	mutable std::vector<double> rhs;
	mutable std::vector<double> sol;
};
