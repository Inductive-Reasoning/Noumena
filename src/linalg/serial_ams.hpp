// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include "mfem.hpp"
#include "../parallel/mpi_runtime.hpp"

#ifdef MFEM_USE_MPI

#include <memory>

/**
 * @brief hypre's AMS (auxiliary-space Maxwell solver) preconditioner for a
 *        curl-curl system assembled on a SERIAL Nedelec space.
 *
 * mfem::HypreAMS builds its auxiliary operators from a ParFiniteElementSpace,
 * whose DOF numbering on a ParMesh differs from the serial space every other
 * part of this code (conductor loads, the divergence-free projector, the writers)
 * is built on. This class builds the same operators from the serial space
 * instead and hands them to hypre wrapped as single-rank HypreParMatrix
 * objects, mirroring HypreAMS::MakeSolver / MakeGradientAndInterpolation:
 *   - G, the discrete gradient H1_p -> ND_p;
 *   - Pi_x, Pi_y, Pi_z, the components of the interpolation (H1_p)^3 -> ND_p
 *     (used for every order, and for curved meshes, where hypre's vertex-
 *     coordinate shortcut would not apply);
 *   - cycle type 13 and MFEM's BoomerAMG options for the auxiliary problems.
 *
 * With SetSingularProblem() the beta (gradient) correction is dropped, which
 * is the right configuration for magnetostatics: curl(nu curl A) = J with no
 * mass term, solved by CG on the singular but consistent system.
 *
 * Single rank only (the partitioning arrays describe one process owning
 * everything), which is all the MPI build supports so far.
 */
class SerialAmsPreconditioner : public mfem::HypreSolver {
public:
	/// @param K   Constrained system matrix on @p nd (finalized; essential
	///            rows/columns eliminated). Referenced, not copied: it must
	///            outlive this object. Its column order within a row may be
	///            permuted (diagonal first), which does not change the matrix.
	/// @param nd  The Nedelec space K was assembled on.
	SerialAmsPreconditioner(mfem::SparseMatrix& K, mfem::FiniteElementSpace& nd,
							bool singular)
		: h1_fec(nd.GetMaxElementOrder(), nd.GetMesh()->Dimension()),
		  h1(nd.GetMesh(), &h1_fec),
		  h1_vector(nd.GetMesh(), &h1_fec, 3, mfem::Ordering::byNODES) {
		MFEM_VERIFY(mfem::Mpi::WorldSize() == 1,
			"SerialAmsPreconditioner supports a single MPI rank only.");
		MFEM_VERIFY(nd.GetMesh()->Dimension() == 3 && nd.GetMesh()->SpaceDimension() == 3,
			"SerialAmsPreconditioner is for 3D H(curl) problems.");

		nd_starts[0] = 0;
		nd_starts[1] = nd.GetVSize();
		h1_starts[0] = 0;
		h1_starts[1] = h1.GetVSize();

		K_hypre = std::make_unique<mfem::HypreParMatrix>(
			MPI_COMM_WORLD, nd_starts[1], nd_starts, &K);
		// HypreSolver's operator is its protected A (there is no setter), and
		// the base was default-constructed because K_hypre did not exist yet.
		A = K_hypre.get();
		height = width = K_hypre->Height();

		// Discrete gradient.
		{
			mfem::DiscreteLinearOperator grad(&h1, &nd);
			grad.AddDomainInterpolator(new mfem::GradientInterpolator);
			grad.Assemble();
			grad.Finalize();
			G.reset(grad.LoseMat());
		}
		G_hypre = WrapRectangular(*G, h1_starts);

		// Nedelec interpolation, split by component.
		{
			mfem::DiscreteLinearOperator identity(&h1_vector, &nd);
			identity.AddDomainInterpolator(new mfem::IdentityInterpolator);
			identity.Assemble();
			identity.Finalize();
			mfem::Array2D<mfem::SparseMatrix*> blocks;
			identity.GetBlocks(blocks);
			for (int c = 0; c < 3; ++c) {
				Pi[c].reset(blocks(0, c));
				Pi_hypre[c] = WrapRectangular(*Pi[c], h1_starts);
			}
		}

		MakeSolver(singular);
	}

	~SerialAmsPreconditioner() override { HYPRE_AMSDestroy(ams); }

	operator HYPRE_Solver() const override { return ams; }
	HYPRE_PtrToParSolverFcn SetupFcn() const override {
		return (HYPRE_PtrToParSolverFcn) HYPRE_AMSSetup;
	}
	HYPRE_PtrToParSolverFcn SolveFcn() const override {
		return (HYPRE_PtrToParSolverFcn) HYPRE_AMSSolve;
	}

private:
	mfem::H1_FECollection h1_fec;
	mfem::FiniteElementSpace h1;
	mfem::FiniteElementSpace h1_vector;

	// Partitionings: one rank owns all rows. hypre may keep pointers to these.
	HYPRE_BigInt nd_starts[2];
	HYPRE_BigInt h1_starts[2];

	std::unique_ptr<mfem::HypreParMatrix> K_hypre;
	std::unique_ptr<mfem::SparseMatrix> G;
	std::unique_ptr<mfem::HypreParMatrix> G_hypre;
	std::unique_ptr<mfem::SparseMatrix> Pi[3];
	std::unique_ptr<mfem::HypreParMatrix> Pi_hypre[3];

	HYPRE_Solver ams = nullptr;

	std::unique_ptr<mfem::HypreParMatrix> WrapRectangular(mfem::SparseMatrix& m,
														  HYPRE_BigInt* col_starts) {
		return std::make_unique<mfem::HypreParMatrix>(
			MPI_COMM_WORLD, nd_starts[1], col_starts[1], nd_starts, col_starts, &m);
	}

	// The configuration of mfem::HypreAMS::MakeSolver (CPU branch), except
	// the smoothers when threaded. MFEM's (hybrid l1 Gauss-Seidel in AMS,
	// hybrid l1-SSOR in its subspace AMG solves) are the stronger on one
	// thread, but threaded they decouple into per-thread blocks and lose
	// strength; Chebyshev keeps its iteration count at any thread count. On
	// TEAM 7 (0.9M complex unknowns, order 2) the solve took 428 s with
	// Gauss-Seidel on one thread, 410 s with it on four (97 GMRES iterations
	// instead of 84 at order 1), and 168 s with Chebyshev on four, which is
	// 574 s on one.
	void MakeSolver(bool singular) {
		const bool threaded = parallel::Threads() > 1;
		const int rlx_type = threaded ? 16 : 2, rlx_sweeps = 1;
		const mfem::real_t rlx_weight = 1.0, rlx_omega = 1.0;
		const int amg_coarsen_type = 10, amg_agg_levels = 1, amg_rlx_type = threaded ? 16 : 8;
		const mfem::real_t theta = 0.25;
		const int amg_interp_type = 6, amg_Pmax = 4;

		HYPRE_AMSCreate(&ams);
		HYPRE_AMSSetDimension(ams, 3);
		HYPRE_AMSSetTol(ams, 0.0);
		HYPRE_AMSSetMaxIter(ams, 1);  // one cycle: a preconditioner
		HYPRE_AMSSetCycleType(ams, 13);
		HYPRE_AMSSetPrintLevel(ams, 0);
		HYPRE_AMSSetSmoothingOptions(ams, rlx_type, rlx_sweeps, rlx_weight, rlx_omega);
		HYPRE_AMSSetAlphaAMGOptions(ams, amg_coarsen_type, amg_agg_levels, amg_rlx_type,
									theta, amg_interp_type, amg_Pmax);
		HYPRE_AMSSetBetaAMGOptions(ams, amg_coarsen_type, amg_agg_levels, amg_rlx_type,
								   theta, amg_interp_type, amg_Pmax);
		HYPRE_AMSSetAlphaAMGCoarseRelaxType(ams, amg_rlx_type);
		HYPRE_AMSSetBetaAMGCoarseRelaxType(ams, amg_rlx_type);

		HYPRE_AMSSetDiscreteGradient(ams, *G_hypre);
		HYPRE_AMSSetInterpolations(ams, nullptr, *Pi_hypre[0], *Pi_hypre[1], *Pi_hypre[2]);
		if (singular) { HYPRE_AMSSetBetaPoissonMatrix(ams, nullptr); }

		// As in HypreAMS: setup may meet singular auxiliary matrices, which
		// hypre's solve handles correctly but its setup reports as errors.
		error_mode = IGNORE_HYPRE_ERRORS;
	}
};

#endif // MFEM_USE_MPI
