// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "mfem.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Process-level MPI/HYPRE setup, shared by the executable and tests.
 *
 * Everything here compiles in both builds. In a serial build (USE_MPI=OFF)
 * the functions are no-ops that report a single process, so callers need no
 * #ifdefs of their own.
 *
 * Scope today: an MPI build gives the solvers access to HYPRE (BoomerAMG,
 * AMS) on a ParMesh, but runs on ONE rank. The solvers, the coupling-matrix
 * assembly and the result writers are not yet distributed, so launching more
 * ranks would only repeat the same run; the executable rejects it.
 */
namespace parallel {

/// Initialize MPI and HYPRE (MFEM finalizes both at program exit), and, in
/// an OpenMP build, MFEM's "omp" device, which threads its sparse
/// matrix-vector products and vector operations over OMP_NUM_THREADS. It is
/// configured once, before any MFEM object, and lives for the process.
inline void Initialize(int& argc, char**& argv) {
#ifdef MFEM_USE_MPI
	mfem::Mpi::Init(argc, argv);
	mfem::Hypre::Init();
#else
	(void)argc;
	(void)argv;
#endif
#ifdef MFEM_USE_OPENMP
	static mfem::Device device("omp");
#endif
}

/// The number of threads the linear algebra runs on.
inline int Threads() {
#ifdef _OPENMP
	return omp_get_max_threads();
#else
	return 1;
#endif
}

/// Whether this is an MPI build.
constexpr bool Enabled() {
#ifdef MFEM_USE_MPI
	return true;
#else
	return false;
#endif
}

inline int WorldSize() {
#ifdef MFEM_USE_MPI
	return mfem::Mpi::WorldSize();
#else
	return 1;
#endif
}

inline bool IsRoot() {
#ifdef MFEM_USE_MPI
	return mfem::Mpi::Root();
#else
	return true;
#endif
}

#ifdef MFEM_USE_MPI
/// A ParMesh holding all of @p mesh on the single rank of MPI_COMM_WORLD.
///
/// The partitioning is given explicitly (every element on rank 0) because
/// MFEM otherwise calls its METIS partitioner even for one rank, and METIS is
/// optional in this build.
inline std::unique_ptr<mfem::ParMesh> MakeSingleRankParMesh(mfem::Mesh& mesh) {
	MFEM_VERIFY(mfem::Mpi::WorldSize() == 1,
		"Only single-rank MPI runs are supported so far.");
	mfem::Array<int> partitioning(mesh.GetNE());
	partitioning = 0;
	return std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh, partitioning.GetData());
}
#endif

} // namespace parallel
