// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

// Catch2 session with MPI/HYPRE initialized around it in an MPI build (and
// nothing extra in a serial one).
#include <catch2/catch_session.hpp>

#include "parallel/mpi_runtime.hpp"

int main(int argc, char* argv[]) {
	parallel::Initialize(argc, argv);
	return Catch::Session().run(argc, argv);
}
