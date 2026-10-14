// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// Meshes generated with Gmsh at test time, for geometries with curved
// boundaries (circles, spheres) whose closed-form solutions the tests check.
// The meshes are not committed; a test that needs one skips without Gmsh.

#pragma once

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>

#include "io/mesh_loader.hpp"
#include "mfem.hpp"

namespace gmsh_fixture {

/// Meshes the Gmsh script @p geo in @p dimension dimensions with elements of
/// geometric order @p order (curved, nodes on the true boundaries), and
/// loads it. The script and mesh go to a scratch directory of this process,
/// named @p name; Gmsh's log is kept there only if it fails.
inline std::unique_ptr<mfem::Mesh> Mesh(const std::string& name, const std::string& geo,
										int dimension, int order) {
#ifndef NOUMENA_GMSH
	(void)name; (void)geo; (void)dimension; (void)order;
	SKIP("This test generates its mesh with Gmsh, which CMake did not find.");
	return nullptr;
#else
	namespace fs = std::filesystem;
	// Per process, so tests run in parallel do not share files.
	static const fs::path work = fs::temp_directory_path() /
		("noumena_gmsh_" + std::to_string(std::random_device{}()));
	fs::create_directories(work);
	const fs::path script = work / (name + ".geo"), mesh = work / (name + ".msh"),
				   log = work / (name + ".log");
	std::ofstream(script) << geo;
	const std::string command = std::string("\"") + NOUMENA_GMSH + "\" -" +
		std::to_string(dimension) + " -order " + std::to_string(order) +
		" -format msh2 \"" + script.string() + "\" -o \"" + mesh.string() + "\" > \"" +
		log.string() + "\" 2>&1";
	INFO("Gmsh log: " << log.string());
	REQUIRE(std::system(command.c_str()) == 0);
	auto loaded = mesh_io::LoadMesh(mesh.string());
	for (const fs::path& file : { script, mesh, log }) { fs::remove(file); }
	fs::remove(work);  // now empty
	return loaded;
#endif
}

} // namespace gmsh_fixture
