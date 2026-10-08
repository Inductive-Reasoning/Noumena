// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once
#include <algorithm>
#include <filesystem>
#include <list>
#include <string>
#include <vector>
#include "mfem.hpp"
#include "field_export.hpp"
#include "gmsh_results_writer.hpp"
#include "status_reporter.hpp"

/**
 * @brief Serializes a FieldExportSet to the supported result formats.
 *
 * A dumb sink: it computes nothing and invents no fields. Solvers decide WHAT
 * to export by building a FieldExportSet; this class only decides HOW to write
 * it. Keeping it out of the solver hierarchy means the serialization details
 * (element orders, projection spaces, file naming) are not mixed in with the
 * physics.
 *
 * @warning The mesh reference must outlive this writer instance.
 */
class SolverFieldWriter {
public:
	// @p solution_order is the order of the H1 space the primary fields live in;
	// it drives both the ParaView Lagrange cell order and the order of the
	// native Gmsh Lagrange elements emitted by WriteGmsh.
	SolverFieldWriter(mfem::Mesh& mesh, int solution_order)
		: mesh(mesh),
		  solution_order(solution_order) {}

	// ParaView serializer. Primary scalars are registered at native (high) order
	// so ParaView's Lagrange cells render them faithfully; derived coefficients
	// are projected into an L2 space one order below the H1 solution.
	void WriteParaview(const std::filesystem::path& directory,
					   const std::string& collection_name,
					   const FieldExportSet& fields) const;

	// Gmsh serializer. Emits the mesh as native Gmsh Lagrange elements of the
	// solution order, with each field sampled at that element's own node
	// lattice, so Gmsh reconstructs the field with matching high-order shape
	// functions instead of a tessellated linear approximation.
	void WriteGmsh(const std::filesystem::path& out_path,
				   const FieldExportSet& fields,
				   gmsh_results::MshVersion gmsh_version) const;

private:
	mfem::Mesh& mesh;
	int solution_order;
	// The mesh-side part of the Gmsh files, shared by every scenario on a mesh.
	mutable gmsh_results::MeshExport gmsh_mesh;

	StatusReporter& Reporter() const {
		return StatusReporter::Global();
	}
};
