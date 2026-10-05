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
					   const FieldExportSet& fields) const
	{
		// ParaViewDataCollection and the L2 projections below are dimension-
		// generic, so 2D and 3D meshes are both supported.
		const int dim = mesh.Dimension();
		MFEM_VERIFY(dim == 2 || dim == 3,
			"ParaView export supports 2D and 3D meshes, not " << dim << "D.");

		mfem::ParaViewDataCollection pv(collection_name, &mesh);
		pv.SetPrefixPath(directory.string());

		// SetHighOrderOutput emits Lagrange cells of the solution order rather
		// than subdividing into linear cells.
		pv.SetLevelsOfDetail(solution_order);
		pv.SetHighOrderOutput(solution_order > 1);

		// Derived fields live in L2 spaces one order below the H1 solution.
		// ParaView stores raw pointers, so the backing spaces and grid functions
		// must outlive Save(); a std::list never relocates its elements, so the
		// registered addresses stay valid as we append.
		const int l2_order = std::max(0, solution_order - 1);
		mfem::L2_FECollection fec_l2(l2_order, dim);
		mfem::FiniteElementSpace fes_l2_scalar(&mesh, &fec_l2);
		std::list<mfem::FiniteElementSpace> vector_spaces;
		std::list<mfem::GridFunction> derived;

		for (const auto& f : fields.Fields()) {
			switch (f.kind) {
				case FieldExport::Kind::Primary:
					pv.RegisterField(f.name, f.primary);
					break;
				case FieldExport::Kind::DerivedScalar:
					derived.emplace_back(&fes_l2_scalar);
					derived.back().ProjectCoefficient(*f.scalar);
					pv.RegisterField(f.name, &derived.back());
					break;
				case FieldExport::Kind::DerivedVector:
					vector_spaces.emplace_back(&mesh, &fec_l2, f.vector->GetVDim());
					derived.emplace_back(&vector_spaces.back());
					derived.back() = 0.0;
					derived.back().ProjectCoefficient(*f.vector);
					pv.RegisterField(f.name, &derived.back());
					break;
			}
		}

		pv.SetCycle(0);
		pv.SetTime(0.0);
		pv.Save();
		MFEM_VERIFY(pv.Error() == mfem::DataCollection::NO_ERROR,
			"Failed to write ParaView collection " + collection_name);

		Reporter().Diagnostic("Wrote ParaView collection " + collection_name);
	}

	// Gmsh serializer. Emits the mesh as native Gmsh Lagrange elements of the
	// solution order, with each field sampled at that element's own node
	// lattice, so Gmsh reconstructs the field with matching high-order shape
	// functions instead of a tessellated linear approximation.
	void WriteGmsh(const std::filesystem::path& out_path,
				   const FieldExportSet& fields,
				   gmsh_results::MshVersion gmsh_version) const
	{
		// Node layouts exist for triangles and quadrilaterals (2D) and for
		// tetrahedra and hexahedra (3D); an unsupported element type is
		// reported by the layout code itself.
		MFEM_VERIFY(mesh.Dimension() == 2 || mesh.Dimension() == 3,
			"Gmsh export supports 2D and 3D meshes, not "
			<< mesh.Dimension() << "D.");

		const int order = std::max(1, solution_order);
		std::vector<gmsh_results::View> views;

		for (const auto& f : fields.Fields()) {
			switch (f.kind) {
				case FieldExport::Kind::Primary: {
					// A scalar primary (H1) is continuous and goes out as nodal
					// data. A vector primary (the 3D Nedelec A) is only
					// tangentially continuous, so it goes out per element.
					if (f.primary->VectorDim() == 1) {
						views.push_back(
							gmsh_results::MakeScalarNodeView(f.name, *f.primary));
					} else {
						views.push_back(
							gmsh_results::MakeVectorGridFunctionView(f.name, *f.primary));
					}
					break;
				}
				case FieldExport::Kind::DerivedScalar:
					views.push_back(
						gmsh_results::MakeScalarCoefficientView(f.name, *f.scalar));
					break;
				case FieldExport::Kind::DerivedVector:
					views.push_back(
						gmsh_results::MakeVectorCoefficientView(f.name, *f.vector));
					break;
			}
		}

		gmsh_results::WriteGmshResults(out_path.string(), mesh, order, views,
									   gmsh_version);
		Reporter().Diagnostic("Wrote " + out_path.string());
	}

private:
	mfem::Mesh& mesh;
	int solution_order;

	StatusReporter& Reporter() const {
		return StatusReporter::Global();
	}
};
