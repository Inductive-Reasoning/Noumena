#pragma once

#include <istream>
#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include "mfem.hpp"
#include "netgen_vol.hpp"

namespace mesh_io {

namespace detail {

class InputMesh final : public mfem::Mesh {
public:
	void Read(std::istream& input) {
		const std::string text{std::istreambuf_iterator<char>(input),
							   std::istreambuf_iterator<char>()};
		if (netgen_vol::IsVolMesh(text)) {
			std::istringstream stream(text);
			netgen_vol::VolMesh vol;
			try { vol = netgen_vol::Parse(stream); }
			catch (const std::runtime_error& e) { MFEM_ABORT(e.what()); }
			Build(vol);
		}
		else {
			// MFEM's readers keep the names a format carries (Gmsh
			// $PhysicalNames, MFEM attribute_sets) as named attribute sets.
			std::istringstream stream(text);
			Loader(stream, 1);
		}
		for (int boundary = 0; boundary < GetNBE(); ++boundary) {
			const int face = GetBdrElementFaceIndex(boundary);
			MFEM_VERIFY(face >= 0 && face < GetNumFaces(),
				"Boundary element " << boundary << " (attribute " << GetBdrAttribute(boundary)
				<< ") does not match a mesh face; regenerate a conforming mesh with shared boundary nodes.");
		}
		// refine=true marks a tetrahedral mesh for conforming bisection
		// (MarkForRefinement reorders each tet's vertices), which conforming
		// AMR on tets relies on. It is limited to 3D so the vertex order, and
		// therefore every exported artifact, of existing 2D meshes is unchanged.
		//
		// fix_orientation=true, as in MFEM's own Mesh constructors: a vertex
		// order of the opposite handedness is a convention (Netgen writes every
		// tetrahedron that way), not damage, and reordering the vertices
		// repairs it exactly. MFEM repairs triangles, quadrilaterals,
		// tetrahedra and pyramids; an element it cannot reorder (a hexahedron,
		// a prism) is rejected below.
		Finalize(/*refine=*/Dimension() == 3, /*fix_orientation=*/true);
		const int bad_elements = CheckElementOrientation(false);
		const int bad_boundaries = CheckBdrElementOrientation(false);
		MFEM_VERIFY(bad_elements == 0 && bad_boundaries == 0,
			"Invalid mesh orientation: " << bad_elements << " domain elements and "
			<< bad_boundaries << " boundary elements could not be reoriented; "
			"regenerate with consistent orientations.");
	}

private:
	// A Netgen mesh, with Netgen's domain and boundary numbers as attributes
	// and its names as named attribute sets.
	void Build(const netgen_vol::VolMesh& vol) {
		const int dim = vol.dimension;
		InitMesh(dim, dim, static_cast<int>(vol.points.size()),
				 static_cast<int>(vol.elements.size()), static_cast<int>(vol.boundary.size()));
		for (const auto& p : vol.points) { AddVertex(p[0], p[1], p[2]); }
		for (const auto& cell : vol.elements) {
			const int* v = cell.vertices.data();
			switch (cell.vertices.size()) {
				case 3: AddTri(v, cell.attribute); break;
				case 4: dim == 2 ? AddQuad(v, cell.attribute) : AddTet(v, cell.attribute); break;
				case 5: AddPyramid(v, cell.attribute); break;
				case 6: AddWedge(v, cell.attribute); break;
				case 8: AddHex(v, cell.attribute); break;
			}
		}
		for (const auto& cell : vol.boundary) {
			const int* v = cell.vertices.data();
			switch (cell.vertices.size()) {
				case 2: AddBdrSegment(v, cell.attribute); break;
				case 3: AddBdrTriangle(v, cell.attribute); break;
				case 4: AddBdrQuad(v, cell.attribute); break;
			}
		}
		FinalizeTopology(/*generate_bdr=*/false);
		if (!vol.edge_nodes.empty()) { SetEdgeNodes(vol); }
		const auto name_sets = [](mfem::AttributeSets& sets, const std::map<int, std::string>& names) {
			for (const auto& [attribute, name] : names) {
				if (!sets.AttributeSetExists(name)) { sets.CreateAttributeSet(name); }
				sets.AddToAttributeSet(name, attribute);
			}
		};
		name_sets(attribute_sets, vol.domain_names);
		name_sets(bdr_attribute_sets, vol.boundary_names);
	}

	// A second-order Netgen mesh as an order-2 curved mesh: its nodes are the
	// vertices plus, on each edge, the node Netgen placed on the geometry. On
	// an order-2 Lagrange edge the one interior node is the midpoint of the
	// reference edge, so the file's node is its nodal value.
	void SetEdgeNodes(const netgen_vol::VolMesh& vol) {
		SetCurvature(2, /*discontinuous=*/false, Dimension(), mfem::Ordering::byVDIM);
		mfem::GridFunction& nodes = *GetNodes();
		const mfem::FiniteElementSpace& fes = *nodes.FESpace();
		mfem::Array<int> vertices, dofs;
		for (int edge = 0; edge < GetNEdges(); ++edge) {
			GetEdgeVertices(edge, vertices);
			const auto node = vol.edge_nodes.find(std::minmax(vertices[0], vertices[1]));
			MFEM_VERIFY(node != vol.edge_nodes.end(), "Netgen .vol mesh: the edge between "
				"vertices " << vertices[0] << " and " << vertices[1] << " has no second-order node");
			fes.GetEdgeInteriorDofs(edge, dofs);
			MFEM_VERIFY(dofs.Size() == 1, "an order-2 edge has one interior node");
			for (int c = 0; c < Dimension(); ++c) {
				nodes(fes.DofToVDof(dofs[0], c)) = node->second[c];
			}
		}
	}
};

}

inline std::unique_ptr<mfem::Mesh> LoadMesh(std::istream& input) {
	auto mesh = std::make_unique<detail::InputMesh>();
	mesh->Read(input);
	return mesh;
}

inline std::unique_ptr<mfem::Mesh> LoadMesh(const std::string& path) {
	mfem::named_ifgzstream input(path);
	MFEM_VERIFY(input, "Cannot open mesh file '" << path << "'.");
	return LoadMesh(input);
}

}
