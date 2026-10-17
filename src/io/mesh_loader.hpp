#pragma once

#include <istream>
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

// Gmsh's "$PhysicalNames" rows (`dim tag "name"`), which MFEM's Gmsh reader
// does not read, as named attribute sets: a name of the mesh dimension names
// domain attributes, one of a dimension less boundary attributes. The section
// is text in binary .msh files too, and precedes the binary data.
inline void AddGmshPhysicalNames(const std::string& text, mfem::Mesh& mesh) {
	const auto begin = text.find("$PhysicalNames");
	if (begin == std::string::npos) { return; }
	std::istringstream section(text.substr(begin));
	std::string line;
	std::getline(section, line);
	int n = 0;
	section >> n;
	std::getline(section, line);
	for (int i = 0; i < n && std::getline(section, line); ++i) {
		std::istringstream fields(line);
		int dim = 0, tag = 0;
		fields >> dim >> tag;
		const auto open = line.find('"');
		const auto close = line.rfind('"');
		MFEM_VERIFY(fields && open != std::string::npos && close > open,
			"Malformed $PhysicalNames row: " << line);
		const std::string name = line.substr(open + 1, close - open - 1);
		mfem::AttributeSets* sets =
			dim == mesh.Dimension() ? &mesh.attribute_sets
			: dim == mesh.Dimension() - 1 ? &mesh.bdr_attribute_sets : nullptr;
		if (!sets || name.empty()) { continue; }
		if (!sets->AttributeSetExists(name)) { sets->CreateAttributeSet(name); }
		sets->AddToAttributeSet(name, tag);
	}
}

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
			std::istringstream stream(text);
			Loader(stream, 1);
			AddGmshPhysicalNames(text, *this);
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
		const auto name_sets = [](mfem::AttributeSets& sets, const std::map<int, std::string>& names) {
			for (const auto& [attribute, name] : names) {
				if (!sets.AttributeSetExists(name)) { sets.CreateAttributeSet(name); }
				sets.AddToAttributeSet(name, attribute);
			}
		};
		name_sets(attribute_sets, vol.domain_names);
		name_sets(bdr_attribute_sets, vol.boundary_names);
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
