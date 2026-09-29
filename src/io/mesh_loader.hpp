#pragma once

#include <istream>
#include <memory>
#include <string>
#include "mfem.hpp"

namespace mesh_io {

namespace detail {

class InputMesh final : public mfem::Mesh {
public:
	void Read(std::istream& input) {
		Loader(input, 1);
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
		Finalize(/*refine=*/Dimension() == 3, /*fix_orientation=*/false);
		const int bad_elements = CheckElementOrientation(false);
		const int bad_boundaries = CheckBdrElementOrientation(false);
		MFEM_VERIFY(bad_elements == 0 && bad_boundaries == 0,
			"Invalid mesh orientation: " << bad_elements << " domain elements and "
			<< bad_boundaries << " boundary elements; regenerate with consistent orientations.");
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