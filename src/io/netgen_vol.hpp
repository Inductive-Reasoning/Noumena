// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <istream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief Parser for Netgen's native mesh format (.vol), which MFEM cannot read.
 *
 * Netgen writes it with Mesh::Save (Python Mesh.Save, nglib Ng_SaveMesh). It is
 * the one Netgen output that keeps both the boundary elements and the names of
 * the domains and boundaries, in 2D and 3D alike. The layout follows
 * Mesh::Load in Netgen's libsrc/meshing/meshclass.cpp:
 *
 *   - 3D: "volumeelements" rows are `matnr np p1..pnp`; the material number is
 *     the domain attribute and "materials" names it. "surfaceelements" rows
 *     are `surfnr bcnr domin domout np p1..pnp`; the boundary condition number
 *     bcnr is the boundary attribute and "bcnames" names it.
 *   - 2D: the domain elements are the "surfaceelements". Each belongs to the
 *     face descriptor matching its (surfnr - 1, bcnr, domin, domout); that
 *     descriptor's 1-based index is the domain attribute and "materials" names
 *     it. The boundary edges are "edgesegmentsgi3" rows ending in a 0-based
 *     edge descriptor index; that index + 1 is the boundary attribute and
 *     "bcnames" names it.
 *
 * Netgen orders the vertices of its volume elements with the opposite
 * handedness to MFEM's: every tetrahedron and prism of a Netgen mesh is
 * inverted to MFEM. The parser mirrors each volume element into MFEM's order,
 * which also covers the prisms and hexahedra MFEM cannot reorient itself.
 *
 * Netgen's name for an unnamed entity, "default", is not reported as a name.
 * Only first-order elements are read; Netgen writes second-order (curved)
 * elements with more nodes per row, which are rejected.
 */
namespace netgen_vol {

/// One element or boundary element: its attribute and 0-based vertex indices.
struct Cell {
	int attribute = 0;
	std::vector<int> vertices;
};

struct VolMesh {
	int dimension = 0;
	std::vector<std::array<double, 3>> points;
	std::vector<Cell> elements;
	std::vector<Cell> boundary;
	/// Attribute -> name, for the domains and the boundaries.
	std::map<int, std::string> domain_names;
	std::map<int, std::string> boundary_names;
};

/// True when the text is a Netgen mesh: its first line that is not blank or a
/// '#' comment is "mesh3d" (Netgen writes that header in 2D as well).
inline bool IsVolMesh(const std::string& text) {
	std::istringstream lines(text);
	std::string line;
	while (std::getline(lines, line)) {
		const auto first = line.find_first_not_of(" \t\r");
		if (first == std::string::npos || line[first] == '#') { continue; }
		const auto last = line.find_last_not_of(" \t\r");
		return line.substr(first, last - first + 1) == "mesh3d";
	}
	return false;
}

namespace detail {

[[noreturn]] inline void Fail(const std::string& what) {
	throw std::runtime_error("Netgen .vol mesh: " + what);
}

template <typename T>
T Read(std::istream& in, const char* section) {
	T value{};
	if (!(in >> value)) { Fail(std::string("truncated or malformed '") + section + "' section"); }
	return value;
}

inline int ReadCount(std::istream& in, const char* section) {
	const int n = Read<int>(in, section);
	if (n < 0) { Fail(std::string("negative count in '") + section + "' section"); }
	return n;
}

// `nr name` on one line; the name is the rest of the line, as Netgen's
// ReadNumberAndName reads it, so it may contain spaces.
inline std::pair<int, std::string> ReadNumberAndName(std::istream& in, const char* section) {
	std::string line;
	while (std::getline(in, line)) {
		std::istringstream fields(line);
		int nr = 0;
		if (!(fields >> nr)) { continue; }
		std::string name;
		std::getline(fields >> std::ws, name);
		while (!name.empty() && (name.back() == '\r' || name.back() == ' ' || name.back() == '\t')) {
			name.pop_back();
		}
		return {nr, name};
	}
	Fail(std::string("truncated '") + section + "' section");
}

inline void AddName(std::map<int, std::string>& names, int nr, const std::string& name) {
	if (nr >= 1 && !name.empty() && name != "default") { names[nr] = name; }
}

// Mirror a Netgen volume element into MFEM's vertex order: swap two corners
// of each triangular or quadrilateral end, keeping the other vertices'
// correspondence (tetrahedron 0<->1, pyramid base 1<->3, prism 1<->2 and
// 4<->5, hexahedron 1<->3 and 5<->7).
inline void MirrorVolumeElement(std::vector<int>& v) {
	switch (v.size()) {
		case 4: std::swap(v[0], v[1]); break;
		case 5: std::swap(v[1], v[3]); break;
		case 6: std::swap(v[1], v[2]); std::swap(v[4], v[5]); break;
		case 8: std::swap(v[1], v[3]); std::swap(v[5], v[7]); break;
	}
}

inline Cell ReadVertices(std::istream& in, int attribute, int np, const char* section) {
	Cell cell;
	cell.attribute = attribute;
	cell.vertices.resize(np);
	for (int& v : cell.vertices) { v = Read<int>(in, section) - 1; }
	return cell;
}

} // namespace detail

/// Parse a Netgen .vol mesh. Throws std::runtime_error on malformed input.
inline VolMesh Parse(std::istream& in) {
	using namespace detail;

	// Netgen's own reader scans whitespace-separated tokens for section
	// keywords; '#' comment lines are dropped first so their words cannot be
	// taken for one.
	std::stringstream body;
	{
		std::string line;
		while (std::getline(in, line)) {
			const auto first = line.find_first_not_of(" \t\r");
			if (first != std::string::npos && line[first] == '#') { continue; }
			body << line << '\n';
		}
	}

	struct FaceDescriptor { int surfnr, domin, domout, bcprop; };
	struct SurfaceRow { int surfnr, bcprop, domin, domout; Cell cell; };

	VolMesh mesh;
	std::vector<FaceDescriptor> face_descriptors;
	std::vector<SurfaceRow> surface_rows;
	std::vector<Cell> segments;  // the boundary in 2D; feature edges in 3D
	std::map<int, std::string> bcnames;
	bool have_segments_gi3 = false;
	bool have_legacy_segments = false;

	std::string token;
	while (body >> token) {
		if (token == "endmesh") { break; }
		if (token == "dimension") {
			mesh.dimension = Read<int>(body, "dimension");
		}
		else if (token == "facedescriptors") {
			const int n = ReadCount(body, "facedescriptors");
			for (int i = 0; i < n; ++i) {
				FaceDescriptor fd{};
				fd.surfnr = Read<int>(body, "facedescriptors");
				fd.domin = Read<int>(body, "facedescriptors");
				fd.domout = Read<int>(body, "facedescriptors");
				Read<int>(body, "facedescriptors");  // tlosurf
				fd.bcprop = Read<int>(body, "facedescriptors");
				face_descriptors.push_back(fd);
			}
		}
		else if (token == "surfaceelements" || token == "surfaceelementsgi"
				 || token == "surfaceelementsuv") {
			const bool gi = token == "surfaceelementsgi";
			const bool uv = token == "surfaceelementsuv";
			const int n = ReadCount(body, "surfaceelements");
			for (int i = 0; i < n; ++i) {
				SurfaceRow row{};
				row.surfnr = Read<int>(body, "surfaceelements") - 1;
				row.bcprop = Read<int>(body, "surfaceelements");
				row.domin = Read<int>(body, "surfaceelements");
				row.domout = Read<int>(body, "surfaceelements");
				int np = Read<int>(body, "surfaceelements");
				if (np == 0) { np = 3; }
				if (np != 3 && np != 4) {
					Fail("surface element with " + std::to_string(np) + " nodes; only "
						 "first-order triangles and quadrilaterals are supported (a "
						 "second-order mesh is not)");
				}
				row.cell = ReadVertices(body, 0, np, "surfaceelements");
				for (int j = 0; j < (gi ? np : uv ? 2 * np : 0); ++j) {
					Read<double>(body, "surfaceelements");  // geometry info
				}
				surface_rows.push_back(std::move(row));
			}
		}
		else if (token == "volumeelements") {
			const int n = ReadCount(body, "volumeelements");
			for (int i = 0; i < n; ++i) {
				int matnr = Read<int>(body, "volumeelements");
				if (matnr == 0) { matnr = 1; }  // as Netgen reads it
				const int np = Read<int>(body, "volumeelements");
				if (np != 4 && np != 5 && np != 6 && np != 8) {
					Fail("volume element with " + std::to_string(np) + " nodes; only "
						 "first-order tetrahedra, pyramids, prisms and hexahedra are "
						 "supported (a second-order mesh is not)");
				}
				Cell cell = ReadVertices(body, matnr, np, "volumeelements");
				MirrorVolumeElement(cell.vertices);
				mesh.elements.push_back(std::move(cell));
			}
		}
		else if (token == "edgesegmentsgi3") {
			have_segments_gi3 = true;
			const int n = ReadCount(body, "edgesegmentsgi3");
			for (int i = 0; i < n; ++i) {
				Cell seg = ReadVertices(body, 0, 2, "edgesegmentsgi3");
				Read<int>(body, "edgesegmentsgi3");     // trignum1
				Read<int>(body, "edgesegmentsgi3");     // trignum2
				Read<double>(body, "edgesegmentsgi3");  // dist1
				Read<double>(body, "edgesegmentsgi3");  // dist2
				seg.attribute = Read<int>(body, "edgesegmentsgi3") + 1;
				segments.push_back(std::move(seg));
			}
		}
		else if (token == "edgesegments" || token == "edgesegmentsgi"
				 || token == "edgesegmentsgi2") {
			// Written by Netgen before 6.2. Consumed so the token scan stays in
			// step; 3D has no use for edges, and 2D rejects them below.
			have_legacy_segments = true;
			const int per_row = token == "edgesegments" ? 4 : token == "edgesegmentsgi" ? 6 : 12;
			const int n = ReadCount(body, "edgesegments");
			for (int i = 0; i < n * per_row; ++i) { Read<double>(body, "edgesegments"); }
		}
		else if (token == "points") {
			const int n = ReadCount(body, "points");
			mesh.points.resize(n);
			for (auto& p : mesh.points) {
				for (double& x : p) { x = Read<double>(body, "points"); }
			}
		}
		else if (token == "materials") {
			const int n = ReadCount(body, "materials");
			for (int i = 0; i < n; ++i) {
				const auto [nr, name] = ReadNumberAndName(body, "materials");
				AddName(mesh.domain_names, nr, name);
			}
		}
		else if (token == "bcnames") {
			const int n = ReadCount(body, "bcnames");
			for (int i = 0; i < n; ++i) {
				const auto [nr, name] = ReadNumberAndName(body, "bcnames");
				AddName(bcnames, nr, name);
			}
		}
		else if (token == "cd2names" || token == "cd3names" || token == "identificationnames") {
			// Names of lower-dimensional entities, which carry no attribute
			// here; consumed so a name cannot be taken for a keyword.
			const int n = ReadCount(body, token.c_str());
			for (int i = 0; i < n; ++i) {
				if (token == "identificationnames") { Read<std::string>(body, token.c_str()); }
				else { ReadNumberAndName(body, token.c_str()); }
			}
		}
		else if (token == "edgedescriptors") {
			// `ednr surf0 surf1 singleft singright tlosurf domin domout name`;
			// the names repeat "bcnames" in 2D. Consumed line by line.
			const int n = ReadCount(body, "edgedescriptors");
			std::string line;
			std::getline(body, line);
			for (int i = 0; i < n; ++i) {
				if (!std::getline(body, line)) { Fail("truncated 'edgedescriptors' section"); }
			}
		}
	}

	if (mesh.dimension != 2 && mesh.dimension != 3) {
		Fail("missing or unsupported 'dimension' (" + std::to_string(mesh.dimension) + ")");
	}
	if (mesh.points.empty()) { Fail("no 'points' section"); }

	if (mesh.dimension == 3) {
		// Boundary faces: attribute = boundary condition number.
		for (auto& row : surface_rows) {
			if (row.bcprop < 1) {
				Fail("surface element with boundary condition number " +
					 std::to_string(row.bcprop) + "; every face needs a positive one");
			}
			row.cell.attribute = row.bcprop;
			mesh.boundary.push_back(std::move(row.cell));
		}
		mesh.boundary_names = std::move(bcnames);
		if (mesh.elements.empty()) { Fail("3D mesh without 'volumeelements'"); }
	}
	else {
		if (have_legacy_segments && !have_segments_gi3) {
			Fail("2D boundary segments in a format older than Netgen 6.2 "
				 "('edgesegmentsgi2'); save the mesh with Netgen 6.2 or newer");
		}
		mesh.boundary = std::move(segments);
		// Domain elements: attribute = index of the face descriptor the row
		// belongs to, matched as Netgen's loader matches it.
		for (auto& row : surface_rows) {
			int index = 0;
			for (std::size_t j = 0; j < face_descriptors.size(); ++j) {
				const auto& fd = face_descriptors[j];
				if (fd.surfnr == row.surfnr && fd.bcprop == row.bcprop &&
					fd.domin == row.domin && fd.domout == row.domout) {
					index = static_cast<int>(j) + 1;
				}
			}
			if (index == 0) {
				face_descriptors.push_back({row.surfnr, row.domin, row.domout, row.bcprop});
				index = static_cast<int>(face_descriptors.size());
			}
			row.cell.attribute = index;
			mesh.elements.push_back(std::move(row.cell));
		}
		mesh.boundary_names = std::move(bcnames);
		if (mesh.elements.empty()) { Fail("2D mesh without 'surfaceelements'"); }
	}

	const int np = static_cast<int>(mesh.points.size());
	for (const auto* cells : { &mesh.elements, &mesh.boundary }) {
		for (const Cell& cell : *cells) {
			for (int v : cell.vertices) {
				if (v < 0 || v >= np) {
					Fail("vertex " + std::to_string(v + 1) + " is out of range [1, " +
						 std::to_string(np) + "]");
				}
			}
		}
	}
	return mesh;
}

} // namespace netgen_vol
