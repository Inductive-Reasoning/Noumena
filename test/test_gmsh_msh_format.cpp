// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// Covers both directions of the Gmsh MSH format contract:
//
//  INPUT  - MFEM 4.10 added MSH 4.1 reading alongside the long-standing 2.2
//           support. The mesher that produces case.msh emits 2.2 today, so the
//           guarantee we care about is that BOTH versions load and yield the
//           same element attributes, since attributes are what the config's
//           entity_groups/attribute_ids bind to. If 4.1 reading ever silently
//           dropped or renumbered attributes, every region and boundary
//           condition would bind to the wrong elements.
//
//  OUTPUT - gmsh_results_writer emits 2.2 by default and 4.1 on request. The
//           downstream C# consumer reads 2.2, so the default must not drift.
//           For 4.1 the attribute moves out of the element line and into
//           $Entities as a physical tag; these tests pin that mapping.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/catch_approx.hpp>

#include <filesystem>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "io/gmsh_results_writer.hpp"
#include "io/mesh_loader.hpp"

namespace {

// Two triangles forming a unit square, split across two physical surfaces
// (attributes 7 and 12, mirroring the winding/conductor ids used in the
// examples). The same mesh is expressed below in MSH 2.2 and MSH 4.1.
const char* kMsh22 = R"MSHFILE(
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
$EndNodes
$Elements
2
1 2 2 7 7 1 2 3
2 2 2 12 12 1 3 4
$EndElements
)MSHFILE";

const char* kMsh41 = R"MSHFILE(
$MeshFormat
4.1 0 8
$EndMeshFormat
$Entities
0 0 2 0
7 0 0 0 1 1 0 1 7 0
12 0 0 0 1 1 0 1 12 0
$EndEntities
$Nodes
1 4 1 4
2 7 0 4
1
2
3
4
0 0 0
1 0 0
1 1 0
0 1 0
$EndNodes
$Elements
2 2 1 2
2 7 2 1
1 1 2 3
2 12 2 1
2 1 3 4
$EndElements
)MSHFILE";

// MFEM's Gmsh reader works off a stream, so no temp files are needed.
mfem::Mesh LoadMsh(const char* text) {
	// Skip the leading newline the raw literal introduces.
	std::string s(text);
	if (!s.empty() && s.front() == '\n') { s.erase(0, 1); }
	std::istringstream in(s);
	auto mesh = mesh_io::LoadMesh(in);
	return std::move(*mesh);
}

std::vector<int> Attributes(mfem::Mesh& mesh) {
	std::vector<int> attrs;
	for (int e = 0; e < mesh.GetNE(); ++e) {
		attrs.push_back(mesh.GetAttribute(e));
	}
	return attrs;
}

std::string WriteToString(mfem::Mesh& mesh, int order,
						  gmsh_results::MshVersion version) {
	const auto path = std::filesystem::temp_directory_path()
					  / (version == gmsh_results::MshVersion::V4_1
							 ? "mfem_em_fmt_41.msh"
							 : "mfem_em_fmt_22.msh");
	gmsh_results::WriteGmshResults(path.string(), mesh, order, {}, version);

	std::ifstream in(path, std::ios::binary);
	std::ostringstream ss;
	ss << in.rdbuf();
	return ss.str();
}

bool Contains(const std::string& haystack, const std::string& needle) {
	return haystack.find(needle) != std::string::npos;
}

} // namespace

TEST_CASE("MSH 2.2 and 4.1 input meshes load identically", "[gmsh][msh][input]") {
	mfem::Mesh m22 = LoadMsh(kMsh22);
	mfem::Mesh m41 = LoadMsh(kMsh41);

	REQUIRE(m22.GetNE() == 2);
	REQUIRE(m41.GetNE() == m22.GetNE());
	REQUIRE(m41.GetNV() == m22.GetNV());
	REQUIRE(m41.Dimension() == m22.Dimension());

	// The contract that entity_groups/attribute_ids depend on: physical group
	// ids survive as MFEM element attributes, identically in both formats.
	REQUIRE(Attributes(m41) == Attributes(m22));
	REQUIRE(Attributes(m22) == std::vector<int>{ 7, 12 });
}

TEST_CASE("MSH 4.1 input preserves per-attribute element counts", "[gmsh][msh][input]") {
	mfem::Mesh m41 = LoadMsh(kMsh41);

	int n7 = 0, n12 = 0;
	for (int e = 0; e < m41.GetNE(); ++e) {
		if (m41.GetAttribute(e) == 7) { ++n7; }
		if (m41.GetAttribute(e) == 12) { ++n12; }
	}
	REQUIRE(n7 == 1);
	REQUIRE(n12 == 1);
}

TEST_CASE("Owned mesh loading releases rejected topology", "[gmsh][msh][input][mesh_loader]") {
	const std::string malformed =
		"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
		"$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n$EndNodes\n"
		"$Elements\n3\n1 1 2 1 1 2 4\n2 2 2 7 7 1 2 3\n"
		"3 2 2 12 12 1 3 4\n$EndElements\n";
	for (int attempt = 0; attempt < 3; ++attempt) {
		std::istringstream input(malformed);
		REQUIRE_THROWS_WITH(mesh_io::LoadMesh(input),
			Catch::Matchers::ContainsSubstring("does not match a mesh face"));
	}
	std::string valid(kMsh22);
	std::istringstream input(valid.substr(1));
	const auto mesh = mesh_io::LoadMesh(input);
	REQUIRE(mesh->GetNE() == 2);
}

TEST_CASE("Owned mesh loading normalizes boundary directions", "[gmsh][msh][input][mesh_loader]") {
	std::istringstream input(
		"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
		"$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n"
		"$Elements\n2\n1 1 2 1 1 2 1\n2 2 2 7 7 1 2 3\n$EndElements\n");
	const auto mesh = mesh_io::LoadMesh(input);
	REQUIRE(mesh->CheckBdrElementOrientation(false) == 0);
	REQUIRE(mesh->GetNBE() == 1);
}

TEST_CASE("Owned mesh loading releases failed parses", "[msh][input][mesh_loader]") {
	std::string text;
	SECTION("unknown format") {
		text = "Not a mesh\n";
	}
	SECTION("missing MFEM end tag after allocating topology") {
		text = "MFEM mesh v1.2\n\ndimension\n2\nelements\n1\n1 2 0 1 2\n"
			"boundary\n3\n1 1 0 1\n2 1 1 2\n3 1 2 0\n"
			"vertices\n3\n2\n0 0\n1 0\n0 1\n";
	}
	SECTION("invalid attribute after allocating Gmsh elements") {
		text = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
			"$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n$EndNodes\n"
			"$Elements\n2\n1 2 2 7 7 1 2 3\n2 2 2 0 12 1 3 4\n$EndElements\n";
	}
	SECTION("inverted domain element") {
		text = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
			"$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n"
			"$Elements\n1\n1 2 2 7 7 1 3 2\n$EndElements\n";
	}
	for (int attempt = 0; attempt < 3; ++attempt) {
		std::istringstream input(text);
		REQUIRE_THROWS_AS(mesh_io::LoadMesh(input), mfem::ErrorException);
	}
}

TEST_CASE("Capacitor example has connected materials and complete plate boundaries",
	"[gmsh][input][mesh_loader][examples]") {
	const auto path = std::filesystem::path(__FILE__).parent_path().parent_path()
		/ "examples/simple_capacitor/capacitor.mesh";
	auto mesh = mesh_io::LoadMesh(path.string());
	REQUIRE(mesh->Dimension() == 2);
	REQUIRE(mesh->attributes.Size() == 2);
	REQUIRE(mesh->attributes.Find(1) >= 0);
	REQUIRE(mesh->attributes.Find(2) >= 0);
	const auto& adjacency = mesh->ElementToElementTable();
	std::vector<bool> visited(mesh->GetNE(), false);
	std::vector<int> pending{0};
	visited[0] = true;
	for (std::size_t index = 0; index < pending.size(); ++index) {
		const int element = pending[index];
		for (int neighbor = 0; neighbor < adjacency.RowSize(element); ++neighbor) {
			const int adjacent = adjacency.GetRow(element)[neighbor];
			if (!visited[adjacent]) {
				visited[adjacent] = true;
				pending.push_back(adjacent);
			}
		}
	}
	REQUIRE(pending.size() == mesh->GetNE());
	std::array<std::array<bool, 4>, 2> plate_sides{};
	for (int boundary = 0; boundary < mesh->GetNBE(); ++boundary) {
		const int attribute = mesh->GetBdrAttribute(boundary);
		REQUIRE(attribute >= 1);
		REQUIRE(attribute <= 3);
		if (attribute == 3) continue;
		int inside, outside;
		mesh->GetFaceElements(mesh->GetBdrElementFaceIndex(boundary), &inside, &outside);
		REQUIRE(inside >= 0);
		REQUIRE(outside < 0);
		mfem::Array<int> vertices;
		mesh->GetBdrElementVertices(boundary, vertices);
		const double* first = mesh->GetVertex(vertices[0]);
		const double* second = mesh->GetVertex(vertices[1]);
		const double bottom = attribute == 1 ? 0.011 : 0.0;
		const double top = bottom + 0.001;
		const auto at = [](double value, double expected) { return std::abs(value - expected) < 1e-10; };
		if (at(first[0], 0.001) && at(second[0], 0.001)) plate_sides[attribute - 1][0] = true;
		if (at(first[0], 0.1) && at(second[0], 0.1)) plate_sides[attribute - 1][1] = true;
		if (at(first[1], bottom) && at(second[1], bottom)) plate_sides[attribute - 1][2] = true;
		if (at(first[1], top) && at(second[1], top)) plate_sides[attribute - 1][3] = true;
	}
	for (const auto& plate : plate_sides) {
		for (bool side : plate) REQUIRE(side);
	}
}

TEST_CASE("Results writer defaults to MSH 2.2", "[gmsh][msh][output]") {
	mfem::Mesh mesh = LoadMsh(kMsh22);

	// Called without a version argument, exactly as existing callers do.
	const auto path = std::filesystem::temp_directory_path()
					  / "mfem_em_fmt_default.msh";
	gmsh_results::WriteGmshResults(path.string(), mesh, 1, {});

	std::ifstream in(path, std::ios::binary);
	std::ostringstream ss;
	ss << in.rdbuf();
	const std::string out = ss.str();

	// The downstream C# consumer reads 2.2; this default must not drift.
	REQUIRE(Contains(out, "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"));
	REQUIRE_FALSE(Contains(out, "$Entities"));
}

TEST_CASE("MSH 4.1 output carries attributes as entity physical tags", "[gmsh][msh][output]") {
	mfem::Mesh mesh = LoadMsh(kMsh22);
	const std::string out =
		WriteToString(mesh, 1, gmsh_results::MshVersion::V4_1);

	REQUIRE(Contains(out, "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"));

	// One surface entity per distinct attribute, each carrying exactly one
	// physical tag equal to the attribute.
	REQUIRE(Contains(out, "$Entities\n0 0 2 0\n"));
	REQUIRE(Contains(out, " 1 7 0\n"));
	REQUIRE(Contains(out, " 1 12 0\n"));

	// Element blocks are keyed by (entityTag == attribute, elementType).
	// Type 2 is the 3-node triangle, one element per block here.
	REQUIRE(Contains(out, "2 7 2 1\n"));
	REQUIRE(Contains(out, "2 12 2 1\n"));
}

TEST_CASE("MSH 4.1 output round-trips through the MFEM reader", "[gmsh][msh][output]") {
	// The strongest available check that the emitted 4.1 is well formed:
	// read it back and confirm the mesh and its attributes are unchanged.
	mfem::Mesh original = LoadMsh(kMsh22);
	const std::string out =
		WriteToString(original, 1, gmsh_results::MshVersion::V4_1);

	std::istringstream in(out);
	mfem::Mesh reloaded(in, /*generate_edges=*/0, /*refine=*/0);

	REQUIRE(reloaded.GetNE() == original.GetNE());
	REQUIRE(reloaded.GetNV() == original.GetNV());
	REQUIRE(Attributes(reloaded) == Attributes(original));
}

namespace {

// A unit-cube mesh of @p etype elements whose lower and upper halves (in x)
// carry attributes 3 and 8.
mfem::Mesh MakeTwoAttributeCube(mfem::Element::Type etype) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(2, 1, 1, etype);
	for (int e = 0; e < mesh.GetNE(); ++e) {
		mfem::Vector c;
		mesh.GetElementCenter(e, c);
		mesh.SetAttribute(e, c(0) < 0.5 ? 3 : 8);
	}
	mesh.SetAttributes();
	return mesh;
}

double TotalVolume(mfem::Mesh& mesh) {
	double v = 0.0;
	for (int e = 0; e < mesh.GetNE(); ++e) { v += mesh.GetElementVolume(e); }
	return v;
}

} // namespace

TEST_CASE("MSH 4.1 output of a 3D mesh uses volume entities", "[gmsh][msh][output][3d]") {
	mfem::Mesh mesh = MakeTwoAttributeCube(mfem::Element::TETRAHEDRON);
	const std::string out = WriteToString(mesh, 1, gmsh_results::MshVersion::V4_1);

	// numPoints numCurves numSurfaces numVolumes: two volume entities.
	REQUIRE(Contains(out, "$Entities\n0 0 0 2\n"));
	REQUIRE(Contains(out, " 1 3 0\n"));
	REQUIRE(Contains(out, " 1 8 0\n"));
	// Element blocks: entityDim 3, entityTag = attribute, type 4 (4-node tet).
	REQUIRE(Contains(out, "\n3 3 4 "));
	REQUIRE(Contains(out, "\n3 8 4 "));
	// The interpolation scheme carries three exponent columns.
	REQUIRE(Contains(out, "\n4 3\n"));
}

TEST_CASE("3D MSH output round-trips through the MFEM reader", "[gmsh][msh][output][3d]") {
	for (auto etype : { mfem::Element::TETRAHEDRON, mfem::Element::HEXAHEDRON }) {
		for (auto version : { gmsh_results::MshVersion::V2_2,
							  gmsh_results::MshVersion::V4_1 }) {
			for (int order = 1; order <= 2; ++order) {
				mfem::Mesh original = MakeTwoAttributeCube(etype);
				const std::string out = WriteToString(original, order, version);

				std::istringstream in(out);
				mfem::Mesh reloaded(in, /*generate_edges=*/1, /*refine=*/0);

				REQUIRE(reloaded.Dimension() == 3);
				REQUIRE(reloaded.GetNE() == original.GetNE());
				REQUIRE(Attributes(reloaded) == Attributes(original));
				REQUIRE(TotalVolume(reloaded) == Catch::Approx(1.0).epsilon(1e-10));
			}
		}
	}
}
