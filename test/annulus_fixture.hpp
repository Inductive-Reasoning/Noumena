// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// Annular-cylinder test geometry shared by the 3D magnetic solver tests and
// their axisymmetric references.

#pragma once

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>
#include <vector>

#include "config/input_parser.hpp"
#include "core/constants.hpp"
#include "mfem.hpp"

namespace annulus {

// Domain r_in <= r <= r_out, 0 <= z <= height about the z axis, with n x A = 0
// on every wall. For azimuthal currents that is EXACTLY the axisymmetric
// problem on the (r, z) rectangle with A_phi = 0 on its boundary (n x A = 0
// for an azimuthal A is A_phi = 0), so the 2D solvers are a reference with no
// modeling difference, only discretization. Conductors are rectangles in
// (r, z) aligned with the lattice; attribute 1 is air, 2, 3, ... the
// conductors.

// What a conductor is in the model.
enum class ConductorRole {
	Stranded,  // current terminal, no eddy currents (sigma = 0)
	Massive,   // current terminal of conducting material (a port in MQS)
	Passive    // conducting material with no terminal (a shield)
};

struct ConductorRect {
	double r0, r1, z0, z1;
	ConductorRole role = ConductorRole::Stranded;
};

struct AnnulusSpec {
	double r_in = 0.02, r_out = 0.10, height = 0.10;
	int nr = 8, nz = 10;
	std::vector<ConductorRect> conductors;
	double sigma = 5.8e7;  // of the Massive and Passive conductors
	// Describe terminal c by a cut (boundary attribute c + 2 on the theta = 0
	// half-plane, current crossing it along +y) instead of analytically.
	bool cut = false;
	// Make that cut a staircase instead of a plane: the radially outer half
	// of each conductor's cut sits one cell around (theta = 2 pi / n_theta),
	// joined to the inner half by the constant-r faces between them.
	bool staircase_cut = false;
	// Angle the mesh spans about z. A sector (extent < 2 pi) is closed by the
	// meridian planes theta = 0 (boundary attribute kStartPlane) and theta =
	// extent (kEndPlane), n x A = 0 walls like the rest: for azimuthal fields
	// that is exact, and every inductance and resistance is the full ring's
	// times extent / (2 pi).
	double extent = Constants::TWO_PI;
	// Describe the terminals of a sector by electrodes on its two meridian
	// planes (current from theta = 0 towards theta = extent) instead of
	// analytically.
	bool electrodes = false;

	bool Sector() const { return extent < Constants::TWO_PI; }
};

constexpr int kStartPlane = 10, kEndPlane = 11;  // above every conductor's cut attribute

inline int CellAttribute(const AnnulusSpec& spec, double r, double z) {
	for (size_t c = 0; c < spec.conductors.size(); ++c) {
		const auto& k = spec.conductors[c];
		if (r > k.r0 && r < k.r1 && z > k.z0 && z < k.z1) return static_cast<int>(c) + 2;
	}
	return 1;
}

inline mfem::Mesh MakeAnnulus2D(const AnnulusSpec& spec, int refine) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(spec.nr * refine, spec.nz * refine,
		mfem::Element::QUADRILATERAL, true, spec.r_out - spec.r_in, spec.height);
	for (int v = 0; v < mesh.GetNV(); ++v) { mesh.GetVertex(v)[0] += spec.r_in; }
	for (int e = 0; e < mesh.GetNE(); ++e) {
		mfem::Vector c;
		mesh.GetElementCenter(e, c);
		mesh.SetAttribute(e, CellAttribute(spec, c(0), c(1)));
	}
	mesh.SetAttributes();
	return mesh;
}

// The same lattice revolved about z with n_theta cells around, as curved
// (order-2 geometry) hexahedra whose nodes are snapped radially onto their
// lattice circle, or with @p curved false as straight-sided hexahedra (a
// faceted cylinder). Boundary attribute 1 on every wall but a sector's
// meridian planes. n_theta is the number of cells across spec.extent.
inline mfem::Mesh MakeAnnulus3D(const AnnulusSpec& spec, int n_theta, bool curved = true) {
	REQUIRE(!(spec.cut && spec.Sector()));
	REQUIRE(!(spec.electrodes && !spec.Sector()));
	const int nr = spec.nr, nz = spec.nz;
	const double dr = (spec.r_out - spec.r_in) / nr, dz = spec.height / nz;
	const int n_around = spec.Sector() ? n_theta + 1 : n_theta;  // vertices per circle
	auto id = [&](int i, int j, int k) { return (k * (nr + 1) + i) * n_around + (j % n_around); };

	// Cut quads of each conductor cell row (k): the cells i of conductor
	// attribute a, and for a staircase the radial index where it steps.
	auto attr_of = [&](int i, int k) {
		return CellAttribute(spec, spec.r_in + (i + 0.5) * dr, (k + 0.5) * dz);
	};
	auto step_of = [&](int i, int k) {  // first cell of the row's outer half
		const int a = attr_of(i, k);
		int i0 = i, i1 = i;
		while (i0 > 0 && attr_of(i0 - 1, k) == a) --i0;
		while (i1 + 1 < nr && attr_of(i1 + 1, k) == a) ++i1;
		return (i0 + i1 + 1) / 2 + (i1 == i0 ? 1 : 0);
	};
	int n_cut = 0;
	for (int k = 0; spec.cut && k < nz; ++k) {
		for (int i = 0; i < nr; ++i) {
			if (attr_of(i, k) == 1) continue;
			++n_cut;
			if (spec.staircase_cut && i == step_of(i, k) && attr_of(i - 1, k) == attr_of(i, k)) ++n_cut;
		}
	}
	const int n_planes = spec.Sector() ? 2 * nr * nz : 0;
	mfem::Mesh mesh(3, (nr + 1) * (nz + 1) * n_around, nr * nz * n_theta,
					2 * n_theta * (nr + nz) + n_cut + n_planes, 3);
	for (int k = 0; k <= nz; ++k) {
		for (int i = 0; i <= nr; ++i) {
			for (int j = 0; j < n_around; ++j) {
				const double r = spec.r_in + i * dr, t = spec.extent * j / n_theta;
				mesh.AddVertex(r * std::cos(t), r * std::sin(t), k * dz);
			}
		}
	}
	for (int k = 0; k < nz; ++k) {
		for (int i = 0; i < nr; ++i) {
			const int attr = CellAttribute(spec, spec.r_in + (i + 0.5) * dr, (k + 0.5) * dz);
			for (int j = 0; j < n_theta; ++j) {
				// (r, theta, z) is right-handed, so this corner order is too.
				const int v[8] = { id(i, j, k), id(i + 1, j, k), id(i + 1, j + 1, k), id(i, j + 1, k),
								   id(i, j, k + 1), id(i + 1, j, k + 1), id(i + 1, j + 1, k + 1),
								   id(i, j + 1, k + 1) };
				mesh.AddHex(v, attr);
			}
		}
	}
	for (int j = 0; j < n_theta; ++j) {
		for (int k = 0; k < nz; ++k) {
			for (int i : { 0, nr }) {
				const int q[4] = { id(i, j, k), id(i, j + 1, k), id(i, j + 1, k + 1), id(i, j, k + 1) };
				mesh.AddBdrQuad(q, 1);
			}
		}
		for (int i = 0; i < nr; ++i) {
			for (int k : { 0, nz }) {
				const int q[4] = { id(i, j, k), id(i + 1, j, k), id(i + 1, j + 1, k), id(i, j + 1, k) };
				mesh.AddBdrQuad(q, 1);
			}
		}
	}
	for (int k = 0; spec.Sector() && k < nz; ++k) {
		for (int i = 0; i < nr; ++i) {
			for (int j : { 0, n_theta }) {
				const int q[4] = { id(i, j, k), id(i + 1, j, k), id(i + 1, j, k + 1), id(i, j, k + 1) };
				mesh.AddBdrQuad(q, j == 0 ? kStartPlane : kEndPlane);
			}
		}
	}
	for (int k = 0; spec.cut && k < nz; ++k) {
		for (int i = 0; i < nr; ++i) {
			const int attr = attr_of(i, k);
			if (attr == 1) continue;
			const int j = spec.staircase_cut && i >= step_of(i, k) ? 1 : 0;
			const int q[4] = { id(i, j, k), id(i + 1, j, k), id(i + 1, j, k + 1), id(i, j, k + 1) };
			mesh.AddBdrQuad(q, attr);  // conductor c -> attribute c + 2
			if (spec.staircase_cut && i == step_of(i, k) && attr_of(i - 1, k) == attr) {
				const int riser[4] = { id(i, 0, k), id(i, 1, k), id(i, 1, k + 1), id(i, 0, k + 1) };
				mesh.AddBdrQuad(riser, attr);
			}
		}
	}
	mesh.FinalizeHexMesh(1, 0, true);
	if (!curved) { return mesh; }

	// The radial snap below is unambiguous only while the chord sag of an
	// angular edge stays under a quarter of the radial spacing; beyond that a
	// node can snap to the wrong circle and fold its element.
	const double sag = spec.r_out * (1.0 - std::cos(0.5 * spec.extent / n_theta));
	REQUIRE(sag < 0.25 * dr);

	mesh.SetCurvature(2);
	const double half = 0.5 * dr;
	mesh.Transform([&](const mfem::Vector& x, mfem::Vector& y) {
		y = x;
		const double r = std::hypot(x(0), x(1));
		const double snapped = spec.r_in + half * std::round((r - spec.r_in) / half);
		y(0) *= snapped / r;
		y(1) *= snapped / r;
	});
	return mesh;
}

// Config for the annulus in either model. Conductor c is entity group
// "C<c+1>"; the Stranded and Massive ones are current terminals of that name,
// azimuthal (or through cut "Cut<c+1>", or between the electrodes "Start"
// and "End" on a sector's meridian planes) in 3D. Stranded conductors are air;
// Massive and Passive ones conduct with spec.sigma.
inline json MakeAnnulusConfig(const AnnulusSpec& spec, bool three_d, int order,
							  const std::string& physics = "magnetostatics",
							  const std::string& analysis = "coupling_matrix") {
	json groups = json::array();
	json regions = json::array();
	json terminals = json::array();
	std::vector<int> air = { 1 };
	std::vector<int> walls = three_d ? std::vector<int>{1} : std::vector<int>{1, 2, 3, 4};
	if (three_d && spec.Sector()) {
		walls.push_back(kStartPlane);
		walls.push_back(kEndPlane);
		groups.push_back({{"name", "Start"}, {"dim", 2}, {"attribute_ids", {kStartPlane}}});
		groups.push_back({{"name", "End"}, {"dim", 2}, {"attribute_ids", {kEndPlane}}});
	}
	groups.push_back({{"name", "Walls"}, {"dim", three_d ? 2 : 1}, {"attribute_ids", walls}});
	for (size_t c = 0; c < spec.conductors.size(); ++c) {
		const ConductorRole role = spec.conductors[c].role;
		const int attr = static_cast<int>(c) + 2;
		const std::string name = "C" + std::to_string(c + 1);
		groups.push_back({{"name", name}, {"dim", three_d ? 3 : 2}, {"attribute_ids", {attr}}});
		if (role == ConductorRole::Stranded) {
			air.push_back(attr);
		} else {
			regions.push_back({{"name", name}, {"entity_group", name}, {"material", "Copper"}});
		}
		if (role == ConductorRole::Passive) continue;

		json terminal = {{"name", name}, {"quantity", "current"}, {"entity_group", name},
						 {"conductor_type", role == ConductorRole::Massive ? "massive" : "stranded"}};
		if (three_d && spec.cut) {
			const std::string cut = "Cut" + std::to_string(c + 1);
			groups.push_back({{"name", cut}, {"dim", 2}, {"attribute_ids", {attr}}});
			terminal["direction"] = {{"type", "cut"}, {"cut", cut}, {"normal", {0.0, 1.0, 0.0}}};
		} else if (three_d && spec.electrodes) {
			terminal["direction"] = {{"type", "electrodes"}, {"input", "Start"}, {"output", "End"}};
		} else if (three_d) {
			terminal["direction"] = {{"type", "azimuthal"}, {"origin", {0.0, 0.0, 0.0}},
									 {"axis", {0.0, 0.0, 1.0}}};
		}
		terminals.push_back(terminal);
	}
	groups.push_back({{"name", "Air"}, {"dim", three_d ? 3 : 2}, {"attribute_ids", air}});
	regions.push_back({{"name", "Air"}, {"entity_group", "Air"}, {"material", "Air"}});
	return json{
		{"simulation", {
			{"physics_type", physics}, {"mesh", "unused.mesh"}, {"order", order},
			{"geometry_type", three_d ? "3d" : "axisymmetric"},
			{"analysis_type", analysis}, {"linear_solver", "direct"}
		}},
		{"entity_groups", groups},
		{"regions", regions},
		{"materials", json::array({
			{{"name", "Air"}, {"properties", {{"mu_r", 1.0}}}},
			{{"name", "Copper"}, {"properties", {{"mu_r", 1.0}, {"sigma", spec.sigma}}}}})},
		{"terminals", terminals},
		{"boundary_conditions", json::array({
			{{"name", "Walls"}, {"type", "dirichlet"}, {"entity_group", "Walls"}, {"value", 0.0}}})},
		{"scenarios", json::array()}
	};
}

} // namespace annulus
