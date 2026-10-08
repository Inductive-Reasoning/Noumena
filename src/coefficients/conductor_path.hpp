// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <vector>

#include "mfem.hpp"
#include "../core/constants.hpp"
#include "../core/problem_config.hpp"
#include "../linalg/subset_solve.hpp"

/**
 * @brief Where current flows in a 3D conductor.
 *
 * Every current direction is described the same way: by w = -grad v, with v a
 * unit "conduction potential" of the conductor (v falls by 1 along the current
 * path). How the current spreads over the conductor then depends only on the
 * conductor type:
 *
 *   stranded  J = (I / A_cs) w / |w|,   A_cs = integral |w| dV
 *             Uniform magnitude along the path, as in a bundle of equal fine
 *             strands. A_cs is the cross-section: for a divergence-free J the
 *             current through any cross-section is integral(J . w) = I.
 *   massive   J = sigma V w,  V = I / G,  G = integral sigma |w|^2 dV
 *             The DC conduction distribution, with G the DC conductance. In a
 *             time-harmonic solve V becomes the conductor's port voltage.
 *
 * The potential comes from the terminal's direction (CurrentDirection):
 *
 *   Azimuthal  - v = -theta / extent about an axis: w = phi-hat / (extent r),
 *                analytic, with extent the conductor's angular extent: 2 pi
 *                for a full ring, less for a sector whose ends lie on
 *                n x A = 0 symmetry planes. It is the exact DC potential of
 *                any conductor of revolution, whatever its (axisymmetric)
 *                conductivity.
 *   Electrodes - v = 1 on the input faces, 0 on the output faces, harmonic in
 *                the conductor.
 *   Cut        - a closed conductor: v jumps by 1 across an internal cut
 *                surface. Imposed as v = u + H with u continuous and H the
 *                "thick cut" lifting: on each conductor element touching the
 *                cut on its downstream (+normal) side, H is the sum of the H1
 *                basis functions of the cut DOFs, and 0 elsewhere. H is
 *                continuous everywhere except across the cut, where it jumps
 *                by 1. Which side of the cut an element lies on follows from
 *                the mesh's connectivity (ConductionPath::BuildCutLifting);
 *                the normal only orients the cut. The cut must be a single
 *                two-sided sheet that severs the conductor, its rim lying on
 *                the conductor surface.
 */
class ConductorPath {
public:
	virtual ~ConductorPath() = default;
	/// w = -grad v at @p ip of the conductor element in @p T.
	virtual void Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip,
					  mfem::Vector& w) const = 0;
};

/// Analytic azimuthal path about an axis (right-hand rule).
class AzimuthalPath : public ConductorPath {
public:
	/// @param extent  Angular extent of the conductor [rad], over which v falls
	///                by 1: 2 pi for a full ring.
	explicit AzimuthalPath(const CurrentDirection& d, double extent = Constants::TWO_PI);

	/// Azimuthal angle of @p x about the axis, in [0, 2 pi), increasing
	/// along phi-hat.
	double Angle(const mfem::Vector& x) const;

	/// The unit azimuthal direction phi-hat at @p x.
	void Tangent(const mfem::Vector& x, mfem::Vector& t) const;

	/// Distance of @p x from the axis.
	double RadiusOf(const mfem::Vector& x) const {
		double d[3];
		return Perpendicular(x, d);
	}

	void Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip,
			  mfem::Vector& w) const override;

private:
	double extent;
	double origin[3]{};
	double axis[3]{};  // unit
	double e1[3]{}, e2[3]{};  // unit, across the axis: e1 x e2 = axis

	double Perpendicular(const mfem::Vector& x, double* d) const;
};

/// Path from a solved conduction potential (electrodes or a cut).
///
/// The potential is solved on the parent mesh's H1 space, assembled on the
/// conductor's elements only, for the conductor's free DOFs; no submesh (or
/// element map) is needed.
class ConductionPath : public ConductorPath {
public:
	/// Smallest |cos| between the cut normal and the normal of the cut face
	/// it crosses most squarely. The normal only says which way the current
	/// crosses the cut, so it must cross it clearly somewhere; which side of
	/// the cut each element lies on is then found topologically.
	static constexpr double kMinCutCrossing = 0.5;

	/// @param conductor     Domain-attribute marker of the conductor.
	/// @param conductivity  sigma for the potential (massive conductors, so v
	///                      is the true DC potential); nullptr means uniform
	///                      (stranded conductors, where only the direction
	///                      matters).
	/// @param input/output  Boundary-attribute markers of the electrodes
	///                      (Electrodes); @p cut marks the cut (Cut).
	ConductionPath(mfem::Mesh& mesh, int order, const mfem::Array<int>& conductor,
				   mfem::Coefficient* conductivity, const CurrentDirection& d,
				   const mfem::Array<int>& input, const mfem::Array<int>& output,
				   const mfem::Array<int>& cut);

	void Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip,
			  mfem::Vector& w) const override;

private:
	mfem::H1_FECollection fec;
	mfem::FiniteElementSpace fes;
	mfem::GridFunction v;                    // u (cut) or v (electrodes)
	std::map<int, std::vector<int>> lifted;  // element -> local cut DOFs (+ side)

	// The thick-cut lifting H and its load -integral(sigma grad H . grad w)
	// over the downstream elements.
	//
	// H must be continuous across every face but the cut's, so the side of
	// the cut each nearby element lies on is decided by connectivity, not
	// geometry: over the conductor elements sharing a DOF with the cut,
	// crossing an ordinary face keeps the side and crossing a cut face flips
	// it. Around every cut vertex those elements form a ball (a half-ball on
	// the conductor surface) that the cut splits in two, so an element
	// touching the cut only at a vertex or an edge still gets its side
	// through faces. One geometric decision seeds it: at the cut face the
	// normal crosses most squarely, the element ahead along the normal is
	// downstream. A conflict while propagating means the cut is not a
	// two-sided sheet.
	void BuildCutLifting(mfem::Mesh& mesh, const std::vector<bool>& inside,
						 const std::vector<int>& cut_group_faces,
						 const std::array<double, 3>& normal, mfem::Coefficient& sigma,
						 mfem::Vector& rhs);

	// A cut must sever the conductor: every edge on its rim (an edge of just
	// one cut face) has to lie on the conductor's surface. A cut that stops
	// short leaves its jump ending inside the conductor, where the potential
	// has nowhere to recover it, putting a singular current around that rim.
	static void RequireCutSpansConductor(mfem::Mesh& mesh, const std::vector<bool>& inside,
										 const std::vector<int>& cut_faces);

	// Solve K v = rhs for the DOFs not in @p fixed, which keep their values
	// in v.
	void SolveFree(const mfem::SparseMatrix& K, const mfem::Vector& rhs,
				   const std::set<int>& fixed);
};

/**
 * @brief Unit-current density of a conductor along its path.
 *
 * Stranded (@p conductivity == nullptr): scale * w / |w|, with
 * scale = 1 / A_cs. Massive: scale * sigma * w, with scale = 1 / G for the DC
 * distribution, or 1 for the field sigma w of a unit port voltage.
 */
class ConductorCurrentCoefficient : public mfem::VectorCoefficient {
public:
	ConductorCurrentCoefficient(const ConductorPath& path, mfem::Coefficient* conductivity,
								double scale)
		: mfem::VectorCoefficient(3), path(path), conductivity(conductivity), scale(scale) {}

	void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
			  const mfem::IntegrationPoint& ip) override;

private:
	const ConductorPath& path;
	mfem::Coefficient* conductivity;
	double scale;
};

/// Over a conductor: A_cs = integral |w| dV (@p conductivity == nullptr), or
/// the DC conductance G = integral sigma |w|^2 dV.
double ConductorPathIntegral(mfem::Mesh& mesh, const mfem::Array<int>& conductor,
									const ConductorPath& path, mfem::Coefficient* conductivity,
									int order);

/// integral |J|^2 dV over a conductor, for a current density J such as a
/// ConductorCurrentCoefficient.
double ConductorCurrentNormSquared(mfem::Mesh& mesh, const mfem::Array<int>& conductor,
										  mfem::VectorCoefficient& J, int order);
