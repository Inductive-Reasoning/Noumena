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
#include "../linalg/amg_preconditioner.hpp"

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
 *   Azimuthal  - v = -theta / (2 pi) about an axis: w = phi-hat / (2 pi r),
 *                analytic. It is the exact DC potential of any conductor of
 *                revolution, whatever its (axisymmetric) conductivity.
 *   Electrodes - v = 1 on the input faces, 0 on the output faces, harmonic in
 *                the conductor.
 *   Cut        - a closed conductor: v jumps by 1 across an internal cut
 *                surface. Imposed as v = u + H with u continuous and H the
 *                "thick cut" lifting: on each conductor element touching the
 *                cut on its downstream (+normal) side, H is the sum of the H1
 *                basis functions of the cut DOFs, and 0 elsewhere. H is
 *                continuous everywhere except across the cut, where it jumps
 *                by 1. Which side of the cut an element lies on is read from
 *                the cut's normal, so the normal must cross every cut face
 *                clearly (ConductionPath::kMinCutCrossing).
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
	explicit AzimuthalPath(const CurrentDirection& d) {
		double norm = 0.0;
		for (int c = 0; c < 3; ++c) {
			origin[c] = d.Origin[c];
			axis[c] = d.Axis[c];
			norm += axis[c] * axis[c];
		}
		norm = std::sqrt(norm);
		MFEM_VERIFY(norm > 0.0, "Current direction axis must be a nonzero vector.");
		for (double& c : axis) { c /= norm; }
	}

	/// Distance of @p x from the axis.
	double RadiusOf(const mfem::Vector& x) const {
		double d[3];
		return Perpendicular(x, d);
	}

	void Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip,
			  mfem::Vector& w) const override {
		mfem::Vector x;
		T.Transform(ip, x);
		double d[3];
		const double r = Perpendicular(x, d);
		MFEM_VERIFY(r > 0.0, "The azimuthal current direction is undefined on its axis.");
		// phi-hat / (2 pi r) = (e x d) / (2 pi r^2)
		const double s = 1.0 / (Constants::TWO_PI * r * r);
		w.SetSize(3);
		w(0) = s * (axis[1] * d[2] - axis[2] * d[1]);
		w(1) = s * (axis[2] * d[0] - axis[0] * d[2]);
		w(2) = s * (axis[0] * d[1] - axis[1] * d[0]);
	}

private:
	double origin[3]{};
	double axis[3]{};  // unit

	double Perpendicular(const mfem::Vector& x, double* d) const {
		double along = 0.0;
		for (int c = 0; c < 3; ++c) {
			d[c] = x(c) - origin[c];
			along += d[c] * axis[c];
		}
		double r2 = 0.0;
		for (int c = 0; c < 3; ++c) {
			d[c] -= along * axis[c];
			r2 += d[c] * d[c];
		}
		return std::sqrt(r2);
	}
};

/// Path from a solved conduction potential (electrodes or a cut).
///
/// The potential is solved on the parent mesh's H1 space, assembled on the
/// conductor's elements only, for the conductor's free DOFs; no submesh (or
/// element map) is needed.
class ConductionPath : public ConductorPath {
public:
	/// Smallest |cos| allowed between the cut normal and any cut face's
	/// normal (the angle between them at most 60 degrees).
	///
	/// The side of the cut an element lies on is the sign of its offset from
	/// the nearest cut face along the normal, which is meaningful only where
	/// the normal actually crosses that face: a normal tangent to part of the
	/// cut classifies elements there arbitrarily, and a cut that folds back
	/// on itself flips the side the normal calls "downstream". Either puts a
	/// spurious unit jump inside the conductor. A connected cut can only fold
	/// past 90 degrees by passing through faces nearly tangent to the normal,
	/// so bounding the angle on every face rules out both, while allowing a
	/// normal well off the cut's own (a tilted or gently curved cut).
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
				   const mfem::Array<int>& cut)
		: fec(order, mesh.Dimension()), fes(&mesh, &fec), v(&fes) {
		std::vector<bool> inside(mesh.GetNE(), false);
		std::set<int> conductor_dofs;
		mfem::Array<int> dofs;
		for (int e = 0; e < mesh.GetNE(); ++e) {
			const int a = mesh.GetAttribute(e);
			if (a < 1 || a > conductor.Size() || !conductor[a - 1]) continue;
			inside[e] = true;
			fes.GetElementDofs(e, dofs);
			conductor_dofs.insert(dofs.begin(), dofs.end());
		}

		// Faces of the given boundary attributes that border the conductor.
		auto bordering_faces = [&](const mfem::Array<int>& marker) {
			std::vector<int> faces;
			for (int be = 0; be < mesh.GetNBE(); ++be) {
				const int a = mesh.GetBdrAttribute(be);
				if (a < 1 || a > marker.Size() || !marker[a - 1]) continue;
				const int f = mesh.GetBdrElementFaceIndex(be);
				int e1, e2;
				mesh.GetFaceElements(f, &e1, &e2);
				if ((e1 >= 0 && inside[e1]) || (e2 >= 0 && inside[e2])) faces.push_back(f);
			}
			return faces;
		};
		auto face_dofs = [&](const std::vector<int>& faces) {
			std::set<int> out;
			for (int f : faces) {
				fes.GetFaceDofs(f, dofs);
				out.insert(dofs.begin(), dofs.end());
			}
			return out;
		};

		v = 0.0;
		std::set<int> fixed;  // DOFs held at their value in v
		for (int i = 0; i < fes.GetVSize(); ++i) {
			if (!conductor_dofs.count(i)) fixed.insert(i);
		}

		mfem::ConstantCoefficient uniform(1.0);
		mfem::Coefficient& sigma = conductivity ? *conductivity : uniform;
		mfem::Array<int> marker(conductor);
		mfem::BilinearForm k(&fes);
		k.AddDomainIntegrator(new mfem::DiffusionIntegrator(sigma), marker);
		k.Assemble();
		k.Finalize();
		mfem::Vector rhs(fes.GetVSize());
		rhs = 0.0;

		if (d.Type == CurrentDirection::Kind::Electrodes) {
			const auto in = face_dofs(bordering_faces(input));
			const auto out = face_dofs(bordering_faces(output));
			MFEM_VERIFY(!in.empty() && !out.empty(),
				"The electrodes do not touch their conductor.");
			for (int i : in) { v(i) = 1.0; fixed.insert(i); }
			for (int i : out) { fixed.insert(i); }
		}
		else {
			BuildCutLifting(mesh, inside, bordering_faces(cut), d.Normal, sigma, rhs);
			fixed.insert(*conductor_dofs.begin());  // u is defined up to a constant
		}
		SolveFree(k.SpMat(), rhs, fixed);
	}

	void Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip,
			  mfem::Vector& w) const override {
		v.GetGradient(T, w);
		const auto lift = lifted.find(T.ElementNo);
		if (lift != lifted.end()) {
			const mfem::FiniteElement& fe = *fes.GetFE(T.ElementNo);
			mfem::DenseMatrix dshape(fe.GetDof(), fe.GetDim()), grad(fe.GetDof(), 3);
			fe.CalcDShape(ip, dshape);
			mfem::Mult(dshape, T.InverseJacobian(), grad);
			for (int j : lift->second) {
				for (int c = 0; c < 3; ++c) { w(c) += grad(j, c); }
			}
		}
		w.Neg();
	}

private:
	mfem::H1_FECollection fec;
	mfem::FiniteElementSpace fes;
	mfem::GridFunction v;                    // u (cut) or v (electrodes)
	std::map<int, std::vector<int>> lifted;  // element -> local cut DOFs (+ side)

	// The thick-cut lifting H and its load -integral(sigma grad H . grad w)
	// over the downstream elements.
	void BuildCutLifting(mfem::Mesh& mesh, const std::vector<bool>& inside,
						 const std::vector<int>& cut_faces,
						 const std::array<double, 3>& normal, mfem::Coefficient& sigma,
						 mfem::Vector& rhs) {
		MFEM_VERIFY(!cut_faces.empty(), "The cut does not cross its conductor.");
		mfem::Vector direction(3);
		for (int k = 0; k < 3; ++k) { direction(k) = normal[k]; }
		MFEM_VERIFY(direction.Norml2() > 0.0, "The cut normal must be a nonzero vector.");
		direction /= direction.Norml2();

		std::set<int> cut_dofs;
		std::vector<mfem::Vector> centers;
		mfem::Array<int> dofs;
		double worst = 1.0;
		mfem::Vector face_normal(3);
		for (int f : cut_faces) {
			fes.GetFaceDofs(f, dofs);
			cut_dofs.insert(dofs.begin(), dofs.end());
			mfem::Vector c;
			mfem::ElementTransformation* T = mesh.GetFaceTransformation(f);
			const mfem::IntegrationPoint& center = mfem::Geometries.GetCenter(T->GetGeometryType());
			T->SetIntPoint(&center);
			T->Transform(center, c);
			centers.push_back(c);
			mfem::CalcOrtho(T->Jacobian(), face_normal);
			worst = std::min(worst, std::abs(face_normal * direction) / face_normal.Norml2());
		}
		MFEM_VERIFY(worst >= kMinCutCrossing,
			"The cut normal (" << normal[0] << ", " << normal[1] << ", " << normal[2]
			<< ") is " << std::acos(worst) * 360.0 / Constants::TWO_PI << " degrees from the normal "
			"of part of the cut, beyond the 60 allowed: it must cross the cut, not run "
			"along it. Give the direction the current crosses the cut in, and keep the "
			"cut close to planar.");

		mfem::DiffusionIntegrator diffusion(sigma);
		mfem::DenseMatrix ke;
		for (int e = 0; e < mesh.GetNE(); ++e) {
			if (!inside[e]) continue;
			fes.GetElementDofs(e, dofs);
			std::vector<int> local;
			for (int j = 0; j < dofs.Size(); ++j) {
				if (cut_dofs.count(dofs[j])) local.push_back(j);
			}
			if (local.empty()) continue;

			// Downstream side: centroid ahead of the nearest cut face.
			mfem::Vector c;
			mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
			T->Transform(mfem::Geometries.GetCenter(T->GetGeometryType()), c);
			double best = std::numeric_limits<double>::max(), side = 0.0;
			for (const auto& fc : centers) {
				double dist = 0.0, ahead = 0.0;
				for (int k = 0; k < 3; ++k) {
					dist += (c(k) - fc(k)) * (c(k) - fc(k));
					ahead += (c(k) - fc(k)) * normal[k];
				}
				if (dist < best) { best = dist; side = ahead; }
			}
			if (side <= 0.0) continue;

			lifted[e] = local;
			diffusion.AssembleElementMatrix(*fes.GetFE(e), *T, ke);
			mfem::Vector h(dofs.Size()), load(dofs.Size());
			h = 0.0;
			for (int j : local) { h(j) = 1.0; }
			ke.Mult(h, load);
			for (int j = 0; j < dofs.Size(); ++j) { rhs(dofs[j]) -= load(j); }
		}
		MFEM_VERIFY(!lifted.empty(), "No element lies downstream of the cut; "
			"check the cut normal.");
	}

	// Solve K v = rhs for the DOFs not in @p fixed, which keep their values
	// in v. Only the conductor's free DOFs enter the solve: pinning the (many)
	// DOFs outside it as identity rows would leave them unconnected, which
	// smoothed-aggregation AMG turns into empty aggregates.
	void SolveFree(const mfem::SparseMatrix& K, const mfem::Vector& rhs,
				   const std::set<int>& fixed) {
		std::vector<int> index(fes.GetVSize(), -1), free;
		for (int i = 0; i < fes.GetVSize(); ++i) {
			if (!fixed.count(i)) { index[i] = static_cast<int>(free.size()); free.push_back(i); }
		}
		const int n = static_cast<int>(free.size());
		mfem::SparseMatrix Kff(n, n);
		mfem::Vector b(n), x(n);
		for (int r = 0; r < n; ++r) {
			const int row = free[r];
			b(r) = rhs(row);
			for (int p = K.GetI()[row]; p < K.GetI()[row + 1]; ++p) {
				const int col = K.GetJ()[p];
				const double a = K.GetData()[p];
				if (index[col] >= 0) { Kff.Add(r, index[col], a); }
				else { b(r) -= a * v(col); }  // lift the fixed values
			}
		}
		Kff.Finalize();
		AmgPreconditioner amg(Kff);
		mfem::CGSolver cg;
		cg.SetOperator(Kff);
		cg.SetPreconditioner(amg);
		cg.SetRelTol(1e-12);
		cg.SetMaxIter(2000);
		cg.SetPrintLevel(0);
		x = 0.0;
		cg.Mult(b, x);
		MFEM_VERIFY(cg.GetConverged(), "The conduction potential did not converge.");
		for (int r = 0; r < n; ++r) { v(free[r]) = x(r); }
	}
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
			  const mfem::IntegrationPoint& ip) override {
		path.Eval(T, ip, V);
		if (conductivity) {
			V *= scale * conductivity->Eval(T, ip);
			return;
		}
		const double norm = V.Norml2();
		if (norm > 0.0) { V *= scale / norm; }
	}

private:
	const ConductorPath& path;
	mfem::Coefficient* conductivity;
	double scale;
};

/// Over a conductor: A_cs = integral |w| dV (@p conductivity == nullptr), or
/// the DC conductance G = integral sigma |w|^2 dV.
inline double ConductorPathIntegral(mfem::Mesh& mesh, const mfem::Array<int>& conductor,
									const ConductorPath& path, mfem::Coefficient* conductivity,
									int order) {
	double total = 0.0;
	mfem::Vector w;
	for (int e = 0; e < mesh.GetNE(); ++e) {
		const int a = mesh.GetAttribute(e);
		if (a < 1 || a > conductor.Size() || !conductor[a - 1]) continue;
		mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
		const mfem::IntegrationRule& ir = mfem::IntRules.Get(
			mesh.GetElementBaseGeometry(e), 2 * order + T->OrderW() + 2);
		for (int q = 0; q < ir.GetNPoints(); ++q) {
			const mfem::IntegrationPoint& ip = ir.IntPoint(q);
			T->SetIntPoint(&ip);
			path.Eval(*T, ip, w);
			const double value = conductivity
				? conductivity->Eval(*T, ip) * (w * w) : w.Norml2();
			total += ip.weight * T->Weight() * value;
		}
	}
	return total;
}

/**
 * @brief Current that crosses a conductor's surface, in and out: the sum
 *        over its outer faces of |integral J . n dA|.
 *
 * An imposed current density has to stay inside its conductor, J . n = 0 on
 * every face between the conductor and anything else, except where it is
 * meant to enter or leave (@p openings: the electrodes). A cut lies inside
 * the conductor, so it is never an outer face. Current that does leave has
 * nowhere to go, and the divergence-free projection would silently supply
 * it through the surrounding material instead.
 *
 * The net flux of each face is taken, not its pointwise value: on a faceted
 * surface a correct direction (an azimuthal one on a flat face approximating
 * a cylinder) crosses each face symmetrically in and out, with no net flux.
 *
 * @param conductor  Domain-attribute marker of the conductor.
 * @param openings   Boundary-attribute marker of faces current may cross.
 * @param J          The conductor's current density (e.g. for 1 A).
 */
inline double ConductorSurfaceLeakage(mfem::Mesh& mesh, const mfem::Array<int>& conductor,
									  const mfem::Array<int>& openings,
									  mfem::VectorCoefficient& J, int order) {
	auto inside = [&](int e) {
		if (e < 0) return false;
		const int a = mesh.GetAttribute(e);
		return a >= 1 && a <= conductor.Size() && conductor[a - 1] != 0;
	};
	std::set<int> open_faces;
	for (int be = 0; be < mesh.GetNBE(); ++be) {
		const int a = mesh.GetBdrAttribute(be);
		if (a >= 1 && a <= openings.Size() && openings[a - 1]) {
			open_faces.insert(mesh.GetBdrElementFaceIndex(be));
		}
	}

	double leakage = 0.0;
	mfem::Vector j, n(3);
	for (int f = 0; f < mesh.GetNumFaces(); ++f) {
		int e1, e2;
		mesh.GetFaceElements(f, &e1, &e2);
		const bool in1 = inside(e1), in2 = inside(e2);
		if (in1 == in2 || open_faces.count(f)) continue;

		mfem::FaceElementTransformations* T = mesh.GetFaceElementTransformations(f);
		mfem::ElementTransformation& Te = in1 ? *T->Elem1 : *T->Elem2;
		const mfem::IntegrationRule& ir = mfem::IntRules.Get(
			T->GetGeometryType(), 2 * order + T->OrderW() + 2);
		double flux = 0.0;
		for (int q = 0; q < ir.GetNPoints(); ++q) {
			const mfem::IntegrationPoint& ip = ir.IntPoint(q);
			T->SetAllIntPoints(&ip);
			J.Eval(j, Te, Te.GetIntPoint());
			mfem::CalcOrtho(T->Jacobian(), n);  // scaled by the area element
			flux += ip.weight * (j * n);
		}
		leakage += std::abs(flux);
	}
	return leakage;
}
