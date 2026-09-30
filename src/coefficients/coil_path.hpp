// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "mfem.hpp"
#include "../core/constants.hpp"
#include "../core/problem_config.hpp"
#include "../linalg/amg_preconditioner.hpp"

/**
 * @brief Where current flows in a 3D stranded coil.
 *
 * Every coil type is described the same way: by w = -grad v, with v a unit
 * "conduction potential" of the coil. The coil then carries
 *     J = (I / A_cs) w / |w|,     A_cs = integral over the coil of |w| dV,
 * i.e. uniform current density |J| = I / A_cs along the direction of w, which
 * is what a coil of many fine strands of equal cross-section carries. A_cs is
 * the coil's cross-section: for a divergence-free J, the current through any
 * cross-section equals the integral of J . w over the coil, which is
 * (I / A_cs) * integral |w| = I.
 *
 *   Azimuthal  - v = -theta / (2 pi) about an axis: w = phi-hat / (2 pi r),
 *                analytic, so A_cs = integral dV / (2 pi r) is the meridional
 *                area of a coil of revolution.
 *   Electrodes - v = 1 on the input faces, 0 on the output faces, harmonic in
 *                the coil (an open coil; current enters and leaves there).
 *   Cut        - a closed coil: v jumps by 1 across an internal cut surface.
 *                Imposed as v = u + H with u continuous and H the "thick cut"
 *                lifting: on each coil element touching the cut on its
 *                downstream (+normal) side, H is the sum of the H1 basis
 *                functions of the cut DOFs, and 0 elsewhere. H is continuous
 *                everywhere except across the cut, where it jumps by 1.
 *
 * The conduction potential is solved on the parent mesh's H1 space with unit
 * conductivity in the coil, zero outside, and every DOF outside the coil
 * fixed, so no submesh (or element map) is needed.
 */
class CoilPath {
public:
	virtual ~CoilPath() = default;
	/// w = -grad v at @p ip of the coil element in @p T.
	virtual void Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip,
					  mfem::Vector& w) const = 0;
};

/// Analytic azimuthal path about an axis (right-hand rule).
class AzimuthalPath : public CoilPath {
public:
	explicit AzimuthalPath(const CoilDirection& d) {
		double norm = 0.0;
		for (int c = 0; c < 3; ++c) {
			origin[c] = d.Origin[c];
			axis[c] = d.Axis[c];
			norm += axis[c] * axis[c];
		}
		norm = std::sqrt(norm);
		MFEM_VERIFY(norm > 0.0, "Coil axis must be a nonzero vector.");
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
		MFEM_VERIFY(r > 0.0, "Azimuthal coil direction is undefined on its axis.");
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
class ConductionPath : public CoilPath {
public:
	/// @param coil          Domain-attribute marker of the coil.
	/// @param input/output  Boundary-attribute markers of the electrodes
	///                      (Electrodes); @p cut marks the cut (Cut).
	ConductionPath(mfem::Mesh& mesh, int order, const mfem::Array<int>& coil,
				   const CoilDirection& d, const mfem::Array<int>& input,
				   const mfem::Array<int>& output, const mfem::Array<int>& cut)
		: fec(order, mesh.Dimension()), fes(&mesh, &fec), v(&fes) {
		std::vector<bool> in_coil(mesh.GetNE(), false);
		std::set<int> coil_dofs;
		mfem::Array<int> dofs;
		for (int e = 0; e < mesh.GetNE(); ++e) {
			const int a = mesh.GetAttribute(e);
			if (a < 1 || a > coil.Size() || !coil[a - 1]) continue;
			in_coil[e] = true;
			fes.GetElementDofs(e, dofs);
			coil_dofs.insert(dofs.begin(), dofs.end());
		}

		// Faces of the given boundary attributes that border the coil.
		auto coil_faces = [&](const mfem::Array<int>& marker) {
			std::vector<int> faces;
			for (int be = 0; be < mesh.GetNBE(); ++be) {
				const int a = mesh.GetBdrAttribute(be);
				if (a < 1 || a > marker.Size() || !marker[a - 1]) continue;
				const int f = mesh.GetBdrElementFaceIndex(be);
				int e1, e2;
				mesh.GetFaceElements(f, &e1, &e2);
				if ((e1 >= 0 && in_coil[e1]) || (e2 >= 0 && in_coil[e2])) faces.push_back(f);
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
			if (!coil_dofs.count(i)) fixed.insert(i);
		}

		mfem::ConstantCoefficient one(1.0);
		mfem::Array<int> coil_marker(coil);
		mfem::BilinearForm k(&fes);
		k.AddDomainIntegrator(new mfem::DiffusionIntegrator(one), coil_marker);
		k.Assemble();
		k.Finalize();
		mfem::LinearForm rhs(&fes);
		rhs.Assemble();
		rhs = 0.0;

		if (d.Type == CoilDirection::Kind::Electrodes) {
			const auto in = face_dofs(coil_faces(input));
			const auto out = face_dofs(coil_faces(output));
			MFEM_VERIFY(!in.empty() && !out.empty(),
				"Coil electrodes do not touch the coil.");
			for (int i : in) { v(i) = 1.0; fixed.insert(i); }
			for (int i : out) { fixed.insert(i); }
		}
		else {
			BuildCutLifting(mesh, in_coil, coil_faces(cut), d.Normal, rhs);
			fixed.insert(*coil_dofs.begin());  // u is defined up to a constant
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
	// Solve K v = rhs for the DOFs not in @p fixed, which keep their values
	// in v. Only the coil's free DOFs enter the solve: pinning the (many) DOFs
	// outside the coil as identity rows would leave them unconnected, which
	// smoothed-aggregation AMG turns into empty aggregates.
	void SolveFree(const mfem::SparseMatrix& K, const mfem::LinearForm& rhs,
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
			b(r) = rhs[row];
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
		MFEM_VERIFY(cg.GetConverged(), "Coil conduction potential did not converge.");
		for (int r = 0; r < n; ++r) { v(free[r]) = x(r); }
	}

	mfem::H1_FECollection fec;
	mfem::FiniteElementSpace fes;
	mfem::GridFunction v;                       // u (cut) or v (electrodes)
	std::map<int, std::vector<int>> lifted;     // element -> local cut DOFs (+ side)

	// The thick-cut lifting H and its load -integral(grad H . grad w) over
	// the downstream elements.
	void BuildCutLifting(mfem::Mesh& mesh, const std::vector<bool>& in_coil,
						 const std::vector<int>& cut_faces,
						 const std::array<double, 3>& normal, mfem::LinearForm& rhs) {
		MFEM_VERIFY(!cut_faces.empty(), "Coil cut does not cross the coil.");
		std::set<int> cut_dofs;
		std::vector<mfem::Vector> centers;
		mfem::Array<int> dofs;
		for (int f : cut_faces) {
			fes.GetFaceDofs(f, dofs);
			cut_dofs.insert(dofs.begin(), dofs.end());
			mfem::Vector c;
			mfem::ElementTransformation* T = mesh.GetFaceTransformation(f);
			T->Transform(mfem::Geometries.GetCenter(T->GetGeometryType()), c);
			centers.push_back(c);
		}

		mfem::ConstantCoefficient one(1.0);
		mfem::DiffusionIntegrator diffusion(one);
		mfem::DenseMatrix ke;
		for (int e = 0; e < mesh.GetNE(); ++e) {
			if (!in_coil[e]) continue;
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
			for (int j = 0; j < dofs.Size(); ++j) { rhs[dofs[j]] -= load(j); }
		}
		MFEM_VERIFY(!lifted.empty(), "Coil cut: no element lies downstream of it; "
			"check the cut normal.");
	}
};

/// J = scale * w / |w|: uniform-magnitude current along the coil path.
class CoilCurrentCoefficient : public mfem::VectorCoefficient {
public:
	CoilCurrentCoefficient(const CoilPath& path, double scale)
		: mfem::VectorCoefficient(3), path(path), scale(scale) {}

	void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
			  const mfem::IntegrationPoint& ip) override {
		path.Eval(T, ip, V);
		const double norm = V.Norml2();
		if (norm > 0.0) { V *= scale / norm; }
	}

private:
	const CoilPath& path;
	double scale;
};

/// A_cs = integral over the coil of |w| dV (see CoilPath).
inline double CoilCrossSection(mfem::Mesh& mesh, const mfem::Array<int>& coil,
							   const CoilPath& path, int order) {
	double area = 0.0;
	mfem::Vector w;
	for (int e = 0; e < mesh.GetNE(); ++e) {
		const int a = mesh.GetAttribute(e);
		if (a < 1 || a > coil.Size() || !coil[a - 1]) continue;
		mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
		const mfem::IntegrationRule& ir = mfem::IntRules.Get(
			mesh.GetElementBaseGeometry(e), 2 * order + T->OrderW() + 2);
		for (int q = 0; q < ir.GetNPoints(); ++q) {
			const mfem::IntegrationPoint& ip = ir.IntPoint(q);
			T->SetIntPoint(&ip);
			path.Eval(*T, ip, w);
			area += ip.weight * T->Weight() * w.Norml2();
		}
	}
	return area;
}
