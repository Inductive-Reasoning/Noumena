// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include "conductor_path.hpp"

 AzimuthalPath::AzimuthalPath(const CurrentDirection& d, double extent)
	: extent(extent) {
	MFEM_VERIFY(extent > 0.0 && extent <= Constants::TWO_PI,
		"An azimuthal path's angular extent must lie in (0, 2 pi].");
	double norm = 0.0;
	for (int c = 0; c < 3; ++c) {
		origin[c] = d.Origin[c];
		axis[c] = d.Axis[c];
		norm += axis[c] * axis[c];
	}
	norm = std::sqrt(norm);
	MFEM_VERIFY(norm > 0.0, "Current direction axis must be a nonzero vector.");
	for (double& c : axis) { c /= norm; }
	// An orthonormal pair across the axis, for Angle().
	const int k = std::abs(axis[0]) < 0.9 ? 0 : 1;
	double t[3] = { 0.0, 0.0, 0.0 };
	t[k] = 1.0;
	const double along = t[0] * axis[0] + t[1] * axis[1] + t[2] * axis[2];
	double n2 = 0.0;
	for (int c = 0; c < 3; ++c) { e1[c] = t[c] - along * axis[c]; n2 += e1[c] * e1[c]; }
	for (double& c : e1) { c /= std::sqrt(n2); }
	e2[0] = axis[1] * e1[2] - axis[2] * e1[1];
	e2[1] = axis[2] * e1[0] - axis[0] * e1[2];
	e2[2] = axis[0] * e1[1] - axis[1] * e1[0];
}

double AzimuthalPath::Angle(const mfem::Vector& x) const {
	double d[3];
	Perpendicular(x, d);
	const double angle = std::atan2(d[0] * e2[0] + d[1] * e2[1] + d[2] * e2[2],
									d[0] * e1[0] + d[1] * e1[1] + d[2] * e1[2]);
	return angle < 0.0 ? angle + Constants::TWO_PI : angle;
}

void AzimuthalPath::Tangent(const mfem::Vector& x, mfem::Vector& t) const {
	double d[3];
	const double r = Perpendicular(x, d);
	t.SetSize(3);
	t(0) = (axis[1] * d[2] - axis[2] * d[1]) / r;
	t(1) = (axis[2] * d[0] - axis[0] * d[2]) / r;
	t(2) = (axis[0] * d[1] - axis[1] * d[0]) / r;
}

void AzimuthalPath::Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip,
		  mfem::Vector& w) const  {
	mfem::Vector x;
	T.Transform(ip, x);
	double d[3];
	const double r = Perpendicular(x, d);
	MFEM_VERIFY(r > 0.0, "The azimuthal current direction is undefined on its axis.");
	// phi-hat / (extent r) = (e x d) / (extent r^2)
	const double s = 1.0 / (extent * r * r);
	w.SetSize(3);
	w(0) = s * (axis[1] * d[2] - axis[2] * d[1]);
	w(1) = s * (axis[2] * d[0] - axis[0] * d[2]);
	w(2) = s * (axis[0] * d[1] - axis[1] * d[0]);
}

double AzimuthalPath::Perpendicular(const mfem::Vector& x, double* d) const {
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

ConductionPath::ConductionPath(mfem::Mesh& mesh, int order, const mfem::Array<int>& conductor,
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

void ConductionPath::Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip,
		  mfem::Vector& w) const  {
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

void ConductionPath::BuildCutLifting(mfem::Mesh& mesh, const std::vector<bool>& inside,
					 const std::vector<int>& cut_group_faces,
					 const std::array<double, 3>& normal, mfem::Coefficient& sigma,
					 mfem::Vector& rhs) {
	mfem::Vector direction(3);
	for (int k = 0; k < 3; ++k) { direction(k) = normal[k]; }
	MFEM_VERIFY(direction.Norml2() > 0.0, "The cut normal must be a nonzero vector.");
	direction /= direction.Norml2();

	// The cut proper: faces of the cut group with the conductor on both
	// sides. A cut surface drawn larger than the conductor also has faces
	// on its surface and in the surroundings, which cut nothing.
	std::set<int> cut;
	for (int f : cut_group_faces) {
		int e1, e2;
		mesh.GetFaceElements(f, &e1, &e2);
		if (e1 >= 0 && e2 >= 0 && inside[e1] && inside[e2]) cut.insert(f);
	}
	MFEM_VERIFY(!cut.empty(), "The cut does not cross its conductor.");
	const std::vector<int> cut_faces(cut.begin(), cut.end());
	RequireCutSpansConductor(mesh, inside, cut_faces);

	std::set<int> cut_dofs;
	mfem::Array<int> dofs;
	for (int f : cut_faces) {
		fes.GetFaceDofs(f, dofs);
		cut_dofs.insert(dofs.begin(), dofs.end());
	}

	// The elements near the cut and the cut DOFs each one carries.
	std::map<int, std::vector<int>> near;
	for (int e = 0; e < mesh.GetNE(); ++e) {
		if (!inside[e]) continue;
		fes.GetElementDofs(e, dofs);
		std::vector<int> local;
		for (int j = 0; j < dofs.Size(); ++j) {
			if (cut_dofs.count(dofs[j])) local.push_back(j);
		}
		if (!local.empty()) near[e] = std::move(local);
	}

	// +1 downstream, -1 upstream, 0 not yet known.
	std::map<int, int> side;
	for (const auto& entry : near) { side[entry.first] = 0; }
	auto centroid = [&](int e) {
		mfem::Vector c;
		mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
		T->Transform(mfem::Geometries.GetCenter(T->GetGeometryType()), c);
		return c;
	};
	// Unit normal of cut face f, pointing from element e1 into e2.
	auto oriented_normal = [&](int f, int e1, int e2) {
		mfem::ElementTransformation* T = mesh.GetFaceTransformation(f);
		const mfem::IntegrationPoint& center = mfem::Geometries.GetCenter(T->GetGeometryType());
		T->SetIntPoint(&center);
		mfem::Vector n(3);
		mfem::CalcOrtho(T->Jacobian(), n);
		mfem::Vector across = centroid(e2);
		across -= centroid(e1);
		if (n * across < 0.0) n.Neg();
		return n;  // scaled by the face's area element
	};

	// Seed: the cut face the normal crosses most squarely.
	int seed_face = -1;
	double best = -1.0;
	for (int f : cut_faces) {
		int e1, e2;
		mesh.GetFaceElements(f, &e1, &e2);
		const mfem::Vector n = oriented_normal(f, e1, e2);
		const double alignment = std::abs(n * direction) / n.Norml2();
		if (alignment > best) { best = alignment; seed_face = f; }
	}
	MFEM_VERIFY(best >= kMinCutCrossing,
		"The cut normal (" << normal[0] << ", " << normal[1] << ", " << normal[2]
		<< ") runs along the cut instead of crossing it: it is at least "
		<< std::acos(best) * 360.0 / Constants::TWO_PI << " degrees from every cut "
		"face's normal, beyond the 60 allowed. Give the direction the current "
		"crosses the cut in; it must cross the cut.");
	{
		int e1, e2;
		mesh.GetFaceElements(seed_face, &e1, &e2);
		const bool forward = oriented_normal(seed_face, e1, e2) * direction > 0.0;
		side[e1] = forward ? -1 : 1;
	}

	// Propagate through faces between near elements.
	std::vector<int> queue;
	for (const auto& [e, s] : side) { if (s != 0) queue.push_back(e); }
	mfem::Array<int> faces, orientation;
	while (!queue.empty()) {
		const int e = queue.back();
		queue.pop_back();
		mesh.GetElementFaces(e, faces, orientation);
		for (int f : faces) {
			int e1, e2;
			mesh.GetFaceElements(f, &e1, &e2);
			const int other = e1 == e ? e2 : e1;
			const auto it = side.find(other);
			if (other < 0 || it == side.end()) continue;
			const int expected = cut.count(f) ? -side[e] : side[e];
			if (it->second == 0) {
				it->second = expected;
				queue.push_back(other);
			}
			MFEM_VERIFY(it->second == expected, "The cut is not a two-sided sheet: "
				"going around it reaches the same element from both sides. It must "
				"cross the conductor once, without folding through itself.");
		}
	}
	for (const auto& [e, s] : side) {
		MFEM_VERIFY(s != 0, "The cut has separate pieces; it must be a single "
			"connected surface crossing its conductor once.");
	}

	// The normal must agree with the cut's orientation on balance, not
	// just at the seed.
	double agreement = 0.0;
	for (int f : cut_faces) {
		int e1, e2;
		mesh.GetFaceElements(f, &e1, &e2);
		const int up = side[e1] < 0 ? e1 : e2, down = up == e1 ? e2 : e1;
		const mfem::Vector n = oriented_normal(f, up, down);
		agreement += n * direction;  // area-weighted: n carries the area element
	}
	MFEM_VERIFY(agreement > 0.0, "The cut normal disagrees with most of the cut: "
		"on balance it points against the direction the cut is crossed in.");

	mfem::DiffusionIntegrator diffusion(sigma);
	mfem::DenseMatrix ke;
	for (const auto& [e, local] : near) {
		if (side[e] < 0) continue;
		lifted[e] = local;
		fes.GetElementDofs(e, dofs);
		diffusion.AssembleElementMatrix(*fes.GetFE(e), *mesh.GetElementTransformation(e), ke);
		mfem::Vector h(dofs.Size()), load(dofs.Size());
		h = 0.0;
		for (int j : local) { h(j) = 1.0; }
		ke.Mult(h, load);
		for (int j = 0; j < dofs.Size(); ++j) { rhs(dofs[j]) -= load(j); }
	}
}

 void ConductionPath::RequireCutSpansConductor(mfem::Mesh& mesh, const std::vector<bool>& inside,
									 const std::vector<int>& cut_faces) {
	mfem::Array<int> edges, orientation;
	std::map<int, int> cut_edge_count;
	for (int f : cut_faces) {
		mesh.GetFaceEdges(f, edges, orientation);
		for (int e : edges) { ++cut_edge_count[e]; }
	}
	std::set<int> surface_edges;
	for (int f = 0; f < mesh.GetNumFaces(); ++f) {
		int e1, e2;
		mesh.GetFaceElements(f, &e1, &e2);
		const bool in1 = e1 >= 0 && inside[e1], in2 = e2 >= 0 && inside[e2];
		if (in1 == in2) continue;
		mesh.GetFaceEdges(f, edges, orientation);
		surface_edges.insert(edges.begin(), edges.end());
	}
	int interior_rim = 0;
	for (const auto& [edge, count] : cut_edge_count) {
		if (count == 1 && !surface_edges.count(edge)) ++interior_rim;
	}
	MFEM_VERIFY(interior_rim == 0,
		"The cut does not span its conductor's cross-section: " << interior_rim
		<< " edges of its rim lie inside the conductor instead of on its surface. "
		"The cut must sever the conductor completely.");
}

void ConductionPath::SolveFree(const mfem::SparseMatrix& K, const mfem::Vector& rhs,
			   const std::set<int>& fixed) {
	std::vector<int> free;
	for (int i = 0; i < fes.GetVSize(); ++i) {
		if (!fixed.count(i)) free.push_back(i);
	}
	SolveOnSubset(K, rhs, free, v, 1e-12, "The conduction potential");
}

void ConductorCurrentCoefficient::Eval(mfem::Vector& V, mfem::ElementTransformation& T,
		  const mfem::IntegrationPoint& ip)  {
	path.Eval(T, ip, V);
	if (conductivity) {
		V *= scale * conductivity->Eval(T, ip);
		return;
	}
	const double norm = V.Norml2();
	if (norm > 0.0) { V *= scale / norm; }
}

 double ConductorPathIntegral(mfem::Mesh& mesh, const mfem::Array<int>& conductor,
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

 double ConductorCurrentNormSquared(mfem::Mesh& mesh, const mfem::Array<int>& conductor,
										  mfem::VectorCoefficient& J, int order) {
	double total = 0.0;
	mfem::Vector j;
	for (int e = 0; e < mesh.GetNE(); ++e) {
		const int a = mesh.GetAttribute(e);
		if (a < 1 || a > conductor.Size() || !conductor[a - 1]) continue;
		mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
		const mfem::IntegrationRule& ir = mfem::IntRules.Get(
			mesh.GetElementBaseGeometry(e), 2 * order + T->OrderW() + 2);
		for (int q = 0; q < ir.GetNPoints(); ++q) {
			const mfem::IntegrationPoint& ip = ir.IntPoint(q);
			T->SetIntPoint(&ip);
			J.Eval(j, *T, ip);
			total += ip.weight * T->Weight() * (j * j);
		}
	}
	return total;
}
