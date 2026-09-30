// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>

#include "mfem.hpp"
#include "../core/constants.hpp"
#include "../core/problem_config.hpp"

/**
 * @brief Azimuthal unit direction phi-hat about an arbitrary axis, times a
 *        constant scale: J(x) = scale * phi-hat(x).
 *
 * phi-hat = (e x d_perp) / |d_perp|, where e is the unit axis and d_perp the
 * part of (x - origin) perpendicular to it. That is the right-hand-rule sense:
 * a positive scale is current that produces flux along +e inside the coil.
 *
 * With scale = I / A_cs (see CoilCrossSection) this is the uniform stranded
 * current density of a coil of revolution carrying total current I, the 3D
 * counterpart of the 2D I/area source.
 *
 * The direction is undefined on the axis itself; callers reject coils that
 * reach it (a coil of revolution cannot enclose its own axis anyway).
 */
class AzimuthalCoilCoefficient : public mfem::VectorCoefficient {
public:
	AzimuthalCoilCoefficient(const CoilDirection& direction, double scale)
		: mfem::VectorCoefficient(3), scale_(scale) {
		double norm = 0.0;
		for (int c = 0; c < 3; ++c) {
			origin_[c] = direction.Origin[c];
			axis_[c] = direction.Axis[c];
			norm += axis_[c] * axis_[c];
		}
		norm = std::sqrt(norm);
		MFEM_VERIFY(norm > 0.0, "Coil axis must be a nonzero vector.");
		for (double& c : axis_) { c /= norm; }
	}

	/// Distance of @p x from the axis.
	double RadiusOf(const mfem::Vector& x) const {
		std::array<double, 3> d_perp;
		Perpendicular(x, d_perp);
		return std::sqrt(d_perp[0] * d_perp[0] + d_perp[1] * d_perp[1] + d_perp[2] * d_perp[2]);
	}

	void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
			  const mfem::IntegrationPoint& ip) override {
		mfem::Vector x;
		T.Transform(ip, x);
		std::array<double, 3> d;
		Perpendicular(x, d);
		const double r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
		MFEM_VERIFY(r > 0.0, "Azimuthal coil direction is undefined on its axis.");
		V.SetSize(3);
		V(0) = scale_ * (axis_[1] * d[2] - axis_[2] * d[1]) / r;
		V(1) = scale_ * (axis_[2] * d[0] - axis_[0] * d[2]) / r;
		V(2) = scale_ * (axis_[0] * d[1] - axis_[1] * d[0]) / r;
	}

private:
	std::array<double, 3> origin_{};
	std::array<double, 3> axis_{};  // unit
	double scale_;

	void Perpendicular(const mfem::Vector& x, std::array<double, 3>& d_perp) const {
		double along = 0.0;
		for (int c = 0; c < 3; ++c) {
			d_perp[c] = x(c) - origin_[c];
			along += d_perp[c] * axis_[c];
		}
		for (int c = 0; c < 3; ++c) { d_perp[c] -= along * axis_[c]; }
	}
};

/**
 * @brief Effective cross-section of a coil of revolution:
 *        A_cs = integral over the coil of dV / (2 pi r).
 *
 * For a coil of revolution this is exactly the area of its meridional cross-
 * section (Pappus), so I / A_cs is the same uniform current density the 2D
 * axisymmetric model uses. Also returns the smallest radius reached, so the
 * caller can reject a coil that touches the axis, where the integrand is
 * singular.
 */
inline double CoilCrossSection(mfem::Mesh& mesh, const mfem::Array<int>& attribute_marker,
							   const AzimuthalCoilCoefficient& coil, int order,
							   double& min_radius) {
	double area = 0.0;
	min_radius = std::numeric_limits<double>::max();
	mfem::Vector x;
	for (int e = 0; e < mesh.GetNE(); ++e) {
		const int attr = mesh.GetAttribute(e);
		if (attr < 1 || attr > attribute_marker.Size() || !attribute_marker[attr - 1]) continue;
		// Vertices too, so a coil whose corner sits on the axis is caught even
		// though no quadrature point lands there.
		mfem::Array<int> vertices;
		mesh.GetElementVertices(e, vertices);
		for (int v : vertices) {
			const double* p = mesh.GetVertex(v);
			mfem::Vector pv(const_cast<double*>(p), mesh.SpaceDimension());
			min_radius = std::min(min_radius, coil.RadiusOf(pv));
		}
		mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
		const mfem::IntegrationRule& ir = mfem::IntRules.Get(
			mesh.GetElementBaseGeometry(e), 2 * order + T->OrderW() + 2);
		for (int q = 0; q < ir.GetNPoints(); ++q) {
			const mfem::IntegrationPoint& ip = ir.IntPoint(q);
			T->SetIntPoint(&ip);
			T->Transform(ip, x);
			const double r = coil.RadiusOf(x);
			min_radius = std::min(min_radius, r);
			if (r > 0.0) { area += ip.weight * T->Weight() / (Constants::TWO_PI * r); }
		}
	}
	return area;
}
