// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <mutex>

#include "mfem.hpp"

/**
 * @brief Quadrature for axisymmetric integrands with a 1/r factor.
 *
 * Several axisymmetric integrals are not polynomial on an element: the
 * curl-curl term N_j N_k / r, a massive conductor's DC conductance
 * sigma / (2 pi r), and the Joule loss of its drive field V / (2 pi r). On an
 * element spanning r in [a, a + h], mapping to the reference interval
 * x in [-1, 1] gives r = a + h(1 + x)/2, so 1/r has a pole at
 * x0 = -(1 + 2s), s = a/h. Gauss-Legendre applied to a function whose nearest
 * singularity lies at x0 converges geometrically like rho^-n, with
 *
 *     rho = |x0| + sqrt(x0^2 - 1)
 *
 * the Bernstein-ellipse parameter, so a relative accuracy eps needs a rule of
 * order about ln(1/eps)/ln(rho) on top of the integrand's polynomial part.
 * The difficulty is set by the element's geometry, not by the basis degree,
 * and grows without bound as the element nears the axis (s -> 0). Measured
 * convergence matches this estimate across s in [1e-3, 3].
 *
 * Elements touching the axis (s = 0) get no extra order: there the integral
 * of N_j N_k / r diverges for basis functions that do not vanish at r = 0,
 * and no finite rule converges. That divergence is a property of individual
 * basis functions, not of the solution; the essential A_phi = 0 condition on
 * the axis removes exactly the offending directions. A conductor cannot touch
 * the axis.
 */
namespace axisym {

// Relative accuracy targeted for the 1/r part.
constexpr double kRadialQuadratureTolerance = 1.0e-10;

// Ceiling on the order added for the 1/r part. It binds only for elements
// whose inner radius is below about 1% of their radial width
// (kResolvedRadiusRatio); the solvers warn about those.
constexpr int kMaxRadialExtraOrder = 120;
constexpr double kResolvedRadiusRatio = 1.0e-2;

// Highest triangle order MFEM 4.10 tabulates with positive weights and
// interior points; above it IntRules.Get falls back to Grundmann-Moller rules
// whose weights alternate in sign (the first negative weight appears at order
// 26), which the 1/r factor would turn into catastrophic cancellation.
// Recheck when MFEM is upgraded.
constexpr int kMaxTabulatedTriangleOrder = 25;

/// Order added for the 1/r part on an element whose radii span
/// [min_radius, min_radius + radial_width]; 0 on the axis.
inline int RadialExtraOrder(double min_radius, double radial_width) {
	if (!(radial_width > 0.0) || !(min_radius > 0.0)) { return 0; }
	const double s = min_radius / radial_width;
	const double x0 = 1.0 + 2.0 * s;
	const double rho = x0 + std::sqrt(x0 * x0 - 1.0);
	if (!(rho > 1.0)) { return kMaxRadialExtraOrder; }
	const double order = std::log(1.0 / kRadialQuadratureTolerance) / std::log(rho);
	return static_cast<int>(std::ceil(std::min(order, double(kMaxRadialExtraOrder))));
}

/// Smallest radius and radial width of an element, from its map sampled on a
/// lattice (vertices included), so a curved edge bowing toward the axis is
/// seen. Sampled, not bounded: an extreme between lattice points is missed.
/// (MFEM's integration-rule hooks pass the transformation as const, but
/// evaluating it updates its cached point, hence the cast.)
inline void RadialExtent(const mfem::ElementTransformation& T, double& min_radius,
						 double& radial_width) {
	auto& map = const_cast<mfem::ElementTransformation&>(T);
	const mfem::RefinedGeometry& lattice =
		*mfem::GlobGeometryRefiner.Refine(map.GetGeometryType(), 4);
	mfem::DenseMatrix x;
	map.Transform(lattice.RefPts, x);
	double min_r = std::numeric_limits<double>::max();
	double max_r = std::numeric_limits<double>::lowest();
	for (int i = 0; i < x.Width(); ++i) {
		min_r = std::min(min_r, double(x(0, i)));
		max_r = std::max(max_r, double(x(0, i)));
	}
	min_radius = min_r;
	radial_width = max_r - min_r;
}

/// A rule of at least @p order on @p geometry whose weights are positive and
/// whose points are interior: MFEM's tables where they have that property,
/// and above kMaxTabulatedTriangleOrder on triangles a collapsed (Duffy)
/// Gauss rule, x = u, y = (1 - u) v on the unit square, weight (1 - u).
inline const mfem::IntegrationRule& PositiveRule(mfem::Geometry::Type geometry, int order) {
	if (geometry != mfem::Geometry::TRIANGLE || order <= kMaxTabulatedTriangleOrder) {
		return mfem::IntRules.Get(geometry, order);
	}
	static std::mutex lock;
	static std::map<int, std::unique_ptr<mfem::IntegrationRule>> collapsed;
	const std::lock_guard<std::mutex> guard(lock);
	auto& rule = collapsed[order];
	if (!rule) {
		// The Jacobian (1 - u) raises the degree in u by one.
		const mfem::IntegrationRule& line = mfem::IntRules.Get(mfem::Geometry::SEGMENT, order + 1);
		const int n = line.GetNPoints();
		rule = std::make_unique<mfem::IntegrationRule>(n * n);
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				const mfem::IntegrationPoint& u = line.IntPoint(i);
				const mfem::IntegrationPoint& v = line.IntPoint(j);
				mfem::IntegrationPoint& ip = rule->IntPoint(i * n + j);
				ip.Set2(u.x, (1.0 - u.x) * v.x);
				ip.weight = u.weight * v.weight * (1.0 - u.x);
			}
		}
		rule->SetOrder(order);
	}
	return *rule;
}

/// The rule for an integrand whose polynomial part has order
/// @p polynomial_order and which carries one 1/r factor, on the element of
/// @p T: positive weights and interior points throughout.
inline const mfem::IntegrationRule& RadialRule(mfem::Geometry::Type geometry,
											   int polynomial_order,
											   const mfem::ElementTransformation& T) {
	double min_radius = 0.0, radial_width = 0.0;
	RadialExtent(T, min_radius, radial_width);
	return PositiveRule(geometry, polynomial_order + RadialExtraOrder(min_radius, radial_width));
}

} // namespace axisym
