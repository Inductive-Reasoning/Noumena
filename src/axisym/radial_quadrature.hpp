// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

#include "mfem.hpp"
#include "axis_geometry.hpp"

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
 * An element meeting the axis has its contact factored out of the radius,
 * r = lambda q, with lambda the reference distance from the axis vertex or
 * edge and q smooth and positive. Along an axis edge the retained basis
 * functions vanish and cancel lambda (essential A_phi = 0 eliminates the rest,
 * whose integrals diverge); at an isolated axis vertex a rule collapsed onto
 * the vertex has a Jacobian proportional to lambda. Either way a polynomial
 * times 1/q remains, and the estimate above applied to q instead of r sets the
 * added order. An element clear of the axis is the case lambda = 1. q is
 * sampled, not bounded, so on curved maps the estimate is a heuristic. A
 * massive conductor must remain separated from the axis.
 */
namespace axisym {

// Relative accuracy targeted for the 1/r part.
constexpr double kRadialQuadratureTolerance = 1.0e-10;

// Ceiling on the order added for the 1/r part. It binds only where the ratio
// returned by RadialResolution is below about 1% (kResolvedRadiusRatio); the
// solvers warn about those elements.
constexpr int kMaxRadialExtraOrder = 120;
constexpr double kResolvedRadiusRatio = 1.0e-2;

// Highest triangle order MFEM 4.10 tabulates with positive weights and
// interior points; above it IntRules.Get falls back to Grundmann-Moller rules
// whose weights alternate in sign (the first negative weight appears at order
// 26), which the 1/r factor would turn into catastrophic cancellation.
// Recheck when MFEM is upgraded.
constexpr int kMaxTabulatedTriangleOrder = 25;

// The one tabulated triangle order up to kMaxTabulatedTriangleOrder with a
// point on an edge, which lands on r = 0 when that edge is the axis.
constexpr int kEdgePointTriangleOrder = 16;

/// Order added for a factor 1/q whose values span [q_min, q_min + width],
/// given ratio = q_min / width (infinite for constant q).
inline int RadialExtraOrder(double ratio) {
	if (!(ratio > 0.0)) { return kMaxRadialExtraOrder; }
	const double x0 = 1.0 + 2.0 * ratio;
	const double rho = x0 + std::sqrt(x0 * x0 - 1.0);
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
		*mfem::GlobGeometryRefiner.Refine(map.GetGeometryType(), std::max(4, 2 * T.Order()));
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

/// Where an element meets the axis, judged at its vertices: an isolated axis
/// vertex, or the edge (vertex, vertex + 1). Any other contact stays None,
/// where the near-zero radius drives the added order to its ceiling.
struct AxisContact {
	enum class Kind { None, Vertex, Edge } kind = Kind::None;
	int vertex = -1;
};

inline AxisContact FindAxisContact(const mfem::ElementTransformation& T,
								   mfem::real_t axis_tolerance) {
	auto& map = const_cast<mfem::ElementTransformation&>(T);
	const mfem::Geometry::Type geometry = map.GetGeometryType();
	const int vertices = mfem::Geometry::NumVerts[geometry];
	mfem::DenseMatrix x;
	map.Transform(*mfem::Geometries.GetVertices(geometry), x);
	unsigned on_axis = 0;
	int count = 0;
	for (int v = 0; v < vertices; ++v) {
		if (IsOnAxisGeometry(x(0, v), axis_tolerance)) {
			on_axis |= 1u << v;
			++count;
		}
	}
	for (int v = 0; v < vertices; ++v) {
		const bool here = on_axis & (1u << v);
		const bool next = on_axis & (1u << ((v + 1) % vertices));
		if (count == 1 && here) { return { AxisContact::Kind::Vertex, v }; }
		if (count == 2 && here && next) { return { AxisContact::Kind::Edge, v }; }
	}
	return {};
}

/// cross(b - a, p - a) in reference coordinates.
inline double ReferenceCross(const mfem::IntegrationPoint& a, const mfem::IntegrationPoint& b,
							 const mfem::IntegrationPoint& p) {
	return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

/// Reference distance lambda of @p p from the contact, 1 on the far side of
/// the element, so that r / lambda is smooth and positive.
inline double ContactDistance(mfem::Geometry::Type geometry, const AxisContact& contact,
							  const mfem::IntegrationPoint& p) {
	if (contact.kind == AxisContact::Kind::None) { return 1.0; }
	const mfem::IntegrationRule& corners = *mfem::Geometries.GetVertices(geometry);
	const int n = corners.GetNPoints();
	const mfem::IntegrationPoint& o = corners.IntPoint(contact.vertex);
	if (contact.kind == AxisContact::Kind::Edge) {
		const mfem::IntegrationPoint& a = corners.IntPoint((contact.vertex + 1) % n);
		const mfem::IntegrationPoint& far = corners.IntPoint((contact.vertex + 2) % n);
		return ReferenceCross(o, a, p) / ReferenceCross(o, a, far);
	}
	// The collapsed coordinate s of CollapsedRule, over the fan from o.
	double lambda = 0.0;
	for (int k = 1; k + 1 < n; ++k) {
		const mfem::IntegrationPoint& a = corners.IntPoint((contact.vertex + k) % n);
		const mfem::IntegrationPoint& b = corners.IntPoint((contact.vertex + k + 1) % n);
		lambda = std::max(lambda, 1.0 - ReferenceCross(a, b, p) / ReferenceCross(a, b, o));
	}
	return lambda;
}

/// Ratio q_min / (q_max - q_min) of the smooth radial factor q = r / lambda,
/// which sets the order added for 1/q. Sampled on a lattice that omits the
/// contact itself, not bounded: extremes between lattice points are missed.
inline double RadialResolution(const mfem::ElementTransformation& T,
							   const AxisContact& contact) {
	auto& map = const_cast<mfem::ElementTransformation&>(T);
	const mfem::Geometry::Type geometry = map.GetGeometryType();
	const int divisions = std::max(4, 2 * T.Order());
	const mfem::RefinedGeometry& lattice = *mfem::GlobGeometryRefiner.Refine(geometry, divisions);
	mfem::DenseMatrix x;
	map.Transform(lattice.RefPts, x);
	double min_q = std::numeric_limits<double>::max();
	double max_q = std::numeric_limits<double>::lowest();
	for (int i = 0; i < x.Width(); ++i) {
		const double lambda = ContactDistance(geometry, contact, lattice.RefPts.IntPoint(i));
		// Lattice points off the contact have lambda >= 1 / divisions.
		if (lambda < 0.5 / divisions) { continue; }
		min_q = std::min(min_q, double(x(0, i)) / lambda);
		max_q = std::max(max_q, double(x(0, i)) / lambda);
	}
	return min_q / (max_q - min_q);
}

inline double RadialResolution(const mfem::ElementTransformation& T,
							   mfem::real_t axis_tolerance) {
	return RadialResolution(T, FindAxisContact(T, axis_tolerance));
}

/// Positive rule on @p geometry collapsed onto reference vertex @p vertex: the
/// fan of triangles (o, a, b) from the vertex, each mapped from the unit square
/// by (s, t) -> o + s ((1 - t) a + t b - o), whose Jacobian is proportional to
/// s. Exact for degree @p radial_order in s and @p angular_order in t.
inline const mfem::IntegrationRule& CollapsedRule(mfem::Geometry::Type geometry, int vertex,
												  int radial_order, int angular_order) {
	static std::mutex lock;
	static std::map<std::tuple<int, int, int, int>, std::unique_ptr<mfem::IntegrationRule>> rules;
	const std::lock_guard<std::mutex> guard(lock);
	auto& rule = rules[std::make_tuple(int(geometry), vertex, radial_order, angular_order)];
	if (rule) { return *rule; }
	// The Jacobian factor s raises the degree in s by one.
	const mfem::IntegrationRule& radial = mfem::IntRules.Get(mfem::Geometry::SEGMENT, radial_order + 1);
	const mfem::IntegrationRule& angular = mfem::IntRules.Get(mfem::Geometry::SEGMENT, angular_order);
	const mfem::IntegrationRule& corners = *mfem::Geometries.GetVertices(geometry);
	const int n = corners.GetNPoints();
	const mfem::IntegrationPoint& o = corners.IntPoint(vertex);
	rule = std::make_unique<mfem::IntegrationRule>(
		(n - 2) * radial.GetNPoints() * angular.GetNPoints());
	int index = 0;
	for (int k = 1; k + 1 < n; ++k) {
		const mfem::IntegrationPoint& a = corners.IntPoint((vertex + k) % n);
		const mfem::IntegrationPoint& b = corners.IntPoint((vertex + k + 1) % n);
		const double jacobian = std::abs(ReferenceCross(o, a, b));
		for (int i = 0; i < radial.GetNPoints(); ++i) {
			const mfem::IntegrationPoint& s = radial.IntPoint(i);
			for (int j = 0; j < angular.GetNPoints(); ++j) {
				const mfem::IntegrationPoint& t = angular.IntPoint(j);
				mfem::IntegrationPoint& ip = rule->IntPoint(index++);
				ip.Set2(o.x + s.x * ((1.0 - t.x) * a.x + t.x * b.x - o.x),
						o.y + s.x * ((1.0 - t.x) * a.y + t.x * b.y - o.y));
				ip.weight = s.weight * t.weight * s.x * jacobian;
			}
		}
	}
	rule->SetOrder(std::min(radial_order, angular_order));
	return *rule;
}

/// A rule of at least @p order on @p geometry whose weights are positive and
/// whose points are interior: MFEM's tables where they have that property,
/// otherwise on triangles a CollapsedRule.
inline const mfem::IntegrationRule& PositiveRule(mfem::Geometry::Type geometry, int order) {
	if (geometry == mfem::Geometry::TRIANGLE && order == kEdgePointTriangleOrder) { ++order; }
	if (geometry != mfem::Geometry::TRIANGLE || order <= kMaxTabulatedTriangleOrder) {
		return mfem::IntRules.Get(geometry, order);
	}
	return CollapsedRule(mfem::Geometry::TRIANGLE, 1, order, order);
}

/// The rule for an integrand whose polynomial part has order
/// @p polynomial_order and which carries one 1/r factor, on the element of
/// @p T: positive weights and interior points throughout.
inline const mfem::IntegrationRule& RadialRule(mfem::Geometry::Type geometry,
											   int polynomial_order,
											   const mfem::ElementTransformation& T,
											   mfem::real_t axis_tolerance) {
	const AxisContact contact = FindAxisContact(T, axis_tolerance);
	const int extra_order = RadialExtraOrder(RadialResolution(T, contact));
	if (contact.kind != AxisContact::Kind::Vertex) {
		return PositiveRule(geometry, polynomial_order + extra_order);
	}
	// Collapsing a square onto a corner doubles the degree in s of its
	// tensor-product polynomials.
	const int radial_order = (geometry == mfem::Geometry::SQUARE ? 2 : 1) * polynomial_order;
	return CollapsedRule(geometry, contact.vertex, radial_order + extra_order,
						 polynomial_order + extra_order);
}

} // namespace axisym
