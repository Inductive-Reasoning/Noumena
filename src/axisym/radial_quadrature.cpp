// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include "radial_quadrature.hpp"

namespace axisym {

 int RadialExtraOrder(double ratio) {
	if (!(ratio > 0.0)) { return kMaxRadialExtraOrder; }
	const double x0 = 1.0 + 2.0 * ratio;
	const double rho = x0 + std::sqrt(x0 * x0 - 1.0);
	const double order = std::log(1.0 / kRadialQuadratureTolerance) / std::log(rho);
	return static_cast<int>(std::ceil(std::min(order, double(kMaxRadialExtraOrder))));
}

 void RadialExtent(const mfem::ElementTransformation& T, double& min_radius,
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

 AxisContact FindAxisContact(const mfem::ElementTransformation& T,
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

 double ContactDistance(mfem::Geometry::Type geometry, const AxisContact& contact,
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

 double RadialResolution(const mfem::ElementTransformation& T,
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

 const mfem::IntegrationRule& CollapsedRule(mfem::Geometry::Type geometry, int vertex,
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

 const mfem::IntegrationRule& PositiveRule(mfem::Geometry::Type geometry, int order) {
	if (geometry == mfem::Geometry::TRIANGLE && order == kEdgePointTriangleOrder) { ++order; }
	if (geometry != mfem::Geometry::TRIANGLE || order <= kMaxTabulatedTriangleOrder) {
		return mfem::IntRules.Get(geometry, order);
	}
	return CollapsedRule(mfem::Geometry::TRIANGLE, 1, order, order);
}

 const mfem::IntegrationRule& RadialRule(mfem::Geometry::Type geometry,
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
