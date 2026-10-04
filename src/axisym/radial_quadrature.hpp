#pragma once

#include <map>
#include <memory>
#include <cmath>
#include <stdexcept>
#include "mfem.hpp"

namespace axisym {

// Positive interior tensor Gauss rules; the triangle uses the Duffy map.
// No high-order simplex rule with negative weights is ever requested.
inline const mfem::IntegrationRule& PositiveRule(mfem::Geometry::Type geometry, int n) {
    using Key = std::pair<int, int>;
    static thread_local std::map<Key, std::unique_ptr<mfem::IntegrationRule>> cache;
    auto& result = cache[{static_cast<int>(geometry), n}];
    if (result) return *result;
    if (geometry != mfem::Geometry::TRIANGLE && geometry != mfem::Geometry::SQUARE)
        throw std::runtime_error("Radial quadrature requires triangles or quadrilaterals.");
    const auto& line = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * n - 1);
    result = std::make_unique<mfem::IntegrationRule>(line.GetNPoints() * line.GetNPoints());
    int k = 0;
    for (int i = 0; i < line.GetNPoints(); ++i) {
        for (int j = 0; j < line.GetNPoints(); ++j) {
            const auto& u = line.IntPoint(i);
            const auto& v = line.IntPoint(j);
            auto& ip = result->IntPoint(k++);
            const bool triangle = geometry == mfem::Geometry::TRIANGLE;
            ip.Set2(u.x, triangle ? (1 - u.x) * v.x : v.x);
            ip.weight = u.weight * v.weight * (triangle ? 1 - u.x : 1.0);
        }
    }
    result->SetOrder(2 * n - 2);
    return *result;
}

inline const mfem::IntegrationRule& RadialRule(const mfem::FiniteElement& fe,
    const mfem::ElementTransformation& transformation) {
    auto& t = const_cast<mfem::ElementTransformation&>(transformation);
    const auto geometry = fe.GetGeomType();
    const int degree = 2 * fe.GetOrder() + 2;
    const int base = std::max(4, fe.GetOrder() + t.Order() + 2);
    constexpr int cap = 512;
    constexpr double tolerance = 1e-11;
    bool touches_axis = false;
    mfem::Vector position;
    // These are samples, not bounds on a curved map. Every integration point
    // is checked too; convergence below tests transformed radial moments.
    const auto& vertices = *mfem::Geometries.GetVertices(geometry);
    for (int i = 0; i < vertices.GetNPoints(); ++i) {
        t.Transform(vertices.IntPoint(i), position);
        if (position(0) < 0) throw std::runtime_error("Negative radius in radial quadrature.");
        touches_axis = touches_axis || position(0) == 0.0;
    }
    auto moments = [&](const mfem::IntegrationRule& rule) {
        std::vector<double> values((degree + 1) * (degree + 1) * 3, 0.0);
        for (int q = 0; q < rule.GetNPoints(); ++q) {
            const auto& ip = rule.IntPoint(q);
            t.SetIntPoint(&ip);
            t.Transform(ip, position);
            const double r = position(0);
            if (!(r > 0) || !std::isfinite(r))
                throw std::runtime_error("Nonpositive interior radius in radial quadrature.");
            const double w = ip.weight * t.Weight();
            double xp = 1;
            for (int i = 0; i <= degree; ++i) {
                double monomial = xp;
                for (int j = 0; j <= degree - i; ++j) {
                    const int k = 3 * (i * (degree + 1) + j);
                    values[k] += w * monomial;
                    values[k + 1] += w * monomial * r;
                    // Singular eliminated axis basis functions have no finite
                    // integral. Test regular moments there instead.
                    values[k + 2] += w * monomial * (touches_axis ? r : 1 / r);
                    monomial *= ip.y;
                }
                xp *= ip.x;
            }
        }
        return values;
    };
    int n = base;
    auto previous = moments(PositiveRule(geometry, n));
    while (n < cap) {
        n = std::min(cap, 2 * n);
        const auto& rule = PositiveRule(geometry, n);
        auto current = moments(rule);
        bool converged = true;
        for (size_t k = 0; k < current.size(); ++k) {
            if (std::abs(current[k] - previous[k]) >
                tolerance * std::abs(current[k % 3])) converged = false;
        }
        if (converged) return rule;
        previous = std::move(current);
    }
    throw std::runtime_error("Radial quadrature did not converge at 512 Gauss points "
        "per direction; refine the near-axis or curved elements.");
}

// C integrates sigma v in the r-z cross-section, independently of M and G.
class PortLoadIntegrator : public mfem::DomainLFIntegrator {
public:
    explicit PortLoadIntegrator(mfem::Coefficient& coefficient)
        : mfem::DomainLFIntegrator(coefficient) {}
    void AssembleRHSElementVect(const mfem::FiniteElement& fe,
        mfem::ElementTransformation& t, mfem::Vector& result) override {
        SetIntRule(&RadialRule(fe, t));
        mfem::DomainLFIntegrator::AssembleRHSElementVect(fe, t, result);
    }
};
}
