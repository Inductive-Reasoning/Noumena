// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// Free-space inductances of coaxial rings of round wire, computed without any
// finite elements: Maxwell's formula for coaxial filaments, integrated over
// the wire cross-sections with a prescribed current-density shape.

#pragma once

#include <cmath>
#include <functional>
#include <vector>

#include "core/constants.hpp"

namespace coaxial_rings {

/// A ring of round wire: wire centre (R, Z) in the meridian plane, wire
/// radius a.
struct Ring { double R, Z, a; };

/// Current-density shape over the wire cross-section, J(r) up to a factor.
using Shape = std::function<double(double r)>;
inline double Uniform(double) { return 1.0; }       // stranded, or DC of a thin wire
inline double DcMassive(double r) { return 1.0 / r; }  // DC conduction current of a ring

namespace detail {

// Complete elliptic integrals K(k), E(k) from the complementary modulus
// k' = sqrt(1 - k^2) by the arithmetic-geometric mean. Taking k' directly
// keeps them exact for nearly coincident filaments, where k^2 rounds to 1.
inline void Elliptic(double k_complement, double& K, double& E) {
	double a = 1.0, b = k_complement, sum = 0.0, power = 0.5;
	double c2 = 1.0 - k_complement * k_complement;  // k^2
	sum = 0.5 * c2;
	while (std::abs(a - b) > 1e-15 * a) {
		const double next_a = 0.5 * (a + b), c = 0.5 * (a - b);
		b = std::sqrt(a * b);
		a = next_a;
		power *= 2.0;
		sum += power * c * c;
	}
	K = Constants::TWO_PI / (4.0 * a);
	E = K * (1.0 - sum);
}

} // namespace detail

/// Maxwell: mutual inductance [H] of coaxial filaments of radii r1, r2 at
/// heights z1, z2.
inline double FilamentMutual(double r1, double z1, double r2, double z2) {
	const double dz2 = (z1 - z2) * (z1 - z2);
	const double far = (r1 + r2) * (r1 + r2) + dz2;
	const double k = std::sqrt(4.0 * r1 * r2 / far);
	const double k_complement = std::sqrt(((r1 - r2) * (r1 - r2) + dz2) / far);
	double K, E;
	detail::Elliptic(k_complement, K, E);
	return Constants::MU_0 * std::sqrt(r1 * r2) * ((2.0 / k - k) * K - 2.0 / k * E);
}

namespace detail {

// Gauss-Legendre nodes and weights on [0, 1].
inline void GaussLegendre(int n, std::vector<double>& x, std::vector<double>& w) {
	x.resize(n);
	w.resize(n);
	const double pi = 0.5 * Constants::TWO_PI;
	for (int i = 0; i < n; ++i) {
		double z = std::cos(pi * (i + 0.75) / (n + 0.5)), dp = 0.0;
		for (int iteration = 0; iteration < 100; ++iteration) {
			double p1 = 1.0, p2 = 0.0;
			for (int j = 1; j <= n; ++j) {
				const double p3 = p2;
				p2 = p1;
				p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
			}
			dp = n * (z * p1 - p2) / (z * z - 1.0);
			const double previous = z;
			z = previous - p1 / dp;
			if (std::abs(z - previous) < 1e-15) break;
		}
		x[i] = 0.5 * (1.0 - z);
		w[i] = 1.0 / ((1.0 - z * z) * dp * dp);
	}
}

// integral over the wire cross-section of f(r, z) dA, polar Gauss rule
// (n radial points, 2n angular).
template <class F>
double OverWire(const Ring& ring, int n, F&& f) {
	std::vector<double> x, w;
	GaussLegendre(n, x, w);
	const int nt = 2 * n;
	const double dt = Constants::TWO_PI / nt;
	double total = 0.0;
	for (int i = 0; i < n; ++i) {
		const double rho = ring.a * x[i];
		for (int j = 0; j < nt; ++j) {
			const double t = dt * (j + 0.5);
			total += w[i] * ring.a * rho * dt *
				f(ring.R + rho * std::cos(t), ring.Z + rho * std::sin(t));
		}
	}
	return total;
}

} // namespace detail

/// Mutual inductance [H] of two separate rings carrying currents of shapes
/// @p shape_a and @p shape_b: the filament formula averaged over both
/// cross-sections with weights J. The integrand is smooth, so the Gauss rule
/// converges fast.
inline double Mutual(const Ring& A, const Shape& shape_a, const Ring& B, const Shape& shape_b,
					 int n = 16) {
	const double IA = detail::OverWire(A, n, [&](double r, double) { return shape_a(r); });
	const double IB = detail::OverWire(B, n, [&](double r, double) { return shape_b(r); });
	const double total = detail::OverWire(A, n, [&](double r, double z) {
		return shape_a(r) * detail::OverWire(B, n, [&](double r2, double z2) {
			return shape_b(r2) * FilamentMutual(r, z, r2, z2); });
	});
	return total / (IA * IB);
}

/// Self inductance [H] of a ring carrying current of shape @p shape. The
/// filament kernel is log-singular where the two points coincide, so the
/// inner integral is taken in polar coordinates about the outer point, whose
/// area element rho drho dtheta cancels the singularity.
inline double Self(const Ring& ring, const Shape& shape, int n = 24) {
	std::vector<double> x, w;
	detail::GaussLegendre(n, x, w);
	const int nt = 4 * n;
	const double dt = Constants::TWO_PI / nt;
	const double I = detail::OverWire(ring, n, [&](double r, double) { return shape(r); });
	const double total = detail::OverWire(ring, n, [&](double r, double z) {
		const double px = r - ring.R, pz = z - ring.Z;
		double inner = 0.0;
		for (int j = 0; j < nt; ++j) {
			const double t = dt * (j + 0.5), c = std::cos(t), s = std::sin(t);
			// Distance from (r, z) to the wire's edge along direction t.
			const double b = px * c + pz * s;
			const double reach = -b + std::sqrt(b * b - (px * px + pz * pz - ring.a * ring.a));
			for (int i = 0; i < n; ++i) {
				const double rho = reach * x[i];
				const double r2 = r + rho * c, z2 = z + rho * s;
				inner += w[i] * reach * dt * rho * shape(r2) * FilamentMutual(r, z, r2, z2);
			}
		}
		return shape(r) * inner;
	});
	return total / (I * I);
}

/// Self inductance of a ring of round wire with uniform current to second
/// order in a/R (Rosa & Grover), mu0 R [(1 + a^2/8R^2) ln(8R/a) - 7/4 +
/// a^2/24R^2]: an independent check on Self().
inline double SelfUniformSeries(const Ring& ring) {
	const double q = ring.a / ring.R, l = std::log(8.0 / q);
	return Constants::MU_0 * ring.R * ((1.0 + q * q / 8.0) * l - 1.75 + q * q / 24.0);
}

/// DC conductance [S] of a ring of round wire: sigma * integral dA/(2 pi r)
/// = sigma (R - sqrt(R^2 - a^2)).
inline double DcConductance(const Ring& ring, double sigma) {
	return sigma * (ring.R - std::sqrt(ring.R * ring.R - ring.a * ring.a));
}

} // namespace coaxial_rings
