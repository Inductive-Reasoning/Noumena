// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT
//
// Bessel functions of complex argument for the closed-form eddy-current
// references, by their power series. The series converge for every z; for
// |z| up to about 10 (a conductor a few skin depths thick) they lose at most
// a few digits to cancellation, far below the tolerances they serve.
// test_closed_forms.cpp checks them against the standard library on the real
// axis and against the Wronskian off it.

#pragma once

#include <cmath>
#include <complex>

namespace bessel {

using complex = std::complex<double>;

constexpr double kEulerGamma = 0.57721566490153286;

namespace detail {

// sum_k s^k (z/2)^(2k + n) / (k! (k + n)!), s = -1 for J_n, +1 for I_n.
inline complex Series(int n, const complex& z, double s) {
	const complex q = 0.25 * z * z;
	complex term = std::pow(0.5 * z, n);
	for (int k = 1; k <= n; ++k) { term /= double(k); }
	complex sum = term;
	for (int k = 1; k < 300; ++k) {
		term *= s * q / (double(k) * double(k + n));
		sum += term;
		if (std::abs(term) <= 1e-17 * std::abs(sum)) { break; }
	}
	return sum;
}

} // namespace detail

inline complex J0(const complex& z) { return detail::Series(0, z, -1.0); }
inline complex J1(const complex& z) { return detail::Series(1, z, -1.0); }
inline complex I0(const complex& z) { return detail::Series(0, z, 1.0); }
inline complex I1(const complex& z) { return detail::Series(1, z, 1.0); }

// K0(z) = -(ln(z/2) + gamma) I0(z) + sum_{k>=1} H_k (z^2/4)^k / (k!)^2,
// H_k the k-th harmonic number. Principal branch, Re z > 0 here.
inline complex K0(const complex& z) {
	const complex q = 0.25 * z * z;
	complex term = 1.0, sum = 0.0;
	double harmonic = 0.0;
	for (int k = 1; k < 300; ++k) {
		term *= q / (double(k) * double(k));
		harmonic += 1.0 / k;
		sum += harmonic * term;
		if (std::abs(harmonic * term) <= 1e-17 * std::abs(sum)) { break; }
	}
	return -(std::log(0.5 * z) + kEulerGamma) * I0(z) + sum;
}

// K1(z) = 1/z + ln(z/2) I1(z)
//         - (z/4) sum_{k>=0} (psi(k+1) + psi(k+2)) (z^2/4)^k / (k! (k+1)!),
// with psi(k+1) = H_k - gamma.
inline complex K1(const complex& z) {
	const complex q = 0.25 * z * z;
	complex term = 1.0;                       // (z^2/4)^k / (k! (k+1)!)
	double psi = -kEulerGamma;                // psi(k + 1)
	complex sum = (psi + (psi + 1.0)) * term;
	for (int k = 1; k < 300; ++k) {
		term *= q / (double(k) * double(k + 1));
		psi += 1.0 / k;
		const complex add = (psi + (psi + 1.0 / (k + 1))) * term;
		sum += add;
		if (std::abs(add) <= 1e-17 * std::abs(sum)) { break; }
	}
	return 1.0 / z + std::log(0.5 * z) * I1(z) - 0.25 * z * sum;
}

} // namespace bessel
