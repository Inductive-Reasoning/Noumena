// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <vector>

#include "mfem.hpp"
#include "conductor_path.hpp"

/**
 * @brief Time-averaged Joule loss density P = 1/2 sigma |E|^2 of a 3D
 *        time-harmonic solution, E = V w - j omega A.
 *
 * The 3D counterpart of MqsLossDensityCoefficient, with the same peak-phasor
 * convention (see there). A massive conductor with port voltage V is driven
 * by the field V w of its ConductorPath; every other conductor -- a passive
 * shield, a brace -- has only the induced field -j omega A. A is the Nedelec
 * potential of the formulation, whose gradient part in the conductors carries
 * their charge-free electric scalar potential, so E is complete.
 *
 * With A = A_re + j A_im and V = V_re + j V_im,
 *     Re E = V_re w + omega A_im,    Im E = V_im w - omega A_re.
 *
 * sigma, the potentials and the paths are referenced, not owned.
 */
class MqsVectorLossDensityCoefficient : public mfem::Coefficient {
public:
	/// A massive conductor's drive: its path and solved port voltage.
	struct Drive {
		const ConductorPath* Path = nullptr;
		std::complex<double> Voltage;
	};

	/// @param drive_of_attribute  Index into @p drives by (attribute - 1), or
	///                            -1 where no port drives the conductor.
	MqsVectorLossDensityCoefficient(mfem::Coefficient& sigma,
									const mfem::GridFunction& a_re,
									const mfem::GridFunction& a_im, double omega,
									std::vector<Drive> drives,
									std::vector<int> drive_of_attribute)
		: sigma(sigma), a_re(a_re), a_im(a_im), omega(omega),
		  drives(std::move(drives)), drive_of_attribute(std::move(drive_of_attribute)) {}

	double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
		T.SetIntPoint(&ip);
		const double s = sigma.Eval(T, ip);
		if (s <= 0.0) { return 0.0; }

		mfem::Vector e_re, e_im;
		a_im.GetVectorValue(T, ip, e_re);
		a_re.GetVectorValue(T, ip, e_im);
		e_re *= omega;
		e_im *= -omega;

		const int index = T.Attribute - 1;
		if (index >= 0 && index < static_cast<int>(drive_of_attribute.size()) &&
			drive_of_attribute[index] >= 0) {
			const Drive& drive = drives[drive_of_attribute[index]];
			mfem::Vector w;
			drive.Path->Eval(T, ip, w);
			e_re.Add(drive.Voltage.real(), w);
			e_im.Add(drive.Voltage.imag(), w);
		}
		return 0.5 * s * (e_re * e_re + e_im * e_im);
	}

private:
	mfem::Coefficient& sigma;
	const mfem::GridFunction& a_re;
	const mfem::GridFunction& a_im;
	double omega;
	std::vector<Drive> drives;
	std::vector<int> drive_of_attribute;
};
