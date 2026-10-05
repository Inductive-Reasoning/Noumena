// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <memory>
#include <vector>

#include "mfem.hpp"
#include "conductor_path.hpp"

/**
 * @brief The electric field E = V w - j omega A of a 3D time-harmonic
 *        solution, and the quantities derived from it in the conductors.
 *
 * A massive conductor with port voltage V is driven by the field V w of its
 * ConductorPath; every other conductor -- a passive shield, a brace -- has
 * only the induced field -j omega A. A is the Nedelec potential of the
 * formulation, whose gradient part in the conductors carries their
 * charge-free electric scalar potential, so E is complete there. (Outside
 * the conductors A is gauged, not physical, and so is this E.)
 *
 * With A = A_re + j A_im and V = V_re + j V_im,
 *     Re E = V_re w + omega A_im,    Im E = V_im w - omega A_re.
 *
 * The potentials and the paths are referenced, not owned.
 */
class MqsVectorElectricField {
public:
	/// A massive conductor's drive: its path and solved port voltage.
	struct Drive {
		const ConductorPath* Path = nullptr;
		std::complex<double> Voltage;
	};

	/// @param drive_of_attribute  Index into @p drives by (attribute - 1), or
	///                            -1 where no port drives the conductor.
	MqsVectorElectricField(const mfem::GridFunction& a_re, const mfem::GridFunction& a_im,
						   double omega, std::vector<Drive> drives,
						   std::vector<int> drive_of_attribute)
		: a_re(a_re), a_im(a_im), omega(omega), drives(std::move(drives)),
		  drive_of_attribute(std::move(drive_of_attribute)) {}

	void Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip,
			  mfem::Vector& e_re, mfem::Vector& e_im) const {
		T.SetIntPoint(&ip);
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
	}

private:
	const mfem::GridFunction& a_re;
	const mfem::GridFunction& a_im;
	double omega;
	std::vector<Drive> drives;
	std::vector<int> drive_of_attribute;
};

/**
 * @brief Time-averaged Joule loss density P = 1/2 sigma |E|^2, with the
 *        peak-phasor convention of MqsLossDensityCoefficient.
 */
class MqsVectorLossDensityCoefficient : public mfem::Coefficient {
public:
	MqsVectorLossDensityCoefficient(mfem::Coefficient& sigma,
									std::shared_ptr<const MqsVectorElectricField> field)
		: sigma(sigma), field(std::move(field)) {}

	double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
		T.SetIntPoint(&ip);
		const double s = sigma.Eval(T, ip);
		if (s <= 0.0) { return 0.0; }
		mfem::Vector e_re, e_im;
		field->Eval(T, ip, e_re, e_im);
		return 0.5 * s * (e_re * e_re + e_im * e_im);
	}

private:
	mfem::Coefficient& sigma;
	std::shared_ptr<const MqsVectorElectricField> field;
};

/**
 * @brief One part (real or imaginary) of the conduction current density
 *        J = sigma E: eddy current in every conductor, plus the driven
 *        current of a massive one. Zero outside the conductors, and in the
 *        stranded ones, whose prescribed source current is not a field of the
 *        solution.
 */
class MqsVectorCurrentDensityCoefficient : public mfem::VectorCoefficient {
public:
	enum class Part { Real, Imag };

	MqsVectorCurrentDensityCoefficient(mfem::Coefficient& sigma,
									   std::shared_ptr<const MqsVectorElectricField> field,
									   Part part)
		: mfem::VectorCoefficient(3), sigma(sigma), field(std::move(field)), part(part) {}

	void Eval(mfem::Vector& j, mfem::ElementTransformation& T,
			  const mfem::IntegrationPoint& ip) override {
		T.SetIntPoint(&ip);
		j.SetSize(3);
		j = 0.0;
		const double s = sigma.Eval(T, ip);
		if (s <= 0.0) { return; }
		mfem::Vector e_re, e_im;
		field->Eval(T, ip, e_re, e_im);
		j.Set(s, part == Part::Real ? e_re : e_im);
	}

private:
	mfem::Coefficient& sigma;
	std::shared_ptr<const MqsVectorElectricField> field;
	Part part;
};
