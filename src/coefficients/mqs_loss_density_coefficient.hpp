// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

/**
 * @file mqs_loss_density_coefficient.hpp
 * @brief Time-averaged Joule (eddy-current) loss density for the
 *        magnetoquasistatic formulation.
 */

#pragma once

#include <vector>

#include "mfem.hpp"
#include "../core/constants.hpp"

/**
 * @brief Time-averaged Joule loss density \f$P = \frac{1}{2}\sigma|E|^2\f$
 *        with \f$E = E_{drive} - j\omega A\f$.
 *
 * The factor \f$\frac{1}{2}\f$ is the time average of a product of
 * peak-amplitude phasors. This peak (amplitude) convention matches the
 * coupling-matrix extraction \f$R = \mathrm{Re}(V/I)\f$ used elsewhere in this
 * project, and the convention of the common commercial AC/DC and eddy-current
 * tools. Passing RMS phasors instead would apply the averaging twice and halve
 * every reported loss.
 *
 * @warning The convention is established by the prescribed excitation value
 * (see Excitation in problem_config.hpp) and is carried unscaled through the
 * RHS into the solved potentials and port voltages. It is documented but NOT
 * enforced: no code validates or converts it. An RMS excitation -- the natural
 * reading of a nameplate current -- is silently accepted and halves every
 * number this coefficient produces. See docs/faq.md, "Are excitations peak or
 * RMS?", for the user-facing statement of this hazard.
 *
 * Splitting \f$E\f$ into real and imaginary parts with
 * \f$A = A_{re} + jA_{im}\f$, so that
 * \f$-j\omega A = \omega A_{im} - j\omega A_{re}\f$, gives
 * \f[
 *   \mathrm{Re}\,E = E_{drive,re} + \omega A_{im}, \qquad
 *   \mathrm{Im}\,E = E_{drive,im} - \omega A_{re}.
 * \f]
 *
 * @par Why the drive term cannot be dropped
 * The simplification \f$P = \frac{1}{2}\sigma\omega^2|A|^2\f$ is valid ONLY
 * where \f$E_{drive} = 0\f$. That holds for conductive regions with no port
 * unknown (a flux shield or a steel brace), but NOT for massive terminals or
 * open-current regions, whose solved port voltage is exactly what enforces
 * their current constraint. Using the simplified form there silently under- or
 * over-reports the loss, so the drive term is carried explicitly.
 *
 * The per-attribute drive table is zero-initialised and populated only where a
 * port unknown exists. Unported conductors therefore fall out of the general
 * expression with no special case, which is why this one coefficient serves
 * driven terminals, open-current regions, and passive conductors alike.
 *
 * @par Geometry
 * Geometry enters in one place only. For an azimuthal massive conductor the
 * drive field produced by a port voltage \f$V\f$ is
 * \f$E_\phi = V/(2\pi r)\f$, the same physical \f$1/r\f$ that appears in
 * AxisymmetricConductanceCoeff. That factor is a property of the field, and is
 * entirely distinct from the \f$2\pi r\f$ volume measure supplied by the
 * integrator when this density is integrated. Integrating this coefficient
 * with a planar integrator on an axisymmetric problem therefore omits that
 * measure and yields a per-radian, not a total, loss.
 *
 * @par Assumed background
 * The vector potential is assumed to be the azimuthal component \f$A_\phi\f$ of
 * a time-harmonic solution at the single frequency @a omega, in the same peak
 * phasor convention as the drive amplitudes. The induced field
 * \f$-j\omega A\f$ is only the whole story because the formulation is
 * quasistatic: displacement current is neglected.
 *
 * @par Limitations
 * - Linear, isotropic, non-hysteretic conduction only. This is resistive
 *   dissipation alone; hysteresis and excess (anomalous) losses are not
 *   represented, so total loss in laminated steel is underestimated.
 * - Single-frequency only. The result is meaningless for a transient solution
 *   or for a superposition of several frequencies.
 * - Axisymmetric mode assumes an azimuthal conductor on an \f$(r,z)\f$ mesh and
 *   reads the radius from component 0 of the transformed point.
 * - A driven azimuthal conductor may not touch the symmetry axis; see Eval().
 *
 * @par Ownership and lifetime
 * @a sigma, @a a_re and @a a_im are held by reference and are NOT owned. The
 * caller must keep all three alive for the lifetime of this coefficient, and
 * must rebuild this object if a mesh refinement invalidates the referenced
 * GridFunction objects. The two drive tables are copied and are self-owned.
 */
class MqsLossDensityCoefficient : public mfem::Coefficient {
public:
	/**
	 * @brief Construct the loss-density coefficient.
	 *
	 * @param sigma        Piecewise conductivity \f$\sigma\f$ [S/m]. Also
	 *                     selects which regions dissipate at all:
	 *                     \f$\sigma \le 0\f$ yields exactly zero. Referenced,
	 *                     not owned; must outlive this object.
	 * @param a_re         Real part of the vector potential. Referenced, not
	 *                     owned; must outlive this object.
	 * @param a_im         Imaginary part of the vector potential. Referenced,
	 *                     not owned; must outlive this object.
	 * @param omega        Angular frequency \f$\omega\f$ [rad/s].
	 * @param drive_re     Per-attribute real part of the drive amplitude,
	 *                     indexed by (attribute - 1). Zero for unported
	 *                     regions. Copied into the object.
	 * @param drive_im     Per-attribute imaginary part, same indexing and
	 *                     ownership. Must match @a drive_re in size.
	 * @param axisymmetric When true the drive amplitude carries the physical
	 *                     \f$1/(2\pi r)\f$ factor described above.
	 */
	MqsLossDensityCoefficient(mfem::Coefficient& sigma,
							  const mfem::GridFunction& a_re,
							  const mfem::GridFunction& a_im,
							  double omega,
							  std::vector<double> drive_re,
							  std::vector<double> drive_im,
							  bool axisymmetric)
		: sigma_(sigma), a_re_(a_re), a_im_(a_im), omega_(omega),
		  drive_re_(std::move(drive_re)), drive_im_(std::move(drive_im)),
		  axisymmetric_(axisymmetric) {
		MFEM_ASSERT(drive_re_.size() == drive_im_.size(),
					"Drive amplitude tables must have matching sizes");
	}

	/**
	 * @brief Evaluate the time-averaged loss density at an integration point.
	 *
	 * @param T  Element transformation; its integration point is set here, and
	 *           its Attribute selects the drive-table entry.
	 * @param ip Integration point in reference coordinates.
	 * @return Loss density \f$\frac{1}{2}\sigma|E|^2\f$ [W/m^3], or exactly
	 *         zero in a non-conducting region.
	 *
	 * @note Returns double rather than mfem::real_t to match the base class
	 *       declaration in the default build. If this project is ever built
	 *       with MFEM_USE_SINGLE, this signature must become mfem::real_t or it
	 *       will silently stop overriding Coefficient::Eval.
	 */
	mfem::real_t Eval(mfem::ElementTransformation& T,
				const mfem::IntegrationPoint& ip) override {
		T.SetIntPoint(&ip);

		const double sigma = sigma_.Eval(T, ip);
		// Non-conducting regions dissipate nothing. Returning early also keeps
		// the axis guard below from firing on regions that cannot dissipate.
		if (sigma <= 0.0) { return 0.0; }

		double e_re = 0.0;
		double e_im = 0.0;
		DriveField(T, ip, e_re, e_im);

		e_re += omega_ * a_im_.GetValue(T, ip);
		e_im -= omega_ * a_re_.GetValue(T, ip);

		return 0.5 * sigma * (e_re * e_re + e_im * e_im);
	}

private:
	/**
	 * @brief Look up the drive field for the element's attribute.
	 *
	 * Yields zero where no port unknown exists. Attributes are 1-based in
	 * MFEM; an attribute beyond the table is treated as unported rather than
	 * as an error, since the table is sized to the mesh and a larger attribute
	 * cannot carry a port.
	 *
	 * @param      T    Element transformation, supplying both the attribute and
	 *                  (in axisymmetric mode) the radius.
	 * @param      ip   Integration point in reference coordinates.
	 * @param[out] e_re Real part of the drive field [V/m]; set to zero when the
	 *                  region is unported.
	 * @param[out] e_im Imaginary part of the drive field [V/m].
	 *
	 * @warning In axisymmetric mode this asserts \f$r > 0\f$ for a ported
	 *          region, because \f$E_\phi = V/(2\pi r)\f$ is singular on the
	 *          axis. A massive port reaching the axis is already rejected at
	 *          assembly time (its DC conductance integral diverges), so
	 *          reaching that check means the earlier guard was bypassed.
	 */
	void DriveField(mfem::ElementTransformation& T,
					const mfem::IntegrationPoint& ip,
					double& e_re, double& e_im) const {
		const int index = T.Attribute - 1;
		if (index < 0 || index >= static_cast<int>(drive_re_.size())) { return; }

		e_re = drive_re_[index];
		e_im = drive_im_[index];
		if (e_re == 0.0 && e_im == 0.0) { return; }

		if (!axisymmetric_) { return; }

		mfem::Vector pos;
		T.Transform(ip, pos);
		const double r = pos(0);
		// Only reachable for a ported region, and a massive port that reaches
		// the axis is rejected at assembly time because its DC conductance
		// integral diverges. Hitting this means that guard was bypassed.
		MFEM_VERIFY(r > 0.0,
			"Axisymmetric drive field V/(2*pi*r) is singular at r = "
			<< r << ". A driven azimuthal conductor cannot touch the "
			"symmetry axis.");

		const double scale = 1.0 / (Constants::TWO_PI * r);
		e_re *= scale;
		e_im *= scale;
	}

	/// Piecewise conductivity [S/m]. Referenced, not owned.
	mfem::Coefficient& sigma_;
	/// Real part of the vector potential. Referenced, not owned.
	const mfem::GridFunction& a_re_;
	/// Imaginary part of the vector potential. Referenced, not owned.
	const mfem::GridFunction& a_im_;
	/// Angular frequency [rad/s].
	double omega_;
	/// Real part of the drive amplitude by (attribute - 1). Owned copy.
	std::vector<double> drive_re_;
	/// Imaginary part of the drive amplitude by (attribute - 1). Owned copy.
	std::vector<double> drive_im_;
	/// When true, the drive amplitude carries the 1/(2*pi*r) field factor.
	bool axisymmetric_;
};
