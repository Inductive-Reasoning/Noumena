// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "mfem.hpp"
#include "../core/enums.hpp"
#include "../axisym/axisymmetric_boundary_lf_integrator.hpp"
#include "../axisym/axisymmetric_diffusion_integrator.hpp"
#include "../axisym/axisymmetric_lf_integrator.hpp"
#include "../axisym/axisymmetric_mass_integrator.hpp"
#include "../axisym/axisymmetric_measure.hpp"

/**
 * @brief Everything about a run that depends on the coordinate model alone.
 *
 * A thin value wrapper over GeometryType that answers, in one place, the
 * questions solvers used to answer with an `if (geometry == Axisymmetric)` at
 * each call site: which integrator carries the geometric measure, what that
 * measure is at a point, what mesh dimension the model needs, and whether the
 * extracted coupling quantities are absolute or per unit length.
 *
 * Every member is an exhaustive switch over GeometryType with no default, so
 * adding a coordinate model is a compile-time checklist (-Wswitch) rather than
 * a search for scattered branches.
 *
 * Scope: the SCALAR H1 forms shared by every current formulation (the
 * electrostatic potential and the 2D magnetic A_z / A_phi). Physics that only
 * one formulation owns -- the axisymmetric curl-curl operator and its axis
 * regularity, B recovery from a scalar potential, the azimuthal port drive
 * field -- stays with that formulation, because a 3D magnetic model replaces it
 * with a vector (H(curl)) discretization instead of adding another branch.
 *
 * Integration measure: Axisymmetric applies the full revolved measure 2*pi*r
 * (see axisymmetric_measure.hpp), Planar integrates the cross-section only
 * (a unit out-of-plane depth), and Cartesian3D integrates the true volume.
 * The Cartesian models use MFEM's stock integrators, which are dimension-
 * generic, so Planar and Cartesian3D differ only in required mesh dimension
 * and in output units.
 */
class GeometryModel {
public:
	explicit GeometryModel(GeometryType type) : type_(type) {}

	[[nodiscard]] GeometryType Type() const { return type_; }

	/// Canonical config string ("axisymmetric", "planar", "3d").
	[[nodiscard]] const char* Name() const { return ToString(type_); }

	/// Topological dimension the mesh must have for this model.
	[[nodiscard]] int MeshDimension() const {
		switch (type_) {
			case GeometryType::Axisymmetric: return 2;
			case GeometryType::Planar:       return 2;
			case GeometryType::Cartesian3D:  return 3;
		}
		return 0;
	}

	/// Whether assembled quantities are per unit out-of-plane length.
	///
	/// Planar assembly integrates over the (x, y) cross-section only, which is
	/// equivalent to a unit depth of a translationally invariant (infinitely
	/// long) structure, so every extracted coupling quantity is per metre. No
	/// extrusion length is configurable. Axisymmetric assembly carries the full
	/// revolved measure and 3D assembly the true volume, so both are absolute.
	[[nodiscard]] bool IsPerUnitLength() const {
		switch (type_) {
			case GeometryType::Axisymmetric: return false;
			case GeometryType::Planar:       return true;
			case GeometryType::Cartesian3D:  return false;
		}
		return false;
	}

	/// SI unit of an extracted coupling quantity under this model.
	[[nodiscard]] std::string CouplingUnits(const std::string& si_unit) const {
		return IsPerUnitLength() ? si_unit + "/m" : si_unit;
	}

	/// Geometric measure at physical point @p x, excluding the quadrature
	/// weight and Jacobian: 2*pi*r (r = x(0)) for Axisymmetric, 1 otherwise.
	[[nodiscard]] mfem::real_t Measure(const mfem::Vector& x) const {
		switch (type_) {
			case GeometryType::Axisymmetric: return Axisymmetric::Measure(x(0));
			case GeometryType::Planar:       return 1.0;
			case GeometryType::Cartesian3D:  return 1.0;
		}
		return 1.0;
	}

	// ---- Integrator factories -------------------------------------------------
	// Each returns a fresh heap instance carrying this model's measure; the
	// caller (normally a BilinearForm / LinearForm) takes ownership. Fresh
	// instances matter: the solve's form and the AMR estimator each own one.

	/// (Q grad u, grad v): electrostatic stiffness, planar magnetostatic
	/// stiffness.
	[[nodiscard]] mfem::BilinearFormIntegrator* NewDiffusionIntegrator(
		mfem::Coefficient& q) const {
		switch (type_) {
			case GeometryType::Axisymmetric: return new AxisymmetricDiffusionIntegrator(q);
			case GeometryType::Planar:       return new mfem::DiffusionIntegrator(q);
			case GeometryType::Cartesian3D:  return new mfem::DiffusionIntegrator(q);
		}
		return nullptr;
	}

	/// (Q u, v): the MQS conductivity mass term.
	[[nodiscard]] mfem::BilinearFormIntegrator* NewMassIntegrator(
		mfem::Coefficient& q) const {
		switch (type_) {
			case GeometryType::Axisymmetric: return new AxisymmetricMassIntegrator(q);
			case GeometryType::Planar:       return new mfem::MassIntegrator(q);
			case GeometryType::Cartesian3D:  return new mfem::MassIntegrator(q);
		}
		return nullptr;
	}

	/// (f, v) over the domain: volume sources and winding functionals.
	[[nodiscard]] mfem::LinearFormIntegrator* NewDomainLFIntegrator(
		mfem::Coefficient& f) const {
		switch (type_) {
			case GeometryType::Axisymmetric: return new AxisymmetricLFIntegrator(f);
			case GeometryType::Planar:       return new mfem::DomainLFIntegrator(f);
			case GeometryType::Cartesian3D:  return new mfem::DomainLFIntegrator(f);
		}
		return nullptr;
	}

	/// (Q u, v) over the boundary: the Robin term. The axisymmetric volume
	/// mass integrator doubles as the boundary one: it reads r from the SPACE
	/// dimension, so on a meridional boundary segment it applies 2*pi*r ds.
	[[nodiscard]] mfem::BilinearFormIntegrator* NewBoundaryMassIntegrator(
		mfem::Coefficient& q) const {
		switch (type_) {
			case GeometryType::Axisymmetric: return new AxisymmetricMassIntegrator(q);
			case GeometryType::Planar:       return new mfem::BoundaryMassIntegrator(q);
			case GeometryType::Cartesian3D:  return new mfem::BoundaryMassIntegrator(q);
		}
		return nullptr;
	}

	/// (g, v) over the boundary: prescribed Neumann flux and Robin data.
	[[nodiscard]] mfem::LinearFormIntegrator* NewBoundaryLFIntegrator(
		mfem::Coefficient& g) const {
		switch (type_) {
			case GeometryType::Axisymmetric: return new AxisymmetricBoundaryLFIntegrator(g);
			case GeometryType::Planar:       return new mfem::BoundaryLFIntegrator(g);
			case GeometryType::Cartesian3D:  return new mfem::BoundaryLFIntegrator(g);
		}
		return nullptr;
	}

	/// Abort unless @p mesh has the dimension this model needs. The config
	/// validator reports the same mismatch with more context; this is the
	/// guard for solvers constructed programmatically.
	void VerifyMeshDimension(const mfem::Mesh& mesh) const {
		MFEM_VERIFY(mesh.Dimension() == MeshDimension(),
			"geometry_type '" << Name() << "' requires a " << MeshDimension()
			<< "D mesh, but the mesh is " << mesh.Dimension() << "D.");
	}

private:
	GeometryType type_;
};
