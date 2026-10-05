// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include "mfem.hpp"
#include "axisymmetric_measure.hpp"

/**
 * @brief Integrates an axisymmetric natural boundary load.
 *
 * Assembles integral(q * v * 2*pi*r ds) on the meridional boundary, carrying the
 * full axisymmetric measure (see axisymmetric_measure.hpp) so the load pairs
 * correctly with the stiffness operator.
 */
class AxisymmetricBoundaryLFIntegrator : public mfem::LinearFormIntegrator
{
private:
	mfem::Coefficient* load;

public:
	explicit AxisymmetricBoundaryLFIntegrator(
		mfem::Coefficient& q, const mfem::IntegrationRule* ir = nullptr)
		: mfem::LinearFormIntegrator(ir), load(&q)
	{
		MFEM_ASSERT(load != nullptr, "Coefficient cannot be null");
	}

	void AssembleRHSElementVect(const mfem::FiniteElement& el,
								mfem::ElementTransformation& trans,
								mfem::Vector& elvect) override
	{
		const int ndof = el.GetDof();
		elvect.SetSize(ndof);
		elvect = 0.0;

		// A meridional boundary element is 1D, but it is embedded in the 2D
		// (r,z) plane; the radius comes from the SPACE dimension, not the
		// element dimension.
		MFEM_VERIFY(trans.GetSpaceDim() == 2,
			"AxisymmetricBoundaryLFIntegrator requires a 2D (r,z) mesh.");

		mfem::Vector shape(ndof);
		mfem::Vector pos(trans.GetSpaceDim());
		const mfem::IntegrationRule* ir = GetIntegrationRule(el, trans);

		for (int i = 0; i < ir->GetNPoints(); ++i)
		{
			const mfem::IntegrationPoint& ip = ir->IntPoint(i);
			trans.SetIntPoint(&ip);
			trans.Transform(ip, pos);

			const mfem::real_t r = pos(0);
			const mfem::real_t value = load->Eval(trans, ip);
			const mfem::real_t weight = ip.weight * trans.Weight()
				* Axisymmetric::Measure(r) * value;

			el.CalcShape(ip, shape);
			elvect.Add(weight, shape);
		}
	}

protected:
	const mfem::IntegrationRule* GetDefaultIntegrationRule(
		const mfem::FiniteElement& trial_fe,
		const mfem::FiniteElement&,
		const mfem::ElementTransformation& trans) const override
	{
		const int order = trial_fe.GetOrder() + trans.Order() + trans.OrderW();
		if (trial_fe.Space() == mfem::FunctionSpace::rQk)
		{
			return &mfem::RefinedIntRules.Get(trial_fe.GetGeomType(), order);
		}
		return &mfem::IntRules.Get(trial_fe.GetGeomType(), order);
	}
};
