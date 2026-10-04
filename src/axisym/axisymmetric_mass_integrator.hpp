// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include "mfem.hpp"
#include "axisymmetric_measure.hpp"
#include "radial_quadrature.hpp"

/**
 * @brief Axisymmetric Mass Integrator for Eddy Currents
 * Solves: Integral( sigma * A * v * 2*pi*r * dr * dz )
 * Used for the mass term in time-harmonic problems (j * omega * sigma * A)
 *
 * Carries the full axisymmetric measure (see axisymmetric_measure.hpp), matching
 * AxisymmetricCurlCurlIntegrator so K + j*omega*M_sigma is physical throughout.
 */
class AxisymmetricMassIntegrator : public mfem::BilinearFormIntegrator
{
private:
   mfem::Coefficient *Q; // Conductivity (sigma)
   mfem::real_t axis_tolerance_;

public:
   explicit AxisymmetricMassIntegrator(
      mfem::Coefficient &q, const mfem::IntegrationRule *ir = nullptr,
      mfem::real_t axis_tolerance = 0.0)
      : mfem::BilinearFormIntegrator(ir), Q(&q), axis_tolerance_(axis_tolerance)
   {
      MFEM_ASSERT(Q != nullptr, "Coefficient cannot be null");
   }

   static const mfem::IntegrationRule &GetRule(
      const mfem::FiniteElement &trial_fe,
      const mfem::FiniteElement &test_fe,
      const mfem::ElementTransformation &Trans)
   {
      if (trial_fe.GetDim() == 1) {
         return mfem::IntRules.Get(trial_fe.GetGeomType(),
            trial_fe.GetOrder() + test_fe.GetOrder() + Trans.Order() + Trans.OrderW());
      }
      return axisym::RadialRule(trial_fe, Trans);
   }

   void AssembleElementMatrix(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &Trans,
                               mfem::DenseMatrix &elmat) override
   {
      int nd = el.GetDof();
      elmat.SetSize(nd);
      elmat = 0.0;

      // pos is sized for (r,z) and the radius is read from pos(0).
      MFEM_VERIFY(Trans.GetSpaceDim() == 2,
         "AxisymmetricMassIntegrator requires a 2D (r,z) mesh.");

      mfem::Vector shape(nd);
      mfem::Vector pos(2);

      const mfem::IntegrationRule *ir = GetIntegrationRule(el, Trans);

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const mfem::IntegrationPoint &ip = ir->IntPoint(i);
         Trans.SetIntPoint(&ip);
         Trans.Transform(ip, pos);
         
         mfem::real_t r = pos(0);

         mfem::real_t val = Q->Eval(Trans, ip); // Conductivity sigma

         // Weight = w * det(J) * 2*pi*r * sigma
         mfem::real_t w = ip.weight * Trans.Weight() * Axisymmetric::Measure(r) * val;

         el.CalcShape(ip, shape);

         for (int j = 0; j < nd; j++)
         {
             for (int k = 0; k <= j; k++) // Symmetry
             {
                 mfem::real_t entry = w * shape(j) * shape(k);
                 elmat(j, k) += entry;
                 if (j != k) elmat(k, j) += entry;
             }
         }
      }
   }

protected:
   const mfem::IntegrationRule *GetDefaultIntegrationRule(
      const mfem::FiniteElement &trial_fe,
      const mfem::FiniteElement &test_fe,
      const mfem::ElementTransformation &Trans) const override
   {
      return trial_fe.GetDim() == 1 ? &GetRule(trial_fe, test_fe, Trans)
         : &axisym::RadialRule(trial_fe, Trans, axis_tolerance_);
   }
};
