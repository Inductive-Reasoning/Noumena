// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#include "axisymmetric_diffusion_integrator.hpp"

 const mfem::IntegrationRule &AxisymmetricDiffusionIntegrator::GetRule(
   const mfem::FiniteElement &trial_fe,
   const mfem::FiniteElement &test_fe,
   const mfem::ElementTransformation &Trans)
{
   // Curved inverse mappings make the transformed-gradient integrand
   // rational, so this is a conservative order estimate.
   const int order = Trans.OrderGrad(&trial_fe)
      + Trans.OrderGrad(&test_fe) + Trans.Order();

   if (trial_fe.Space() == mfem::FunctionSpace::rQk)
   {
      return mfem::RefinedIntRules.Get(trial_fe.GetGeomType(), order);
   }
   return mfem::IntRules.Get(trial_fe.GetGeomType(), order);
}

void AxisymmetricDiffusionIntegrator::AssembleElementMatrix(const mfem::FiniteElement &el,
                            mfem::ElementTransformation &Trans,
                            mfem::DenseMatrix &elmat) 
{
   int nd = el.GetDof();
   int dim = el.GetDim();
   mfem::real_t w;

   // The r coordinate is read from pos(0), so a non-2D element would index
   // a vector sized for (r,z). MFEM_VERIFY rather than MFEM_ASSERT: this
   // must hold in release builds too, where an assert compiles away.
   MFEM_VERIFY(Trans.GetSpaceDim() == 2,
      "AxisymmetricDiffusionIntegrator requires a 2D (r,z) mesh.");

   elmat.SetSize(nd);
   elmat = 0.0;

   mfem::DenseMatrix dshape(nd, dim);
   mfem::DenseMatrix dshapedxt(nd, dim);
   mfem::Vector pos(2); // 2D axisymmetric

   const mfem::IntegrationRule *ir = GetIntegrationRule(el, Trans);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const mfem::IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);
      Trans.Transform(ip, pos);

      mfem::real_t r = pos(0); // Radius is X

      // Weight = quad_weight * det(J) * 2*pi*r * epsilon
      w = ip.weight * Trans.Weight() * Axisymmetric::Measure(r)
         * Q->Eval(Trans, ip);

      el.CalcDShape(ip, dshape);
      // dN/ds * J^-1
      Mult(dshape, Trans.InverseJacobian(), dshapedxt);
      // Integral of grad u dot grad v 2*pi*r dr dz
      AddMult_a_AAt(w, dshapedxt, elmat);
   }
}

void AxisymmetricDiffusionIntegrator::ComputeElementFlux(const mfem::FiniteElement &el,
                        mfem::ElementTransformation &Trans,
                        mfem::Vector &u,
                        const mfem::FiniteElement &fluxelem,
                        mfem::Vector &flux, bool with_coef,
                        const mfem::IntegrationRule *ir) 
{
   const int nd = el.GetDof();
   const int dim = el.GetDim();
   const int spaceDim = Trans.GetSpaceDim();

   mfem::DenseMatrix dshape(nd, dim);
   mfem::DenseMatrix invdfdx(dim, spaceDim);
   mfem::Vector vec(dim);
   mfem::Vector vecdxt(spaceDim);

   if (!ir)
   {
      ir = &fluxelem.GetNodes();
   }
   const int fnd = ir->GetNPoints();
   flux.SetSize(fnd * spaceDim);

   for (int i = 0; i < fnd; i++)
   {
      const mfem::IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);
      dshape.MultTranspose(u, vec);

      Trans.SetIntPoint(&ip);
      mfem::CalcInverse(Trans.Jacobian(), invdfdx);
      invdfdx.MultTranspose(vec, vecdxt);

      if (with_coef)
      {
         vecdxt *= Q->Eval(Trans, ip); // flux = eps * grad(u)
      }
      for (int j = 0; j < spaceDim; j++)
      {
         flux(fnd * j + i) = vecdxt(j);
      }
   }
}

mfem::real_t AxisymmetricDiffusionIntegrator::ComputeFluxEnergy(const mfem::FiniteElement &fluxelem,
                         mfem::ElementTransformation &Trans,
                         mfem::Vector &flux,
                         mfem::Vector *d_energy) 
{
   const int nd = fluxelem.GetDof();
   const int spaceDim = Trans.GetSpaceDim();

   MFEM_VERIFY(spaceDim == 2,
      "AxisymmetricDiffusionIntegrator requires a 2D (r,z) mesh.");

   mfem::Vector shape(nd);
   mfem::Vector pointflux(spaceDim);
   mfem::Vector pos(2); // 2D axisymmetric

   const int order = 2 * fluxelem.GetOrder()
      + Trans.Order() + Trans.OrderW();
   const mfem::IntegrationRule *ir = GetIntRule();
   if (!ir)
   {
      ir = &mfem::IntRules.Get(fluxelem.GetGeomType(), order);
   }

   mfem::real_t energy = 0.0;
   if (d_energy) { *d_energy = 0.0; } // anisotropic estimation unsupported

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const mfem::IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);
      fluxelem.CalcPhysShape(Trans, shape);

      pointflux = 0.0;
      for (int k = 0; k < spaceDim; k++)
      {
         for (int j = 0; j < nd; j++)
         {
            pointflux(k) += flux(k * nd + j) * shape(j);
         }
      }

      Trans.Transform(ip, pos);
      const mfem::real_t r = pos(0); // Radius is X

      // Axisymmetric measure, consistent with AssembleElementMatrix.
      const mfem::real_t w = Trans.Weight() * ip.weight * Axisymmetric::Measure(r);

      mfem::real_t e = (pointflux * pointflux);
      e *= Q->Eval(Trans, ip); // eps
      energy += w * e;
   }

   return energy;
}
