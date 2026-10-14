// Copyright (c) 2026 T. C. Raymond
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include "mfem.hpp"
#include "axisymmetric_measure.hpp"
#include "axisymmetric_field_relations.hpp"
#include "radial_quadrature.hpp"

/**
 * @brief Axisymmetric curl-curl bilinear form integrator for magnetostatics
 *        with A = A_phi(r,z) e_phi.
 *
 * Assembles, with the full axisymmetric measure (see axisymmetric_measure.hpp):
 *
 *   integral nu * [dA_phi/dz * dv/dz + (dA_phi/dr + A_phi/r) * (dv/dr + v/r)] * 2*pi*r dr dz
 *
 * This follows from
 *
 *   curl(A_phi e_phi) = -dA_phi/dz e_r + (dA_phi/dr + A_phi/r) e_z.
 *
 * IMPORTANT:
 *   Enforce the essential BC A_phi = 0 on the symmetry axis r = 0 (regularity).
 *   This is required whenever the domain closure reaches r = 0; annular domains
 *   (r_min > 0) need no axis condition at all.
 *
 * Notes:
 *   - Assembly keeps its scratch storage local, but the axis-aware quadrature
 *     (axisym::RadialRule) mutates the element transformation and fills MFEM's
 *     global rule caches, so elements must not be assembled concurrently.
 *   - Assembly does NOT clamp r. Standard interior quadrature keeps r > 0 even for
 *     elements touching the axis, so a zero radius there signals a bad custom rule
 *     or a bad mesh and is reported rather than papered over. The 1/r term must be
 *     kept exact per basis function: individual shape functions need not vanish on
 *     the axis, so no per-basis limit exists. Regularity is a property of the
 *     constrained solution and is delivered by essential BC elimination.
 *   - Flux/energy recovery, which does see the constrained solution, uses the exact
 *     limit B_z -> 2 dA/dr on the axis (see ComputeElementFlux). Both that path and
 *     MagneticFieldCoefficient call axisym::AxialFluxDensity with the mesh's
 *     scale-relative axis tolerance, so recovery, postprocessing, and geometry
 *     classification cannot disagree about what counts as the axis.
 */
class AxisymmetricCurlCurlIntegrator : public mfem::BilinearFormIntegrator
{
public:
   // @param axis_tolerance  Scale-relative radius below which flux recovery takes
   //                        the axis limit; use axisym::AxisGeometry::tolerance so the
   //                        integrator shares the mesh's axis policy.
   explicit AxisymmetricCurlCurlIntegrator(
      mfem::Coefficient &reluctivity,
      mfem::real_t axis_tolerance,
      const mfem::IntegrationRule *ir = nullptr)
      : mfem::BilinearFormIntegrator(ir), nu_(&reluctivity),
        axis_tolerance_(axis_tolerance)
   {
      MFEM_ASSERT(nu_ != nullptr, "Reluctivity coefficient cannot be null");
   }

   static const mfem::IntegrationRule &GetRule(
      const mfem::FiniteElement &trial_fe,
      const mfem::FiniteElement &test_fe,
      const mfem::ElementTransformation &Trans,
      mfem::real_t axis_tolerance)
   {
      // Polynomial part: integrated exactly by the usual order estimate.
      const int gradient_order = Trans.OrderGrad(&trial_fe)
         + Trans.OrderGrad(&test_fe) + Trans.Order();
      const int radial_reaction_order = trial_fe.GetOrder()
         + test_fe.GetOrder() + Trans.Order() + Trans.OrderW();
      const int polynomial_order = std::max(gradient_order, radial_reaction_order);

      // The N_j N_k / r part: see radial_quadrature.hpp.
      return axisym::RadialRule(trial_fe.GetGeomType(), polynomial_order, Trans,
                   axis_tolerance);
   }

   void AssembleElementMatrix(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &Trans,
                              /*out*/ mfem::DenseMatrix &elmat) override
   {
      const int ndof  = el.GetDof();
      const int dim = el.GetDim();

      MFEM_VERIFY(dim == 2 && Trans.GetSpaceDim() == 2,
         "AxisymmetricCurlCurlIntegrator expects a 2D (r,z) finite element.");

      elmat.SetSize(ndof);
      elmat = 0.0;

      // Scratch storage, local to this call
      mfem::Vector      shape(ndof);
      mfem::DenseMatrix dshape_ref(ndof, dim);
      mfem::DenseMatrix dshape_phys(ndof, dim);
      mfem::Vector      pos(dim);

      const mfem::IntegrationRule *ir = GetIntegrationRule(el, Trans);

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const mfem::IntegrationPoint &ip = ir->IntPoint(i);
         Trans.SetIntPoint(&ip);

         // Physical coordinates: pos(0)=r, pos(1)=z
         Trans.Transform(ip, pos);
         const mfem::real_t r = pos(0);

         // Interior quadrature keeps r > 0 even for elements touching the axis.
         // A non-positive radius here means an invalid mesh or a custom rule
         // with boundary points; clamping would silently deform the geometry.
         MFEM_VERIFY(r > 0.0,
            "AxisymmetricCurlCurlIntegrator evaluated at a non-positive radius (r = "
            << r << "). The 1/r term is singular there: use an interior "
            "integration rule and a mesh with non-negative radii.");

         const mfem::real_t nu = nu_->Eval(Trans, ip);

         // Axisymmetric weight: ip.weight * detJ * 2*pi*r * nu
         const mfem::real_t w = ip.weight * Trans.Weight()
            * Axisymmetric::Measure(r) * nu;

         el.CalcShape(ip, /*out*/ shape);
         el.CalcDShape(ip, /*out*/ dshape_ref);

         // Map row-oriented reference derivatives to physical derivatives.
         // dshape_phys = dshape_ref * InvJ
         Mult(dshape_ref, Trans.InverseJacobian(), /*out*/ dshape_phys);

         for (int j = 0; j < ndof; j++)
         {
            const mfem::real_t Nj     = shape(j);
            const mfem::real_t dNj_dr = dshape_phys(j, 0);
            const mfem::real_t dNj_dz = dshape_phys(j, 1);

            for (int k = j; k < ndof; k++)
            {
               const mfem::real_t Nk     = shape(k);
               const mfem::real_t dNk_dr = dshape_phys(k, 0);
               const mfem::real_t dNk_dz = dshape_phys(k, 1);

               const mfem::real_t Br_j = -dNj_dz;
               const mfem::real_t Bz_j = dNj_dr + Nj / r;

               const mfem::real_t Br_k = -dNk_dz;
               const mfem::real_t Bz_k = dNk_dr + Nk / r;

               const mfem::real_t val = Br_j * Br_k + Bz_j * Bz_k;

               elmat(j, k) += w * val;
            }
         }
      }

      // Symmetrize
      for (int j = 0; j < ndof; j++)
      {
         for (int k = 0; k < j; k++)
         {
            elmat(j, k) = elmat(k, j);
         }
      }
   }

   void ComputeElementFlux(const mfem::FiniteElement& el,
       mfem::ElementTransformation& Trans,
       mfem::Vector& u,
       const mfem::FiniteElement& flux_elem,
       mfem::Vector& flux,
       bool with_coef = true,  // the base class's default
       const mfem::IntegrationRule* ir = nullptr) override
   {
       const int ndof = el.GetDof();
       const int dim = el.GetDim();
       const int space_dim = Trans.GetSpaceDim();

       MFEM_VERIFY(dim == 2 && space_dim == 2,
           "AxisymmetricCurlCurlIntegrator expects a 2D (r,z) mesh.");
       MFEM_ASSERT(u.Size() == ndof,
           "Element solution has an unexpected size.");

       if (!ir)
       {
           ir = &flux_elem.GetNodes();
       }

       const int flux_nd = ir->GetNPoints();
       flux.SetSize(flux_nd * space_dim);

       mfem::Vector shape(ndof);
       mfem::DenseMatrix dshape_ref(ndof, dim);
       // Inverse Jacobian: (dim x space_dim), NOT shaped like dshape.
       mfem::DenseMatrix inv_jacobian(dim, space_dim);
       mfem::Vector grad_ref(dim);
       mfem::Vector grad_phys(space_dim);
       mfem::Vector pos(space_dim);

       for (int i = 0; i < flux_nd; ++i)
       {
           const mfem::IntegrationPoint& ip = ir->IntPoint(i);
           Trans.SetIntPoint(&ip);

           el.CalcShape(ip, shape);
           el.CalcDShape(ip, dshape_ref);

           dshape_ref.MultTranspose(u, grad_ref);
           mfem::CalcInverse(Trans.Jacobian(), inv_jacobian);
           inv_jacobian.MultTranspose(grad_ref, grad_phys);

           Trans.Transform(ip, pos);
           const mfem::real_t r = pos(0);
           const mfem::real_t A_phi = shape * u;

           // curl(A_phi e_phi) in the (r,z) component ordering, with the axis
           // limit applied under the shared scale-relative tolerance.
           mfem::real_t B_r = -grad_phys(1);
           mfem::real_t B_z = axisym::AxialFluxDensity(A_phi, grad_phys(0), r, axis_tolerance_);

           if (with_coef)
           {
               const mfem::real_t nu = nu_->Eval(Trans, ip);
               B_r *= nu;
               B_z *= nu;
           }

           // MFEM vector GridFunction element data is component-major.
           flux(i) = B_r;
           flux(flux_nd + i) = B_z;
       }
   }

   // Returns mfem::real_t to match the base class declaration exactly; see the
   // note on AxisymmetricDiffusionIntegrator::ComputeFluxEnergy for why a
   // double here would silently stop overriding in a single-precision build.
   //
   // The energy is integral nu |flux|^2 r: it takes the flux of B, as from
   // ComputeElementFlux with with_coef = false, the convention of MFEM's
   // DiffusionIntegrator (Q |flux|^2), so an estimator using it calls
   // SetWithCoeff(false). Anisotropic estimation is not supported: d_energy
   // comes back zero.
   mfem::real_t ComputeFluxEnergy(const mfem::FiniteElement& flux_elem,
       mfem::ElementTransformation& Trans,
       mfem::Vector& flux,
       mfem::Vector* d_energy = nullptr) override
   {
       const int nd = flux_elem.GetDof();
       const int space_dim = Trans.GetSpaceDim();

       MFEM_VERIFY(space_dim == 2,
           "AxisymmetricCurlCurlIntegrator expects a 2D (r,z) mesh.");
       MFEM_ASSERT(flux.Size() == nd * space_dim,
           "Flux vector has an unexpected size.");

       mfem::Vector shape(nd);
       mfem::Vector point_flux(space_dim);
       mfem::Vector pos(space_dim);

       const int order = 2 * flux_elem.GetOrder()
           + Trans.Order() + Trans.OrderW();

       const mfem::IntegrationRule* ir = GetIntRule();
       if (!ir)
       {
           ir = &mfem::IntRules.Get(flux_elem.GetGeomType(), order);
       }

       if (d_energy)
       {
           d_energy->SetSize(space_dim);
           *d_energy = 0.0;
       }

       mfem::real_t energy = 0.0;

       for (int i = 0; i < ir->GetNPoints(); ++i)
       {
           const mfem::IntegrationPoint& ip = ir->IntPoint(i);
           Trans.SetIntPoint(&ip);

           flux_elem.CalcPhysShape(Trans, shape);

           point_flux = 0.0;
           for (int component = 0; component < space_dim; ++component)
           {
               for (int j = 0; j < nd; ++j)
               {
                   point_flux(component) +=
                       flux(component * nd + j) * shape(j);
               }
           }

           Trans.Transform(ip, pos);
           // No clamp needed: this term carries the measure r, never 1/r, so
           // the contribution simply vanishes on the axis.
           const mfem::real_t r = pos(0);
           const mfem::real_t nu = nu_->Eval(Trans, ip);
           const mfem::real_t weight = ip.weight * Trans.Weight()
               * Axisymmetric::Measure(r);

           energy += weight * nu * (point_flux * point_flux);
       }

       return energy;
   }

protected:
   const mfem::IntegrationRule *GetDefaultIntegrationRule(
      const mfem::FiniteElement &trial_fe,
      const mfem::FiniteElement &test_fe,
      const mfem::ElementTransformation &Trans) const override
   {
      return &GetRule(trial_fe, test_fe, Trans, axis_tolerance_);
   }

private:
   mfem::Coefficient *nu_;
   mfem::real_t axis_tolerance_;
};

